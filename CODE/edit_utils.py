import json
import re
import random
import torch
import logging
import gc
import shutil
import uuid
import datetime
from copy import deepcopy
from typing import List, Dict, Union, Any, Optional
from contextlib import contextmanager
from pathlib import Path
from tqdm import tqdm
from time import time
from peft import LoraConfig, TaskType, get_peft_model
from EasyEdit.easyeditor.util import nethook
from datasets import Dataset
from transformers import (
    TrainingArguments,
    Trainer,
    StoppingCriteria,
    StoppingCriteriaList,
)
import os

from vllm import SamplingParams, TokensPrompt

from unsloth import FastLanguageModel


# Get logger instance - use root logger to ensure config is inherited
logger = logging.getLogger("mine")


BATCH_LORA_ARTIFACT_METHODS = {"CAKE", "CausalEdit"}
BATCH_WEIGHT_PATCH_METHODS = {"MEMIT", "EMMET", "AlphaEdit"}


@contextmanager
def _temporary_unsloth_lora_request(model):
    model.set_adapter("default")
    FastLanguageModel.for_inference(model)

    device_tag = os.environ.get("CUDA_VISIBLE_DEVICES", "0").replace(",", "") or "0"
    temp_dir = (
        Path.cwd()
        / f".tmp_edit_utils_eval_lora_{device_tag}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    )
    temp_dir.mkdir(parents=True, exist_ok=False)

    try:
        model.save_lora(str(temp_dir))
        lora_request = model.load_lora(str(temp_dir))
        yield lora_request
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def save_batch_edit_artifacts(
    edited_model,
    edit_item: Dict[str, Any],
    hparams,
    alg_name: str,
    all_eval_items: Optional[List[Dict[str, Any]]] = None,
    weights_copy: Optional[Dict[str, torch.Tensor]] = None,
) -> Optional[Path]:
    """
    Save batch edit artifacts (LoRA adapter or changed_weights) to disk.

    Automatically selects the storage format based on the editing method type:
    - LoRA-based methods (CausalEdit, CAKE, etc.): save adapter to lora_adapter/ directory
    - Weight-patch methods (AlphaEdit, MEMIT, EMMET, etc.): save changed_weights.pt

    Args:
        edited_model: The edited model (or model containing the adapter)
        edit_item: Edit item containing requested_rewrite
        hparams: Hyperparameter configuration
        alg_name: Editing method name
        all_eval_items: List of all evaluation cases
        weights_copy: Original weight backup (for computing changed_weights)

    Returns:
        Artifact save directory path, or None if save_dir is missing
    """

    def _is_lora_like_batch_method(alg_name: str) -> bool:
        return alg_name in BATCH_LORA_ARTIFACT_METHODS

    def _is_weight_patch_batch_method(alg_name: str) -> bool:
        return alg_name in BATCH_WEIGHT_PATCH_METHODS

    def _sanitize_artifact_method_name(alg_name: str) -> str:
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(alg_name).strip())
        return safe_name or "unknown_method"

    def _resolve_batch_artifact_dir(
        edit_item: Dict[str, Any], alg_name: str
    ) -> Optional[Path]:
        requests = edit_item.get("requested_rewrite", [])
        if not requests:
            return None
        save_dir = requests[0].get("save_dir")
        if not save_dir:
            return None
        artifact_dir = (
            Path(save_dir)
            / "batch_edit_artifacts"
            / _sanitize_artifact_method_name(alg_name)
        )
        return artifact_dir

    def _json_safe_value(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): _json_safe_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_json_safe_value(v) for v in value]
        return value

    def _collect_changed_weights(
        edited_model,
        weights_copy: Optional[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        if not weights_copy:
            return {}

        changed_weights = {}
        with torch.no_grad():
            for weight_name in weights_copy.keys():
                param = nethook.get_parameter(edited_model, weight_name)
                changed_weights[weight_name] = param.detach().cpu().clone()
        return changed_weights

    def _save_lora_adapter(model, adapter_dir: Path) -> None:
        """
        Persist LoRA adapter weights for both plain PEFT models and Unsloth-wrapped
        models. Unsloth exposes `save_lora`, while regular PEFT models rely on
        `save_pretrained`.
        """
        if hasattr(model, "save_lora"):
            model.save_lora(str(adapter_dir))
            return
        model.save_pretrained(str(adapter_dir))

    def _write_batch_artifact_manifest(
        artifact_dir: Path,
        manifest: Dict[str, Any],
        requests: List[Dict[str, Any]],
    ) -> None:
        with open(artifact_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(_json_safe_value(manifest), f, ensure_ascii=False, indent=2)
        with open(artifact_dir / "requests.json", "w", encoding="utf-8") as f:
            json.dump(_json_safe_value(requests), f, ensure_ascii=False, indent=2)

    artifact_dir = _resolve_batch_artifact_dir(edit_item, alg_name)
    if artifact_dir is None:
        logger.warning("Skip saving batch edit artifacts because save_dir is missing.")
        return None

    requests = deepcopy(edit_item.get("requested_rewrite", []))
    created_at = datetime.datetime.now().isoformat(timespec="seconds")
    manifest = {
        "format_version": 1,
        "editing_method": alg_name,
        "model_name": getattr(hparams, "model_name", None),
        "batch_case_id": edit_item.get("case_id"),
        "num_requests": len(requests),
        "num_eval_cases": len(all_eval_items or []),
        "created_at": created_at,
        "artifact_method_dir": artifact_dir.name,
    }

    if _is_lora_like_batch_method(alg_name):
        artifact_dir.mkdir(parents=True, exist_ok=True)
        adapter_subdir = "lora_adapter"
        adapter_dir = artifact_dir / adapter_subdir
        adapter_dir.mkdir(parents=True, exist_ok=True)
        _save_lora_adapter(edited_model, adapter_dir)
        manifest.update(
            {
                "storage_type": "lora_adapter",
                "adapter_subdir": adapter_subdir,
            }
        )
    elif _is_weight_patch_batch_method(alg_name):
        artifact_dir.mkdir(parents=True, exist_ok=True)
        changed_weights = _collect_changed_weights(edited_model, weights_copy)
        changed_weights_path = artifact_dir / "changed_weights.pt"
        torch.save(changed_weights, changed_weights_path)
        manifest.update(
            {
                "storage_type": "changed_weights",
                "changed_weights_file": changed_weights_path.name,
                "weight_names": sorted(changed_weights.keys()),
                "weight_count": len(changed_weights),
            }
        )
    else:
        manifest.update({"storage_type": "unsupported"})
        logger.warning(
            "Batch edit artifact saving is not configured for method `%s`.", alg_name
        )
        return None

    _write_batch_artifact_manifest(artifact_dir, manifest, requests)
    logger.info("Batch edit artifacts saved to: %s", artifact_dir)
    return artifact_dir


def filter_data(
    data: List[Dict[str, Any]],
    datatype: Optional[str] = None,
    exclude_case_ids: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Filter data by case_id

    Args:
        data: Original data list, each element contains case_id
        datatype: Dataset type, used to get default exclusion list
        exclude_case_ids: List of case_ids to exclude; if None, gets default based on datatype

    Returns:
        Filtered data list
    """
    DATASET_EXCLUDE_CASE_IDS = {
        "MQuAKE-CF-3k-v2": [35],
    }
    if exclude_case_ids is None:
        exclude_case_ids = DATASET_EXCLUDE_CASE_IDS.get(datatype, [])
    filtered = [
        item for item in data if int(item.get("case_id")) not in exclude_case_ids
    ]
    logger.info(f"Data count before filtering: {len(data)}, after filtering: {len(filtered)}")
    return filtered


def shuffle_data(
    data: List[Dict[str, Any]], seed: int = 42, hop: Optional[int] = None
) -> tuple[List[Dict[str, Any]], Dict[int, List[Dict[str, Any]]]]:
    """
    Group by hop count, shuffle within each group, then organize data in round-robin order [2,3,4,2,3,4,...]

    Args:
        data: Original data list
        seed: Random seed, default 42

    Returns:
        tuple: (shuffled data list, hop count group dictionary)
            - Shuffled data organized in round-robin order, e.g. [2-hop, 3-hop, 4-hop, 2-hop, 3-hop, 4-hop, ...]
            - Group dictionary: {2: [2-hop data...], 3: [3-hop data...], 4: [4-hop data...]}
    """
    from collections import defaultdict

    hop_groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for item in data:
        hop_count = len(item.get("new_single_hops", []))
        hop_groups[hop_count].append(item)

    state = random.getstate()
    random.seed(seed)
    for hop_count in hop_groups:
        random.shuffle(hop_groups[hop_count])
    random.setstate(state)

    sorted_hops = sorted(hop_groups.keys())
    result = []
    max_len = max(len(hop_groups[h]) for h in sorted_hops)
    for i in range(max_len):
        for hop_count in sorted_hops:
            if i < len(hop_groups[hop_count]):
                result.append(hop_groups[hop_count][i])

    logger.info(f"Data shuffled by hop groups in round-robin order, seed: {seed}")
    for hop_count in sorted_hops:
        logger.info(f"  {hop_count}-hop: {len(hop_groups[hop_count])} cases")

    if hop is None:
        logger.info(f"All hops data count: {len(result)} cases, testing all hops")
        return result, dict(hop_groups)

    selected = hop_groups[hop]
    logger.info(f" {hop}-hop data count: {len(selected)}, testing {hop}-hop data")
    return selected, dict(hop_groups)


def load_dataset(
    datatype: str,
    suffix: str = "causalenhanced",
    data_dir: str = "./datasets",
) -> List[Dict[str, Any]]:
    """
    Load dataset

    Args:
        datatype: Dataset type, e.g. "MQuAKE-CF-3k-v2"
        suffix: Dataset suffix, e.g. "causalenhanced", "cake"
        data_dir: Dataset directory, default "./datasets"

    Returns:
        Data list
    """
    dataset_path = os.path.join(data_dir, f"{datatype}_{suffix}.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded dataset: {dataset_path}, data count: {len(data)}")
    return data


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def check_answer_in_pred(pred, answers):
    pred = pred.lower()
    return any([a.lower() in pred for a in answers])


def get_sent_embeddings(sents, contriever, tok, device, BSZ=32):
    all_embs = []
    for i in tqdm(range(0, len(sents), BSZ)):
        sent_batch = sents[i : i + BSZ]
        inputs = tok(sent_batch, padding=True, truncation=True, return_tensors="pt").to(
            device
        )
        with torch.no_grad():
            outputs = contriever(**inputs)
            embeddings = mean_pooling(outputs[0], inputs["attention_mask"])
        all_embs.append(embeddings.cpu())
    all_embs = torch.vstack(all_embs)
    return all_embs


def retrieve_facts(query, fact_embs, contriever, tok, device, k=1):
    inputs = tok([query], padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = contriever(**inputs)
        query_emb = mean_pooling(outputs[0], inputs["attention_mask"]).cpu()
    sim = (query_emb @ fact_embs.T)[0]
    knn = sim.topk(k, largest=True)
    return knn.indices


def create_lora_model(
    model,
    r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: List[str] = None,
):
    """
    Creates a LoRA-wrapped causal language model over all layers.
    """
    # Create LoRA config
    peft_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,  # for generative text models
    )

    # Print info about modules LoRA will be applied to
    logger.info(f"Applying LoRA to the following modules: {target_modules}")
    # Wrap the model with LoRA adapters
    lora_model = get_peft_model(model, peft_config)
    logger.info("LoRA model created.")
    return lora_model


class MultiStopCriteria(StoppingCriteria):
    def __init__(self, stop_token_sequences):
        self.stop_sequences = stop_token_sequences

    def __call__(self, input_ids, scores, **kwargs):
        # Check if the latest generated tokens match any stop sequence
        for stop_seq in self.stop_sequences:
            if input_ids.shape[1] >= len(stop_seq):
                if torch.all(
                    input_ids[0, -len(stop_seq) :]
                    == torch.tensor(stop_seq, device=input_ids.device)
                ):
                    return True  # Trigger stop
        return False


def call_model(prompt, stop, model, tokenizer):
    # ==== Key model adaptation ====
    # Set special tokens for different models (e.g. Llama3 and Qwen2.5)
    if "llama3" in model.config.model_type.lower():
        # Llama3-instruct requires system prompt template
        full_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        tokenizer.eos_token = "<|eot_id|>"
    elif "qwen" in model.config.model_type.lower():
        # Qwen2.5-instruct uses <|im_start|> template
        full_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        tokenizer.pad_token = tokenizer.eos_token  # Ensure pad_token is set
    else:
        full_prompt = prompt

    # ==== Stop word handling ====
    # Encode stop words as token ID sequences (handling multi-token cases)
    stop_sequences = [tokenizer.encode(s, add_special_tokens=False) for s in stop]
    stop_criteria = MultiStopCriteria(stop_sequences)

    # ==== Model input encoding ====
    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_attention_mask=True,
    ).to(model.device)

    # ==== Generation config ====
    generate_args = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "max_new_tokens": 256,
        "temperature": 0.0,
        "do_sample": False,
        "stopping_criteria": StoppingCriteriaList([stop_criteria]),
        "pad_token_id": tokenizer.eos_token_id,  # Important: avoid generation interruption errors
    }

    # ==== Execute generation ====
    with torch.no_grad():
        outputs = model.generate(**generate_args)

    # ==== Decoding and post-processing ====
    # Extract newly generated part (excluding original prompt)
    new_tokens = outputs[0, inputs.input_ids.shape[1] :]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Second pass cleanup of stop words (prevent residuals due to tokenization differences)
    for stop_word in stop:
        if stop_word in generated_text:
            generated_text = generated_text.split(stop_word)[0].strip()
            break  # Stop at the first encountered stop word

    return generated_text


def mello(
    task_prompt,
    q,
    model,
    tokenizer,
    stop,
    contriever,
    contriever_tokenizer,
    embs,
    new_facts,
    answer,
    device,
):
    found_ans = False
    prompt = task_prompt + "\n\nQustion: " + q
    print("*********************************")
    for i in range(4):
        # prompt the model to generate a subquestion and a tentative answer
        gen = call_model(prompt, stop, model, tokenizer)
        last_sent = gen.strip().split("\n")[-1]
        # if final answer is there, get the answer and exit
        if last_sent.startswith("Final answer: "):
            found_ans = True
            ans = last_sent[len("Final answer: ") :]
            break
        # otherwise, extract the generated subquestion
        if len(gen.strip().split("\n")) < 2:
            break  # failed case
        subquestion = gen.strip().split("\n")[-2]
        if not subquestion.startswith("Subquestion: "):
            break  # failed case
        subquestion = subquestion[len("Subquestion: ") :]

        # retrieve an edited fact using the generated subquestion
        fact_ids = retrieve_facts(
            subquestion, embs, contriever, contriever_tokenizer, device
        )
        fact_sent = new_facts[fact_ids[0]]

        # put the retrieved fact at the end of the prompt, the model self-checks if it contradicts
        prompt = prompt + "\n" + gen + "\nRetrieved fact: " + fact_sent + "."
        logger.info(f"{i}:{gen}")
        logger.info(f"{i}:{fact_sent}")

    prompt = prompt + gen
    if not found_ans:
        return False, prompt
    return check_answer_in_pred(ans, answer), prompt


def preprocess_function(examples, tokenizer):
    # Combine text and target from each item_case_examples
    all_texts = []
    all_targets = []

    for item in examples["item_case_examples"]:
        for example in item:
            all_texts.append(example["text"])
            all_targets.append(example["target"])

    # Combine input and target texts
    inputs = [f"{text} {target}" for text, target in zip(all_texts, all_targets)]
    random.shuffle(inputs)
    learning_texts = []
    learning_targets = []
    for item in examples["learning_examples"]:
        for example in item:
            learning_texts.append(example["text"])
            learning_targets.append(example["target"])
    learning_inputs = [
        f"{text} {target}." for text, target in zip(learning_texts, learning_targets)
    ]
    random.shuffle(learning_inputs)
    # Tokenize the entire sequence
    final_inputs = inputs + learning_inputs
    model_inputs = tokenizer(
        final_inputs, padding=True, truncation=True, max_length=256, return_tensors="np"
    )
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs


# def check_answer(model, question, tokenizer, answer, device, max_new_tokens=512):
#     """Check if the model can correctly answer the question"""
#     inputs = tokenizer(question, return_tensors="pt").to(device)
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=max_new_tokens,
#         do_sample=False,
#         temperature=None,
#         top_p=None,
#         pad_token_id=tokenizer.eos_token_id,
#     )
#     generated_text = tokenizer.decode(
#         outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
#     )
#     print("-"*50)
#     print(f"Question: {question}")
#     print(f"Answer: {answer}")
#     print(f"Generated: {generated_text}")
#     print("-"*50)

#     return check_answer_in_pred(generated_text, answer)


def check_answer(
    model,
    question,
    tokenizer,
    answer,
    device,
    max_new_tokens=512,
):
    """Check if the model can correctly answer the question"""
    messages = [{"role": "user", "content": question}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    logger.info("-" * 50)
    logger.info(f"Question: {question}")
    logger.info(f"Answer: {answer}")
    logger.info(f"Generated: {generated_text}")
    logger.info("-" * 50)

    is_correct = check_answer_in_pred(generated_text, answer)
    return is_correct, generated_text


from transformers import GenerationConfig


def is_fast_inference(model):
    current = model
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, "vllm_engine"):
            return True
        current = getattr(current, "model", None)
    return False


def is_unsloth_model(model):
    current = model
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, "fast_generate"):
            return True
        current = getattr(current, "model", None)
    return False


def _get_compatible_paged_attn_implementation(model):
    original_attn = getattr(
        getattr(model, "config", None), "_attn_implementation", None
    )
    if original_attn is None:
        return None
    if original_attn == "flex_attention":
        return original_attn
    return f"paged|{original_attn}"


def _batch_generate_answers_chunk(
    model,
    tokenizer,
    questions,
    max_new_tokens=512,
):
    """Generate one chunk of chat answers, preferring Unsloth fast_generate when available."""

    texts = []
    for q in questions:
        messages = [{"role": "user", "content": q}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        texts.append(text)

    simple_batch_inputs = [
        tokenizer(t, add_special_tokens=False)["input_ids"] for t in texts
    ]

    if is_fast_inference(model):
        logger.info("Using unsloth fast_generate")
        original_padding_side = getattr(tokenizer, "padding_side", "right")
        try:
            tokenizer.padding_side = "left"
            prompt_token_ids_batch = [
                TokensPrompt(prompt_token_ids=list(input_ids))
                for input_ids in simple_batch_inputs
            ]
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=max_new_tokens,
                detokenize=True,
                repetition_penalty=1.1,
            )
            with _temporary_unsloth_lora_request(model) as lora_request:
                with torch.inference_mode():
                    outputs = model.fast_generate(
                        prompt_token_ids_batch,
                        sampling_params=sampling_params,
                        lora_request=lora_request,
                    )
        finally:
            tokenizer.padding_side = original_padding_side

        preds = []
        for output in outputs:
            preds.append(output.outputs[0].text)
        return preds

    if is_unsloth_model(model):
        logger.info("Using unsloth model.generate (fast_inference not enabled)")
        preds = []
        for input_ids in simple_batch_inputs:
            inputs = {
                "input_ids": torch.tensor([input_ids], device=model.device),
                "attention_mask": torch.ones(
                    len(input_ids), device=model.device
                ).unsqueeze(0),
            }
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
            pred = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            preds.append(pred)
        return preds

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        use_cuda_graph=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
    )
    restore_attn_impl = None
    if hasattr(model, "set_attn_implementation") and hasattr(model, "config"):
        original_attn_impl = getattr(model.config, "_attn_implementation", None)
        if original_attn_impl == "flex_attention":
            for fallback_attn_impl in ("flash_attention_2", "sdpa", "eager"):
                try:
                    model.set_attn_implementation(fallback_attn_impl)
                    restore_attn_impl = original_attn_impl
                    logger.info(
                        "Temporarily switched attn_implementation from %s to %s for generate_batch compatibility.",
                        original_attn_impl,
                        fallback_attn_impl,
                    )
                    break
                except Exception:
                    continue
    try:
        with torch.inference_mode():
            outputs = model.generate_batch(
                inputs=simple_batch_inputs,
                generation_config=generation_config,
            )
    finally:
        if restore_attn_impl is not None:
            model.set_attn_implementation(restore_attn_impl)

    preds = [None] * len(simple_batch_inputs)
    for request_id, output in outputs.items():
        idx = int(request_id.replace("req_", ""))
        generated_tokens = output.generated_tokens
        preds[idx] = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return preds


def batch_check_answer(
    model,
    questions,
    tokenizer,
    answers_list,
    device,
    max_new_tokens=512,
    chunk_size=None,
):
    """Batch check model answers to accelerate evaluation"""

    def _is_oom_error(exc: BaseException) -> bool:
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
        msg = str(exc).lower()
        oom_markers = [
            "out of memory",
            "cuda out of memory",
            "cudnn_status_not_supported",
            "failed to allocate memory",
        ]
        return any(marker in msg for marker in oom_markers)

    def _cleanup_after_oom():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    if len(questions) == 0:
        return []

    results = []
    n = len(questions)
    if chunk_size is None or chunk_size <= 0 or chunk_size >= n:
        chunk_size = n

    start = 0
    while start < n:
        current_chunk_size = min(chunk_size, n - start)
        while True:
            end = start + current_chunk_size
            q_chunk = questions[start:end]
            a_chunk = answers_list[start:end]

            try:
                preds = _batch_generate_answers_chunk(
                    model=model,
                    tokenizer=tokenizer,
                    questions=q_chunk,
                    max_new_tokens=max_new_tokens,
                )
                missing_idx = next(
                    (idx for idx, pred in enumerate(preds) if pred is None), None
                )
                if missing_idx is not None:
                    absolute_idx = start + missing_idx
                    raise torch.cuda.OutOfMemoryError(
                        "_batch_generate_answers_chunk returned None prediction at "
                        f"absolute_idx={absolute_idx}; retrying with smaller chunk."
                    )
                break
            except torch.cuda.OutOfMemoryError:
                if current_chunk_size <= 1:
                    raise
                new_chunk_size = max(1, current_chunk_size // 2)
                logger.warning(
                    f"batch_check_answer CUDA OOM at range [{start}, {end}), "
                    f"reducing chunk_size from {current_chunk_size} to {new_chunk_size} and retrying."
                )
                _cleanup_after_oom()
                current_chunk_size = new_chunk_size
            except RuntimeError as exc:
                if not _is_oom_error(exc) or current_chunk_size <= 1:
                    raise
                new_chunk_size = max(1, current_chunk_size // 2)
                logger.warning(
                    f"batch_check_answer OOM at range [{start}, {end}), "
                    f"reducing chunk_size from {current_chunk_size} to {new_chunk_size} and retrying."
                )
                _cleanup_after_oom()
                current_chunk_size = new_chunk_size

        for idx, pred in enumerate(preds):
            is_correct = check_answer_in_pred(pred, a_chunk[idx])
            results.append((is_correct, pred))

        start = end
    return results


def _batch_generate_with_assistant_prefix(
    model,
    tokenizer,
    user_inputs: List[str],
    assistant_prefixes: Optional[List[str]] = None,
    max_new_tokens: int = 256,
    chunk_size: Optional[int] = None,
):
    """Batch generation under per-sample assistant prefixes.

    Returns full assistant text per sample: `assistant_prefix + generated_continuation`.
    """
    if len(user_inputs) == 0:
        return []

    n = len(user_inputs)
    if assistant_prefixes is None:
        assistant_prefixes = [""] * n
    if len(assistant_prefixes) != n:
        raise ValueError("assistant_prefixes length must equal user_inputs length")

    if chunk_size is None or chunk_size <= 0 or chunk_size >= n:
        chunk_size = n

    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    eos_token_id = tokenizer.eos_token_id

    all_preds = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        q_chunk = user_inputs[start:end]
        p_chunk = assistant_prefixes[start:end]

        texts = []
        for q, ap in zip(q_chunk, p_chunk):
            messages = [{"role": "user", "content": q}]
            chat_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            texts.append(chat_prompt + ap if ap else chat_prompt)

        simple_batch_inputs = [
            tokenizer(t, add_special_tokens=False)["input_ids"] for t in texts
        ]
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            use_cuda_graph=False,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            do_sample=False,
        )
        with torch.inference_mode():
            outputs = model.generate_batch(
                inputs=simple_batch_inputs,
                generation_config=generation_config,
            )

        preds = [None] * len(simple_batch_inputs)
        for request_id, output in outputs.items():
            idx = int(request_id.replace("req_", ""))
            continuation = tokenizer.decode(
                output.generated_tokens, skip_special_tokens=True
            )
            preds[idx] = f"{p_chunk[idx]}{continuation}"
        all_preds.extend(preds)

    return all_preds


def compute_edit_quality_unedit_model_forced_decoding(
    model, tokenizer, edit_item, hparams, test_generation=False
):
    """Unedited-model forced-prefix decoding with causal-enhanced question coverage."""
    with edit_test_context(model):
        chunk_size = getattr(hparams, "eval_chunk_size", None)

        metrics = {
            "hop_wise": [],
            "hop_wise_pred": [],
            "accuracy": [],
            "accuracy_pred": [],
            "conflict_probe": [],
            "old_knowledge_probe": [],
            "causal_enhanced_hop_wise_pred": [],
            "causal_enhanced_conflict_probe": [],
        }

        hop_count = len(edit_item["new_single_hops"])
        hop_metrics = [[None, None, None] for _ in range(hop_count)]
        hop_preds = [[None, None, None] for _ in range(hop_count)]
        causal_enhanced_hop_preds = [[None, None, None] for _ in range(hop_count)]

        all_questions = []
        all_answers = []
        all_meta = []
        all_ctx = []

        requested_rewrite = edit_item.get("requested_rewrite", [])
        articles_by_rewrite = []
        rewrite_targets = []
        for req in requested_rewrite:
            article = str(req.get("article", "")).strip()
            articles_by_rewrite.append(article)
            target_new = req.get("target_new")
            if isinstance(target_new, dict):
                target_new = target_new.get("str")
            rewrite_targets.append(str(target_new).strip())

        def _norm_answer(text: str) -> str:
            if text is None:
                return ""
            text = str(text).lower()
            text = re.sub(r"\s+", " ", text).strip()
            text = re.sub(r"[^\w\s]", "", text)
            return text

        def _prefix_article(q: str, article: str) -> str:
            if not article:
                return q
            return (
                "Use the following article as authoritative updated knowledge for answering the question. "
                "It is an absolute fact. Do not doubt whether it is true.\n\n"
                f"{article}\n\n{q}"
            )

        def _match_hop_article(ans_list: List[str]) -> str:
            norms = {_norm_answer(a) for a in ans_list if a}
            for idx, target in enumerate(rewrite_targets):
                if _norm_answer(target) in norms:
                    return articles_by_rewrite[idx]
            return ""

        forced_prefix_template = "The answer is {target_new}."

        # hop / causal-enhanced hop
        for hop_i, hop_item in enumerate(edit_item["new_single_hops"]):
            ans = list(hop_item["answer_alias"])
            ans.append(hop_item["answer"])
            hop_article = _match_hop_article(ans)

            q1 = f"Now, {hop_item['cloze']} ? Let's think step by step."
            q2 = f"Now, {hop_item['question']} Let's think step by step."
            q3 = f"Now, {hop_item['question']} Why? Let's think step by step."

            for slot, q in enumerate([q1, q2, q3]):
                target_new = ans[-1]
                ctx = {
                    "new_answer": target_new,
                    "target_new": target_new,
                    "target_true": "",
                    "subject": hop_item.get("subject", ""),
                    "question": hop_item["question"],
                    "cloze": hop_item["cloze"],
                }
                all_questions.append(q)
                all_answers.append(list(ans))
                all_meta.append(
                    {
                        "type": "hop",
                        "hop_i": hop_i,
                        "slot": slot,
                        "q_raw": hop_item["cloze"]
                        if slot == 0
                        else hop_item["question"],
                    }
                )
                all_ctx.append(ctx)

                all_questions.append(_prefix_article(q, hop_article))
                all_answers.append(list(ans))
                all_meta.append(
                    {
                        "type": "ce_hop",
                        "hop_i": hop_i,
                        "slot": slot,
                        "q_raw": hop_item["cloze"]
                        if slot == 0
                        else hop_item["question"],
                    }
                )
                all_ctx.append(dict(ctx))

        # conflict probe / causal-enhanced conflict probe
        for req_i, req in enumerate(requested_rewrite):
            subject = req.get("subject")
            target_true = req.get("target_true").get("str")
            target_new = (
                req.get("target_new").get("str")
                if isinstance(req.get("target_new"), dict)
                else req.get("target_new")
            )
            cp_article = articles_by_rewrite[req_i]

            cp_questions = [
                f"{req.get('question')}\n(A) {target_true} (B) {target_new}\nTell me about {subject} first, and then answer.",
                f"{req.get('question')}\n(A) {target_new} (B) {target_true}\nTell me about {subject} first, and then answer.",
                f"{req.get('prompt').format(subject)} {target_true} or {target_new}? Tell me about {subject} first, and then answer.",
                f"{req.get('prompt').format(subject)} {target_new} or {target_true}? Tell me about {subject} first, and then answer.",
                f"{req.get('question')}\n(A) {target_true} (B) {target_new}\nAnswer first, and then tell me about {subject}.",
                f"{req.get('question')}\n(A) {target_new} (B) {target_true}\nAnswer first, and then tell me about {subject}.",
                f"{req.get('prompt').format(subject)} {target_true} or {target_new}? Answer first, and then tell me about {subject}.",
                f"{req.get('prompt').format(subject)} {target_new} or {target_true}? Answer first, and then tell me about {subject}.",
            ]
            for q in cp_questions:
                ctx = {
                    "new_answer": target_new,
                    "target_new": target_new,
                    "target_true": target_true,
                    "subject": subject,
                    "question": req.get("question"),
                    "cloze": req.get("prompt", "").format(subject),
                }
                all_questions.append(q)
                all_answers.append([target_new])
                all_meta.append({"type": "cp", "target_new": target_new})
                all_ctx.append(ctx)

                all_questions.append(_prefix_article(q, cp_article))
                all_answers.append([target_new])
                all_meta.append({"type": "ce_cp", "target_new": target_new})
                all_ctx.append(dict(ctx))

        forced_prefixes = []
        for ctx in all_ctx:
            target_new = str(ctx.get("target_new")).strip()
            forced_prefixes.append(
                forced_prefix_template.replace("{target_new}", target_new).strip()
            )
        all_preds = _batch_generate_with_assistant_prefix(
            model=model,
            tokenizer=tokenizer,
            user_inputs=all_questions,
            assistant_prefixes=forced_prefixes,
            max_new_tokens=768,
            chunk_size=chunk_size,
        )

        for i, (meta, pred, q, ans, fp) in enumerate(
            zip(all_meta, all_preds, all_questions, all_answers, forced_prefixes)
        ):
            is_ok = check_answer_in_pred(pred, ans)
            etype = meta["type"]

            if etype == "hop":
                hop_i = meta["hop_i"]
                slot = meta["slot"]
                hop_metrics[hop_i][slot] = is_ok
                hop_preds[hop_i][slot] = {
                    "q": meta["q_raw"],
                    "prompt": q,
                    "ref": list(ans),
                    "pred": pred,
                    "forced_prefix": fp,
                }
            elif etype == "ce_hop":
                hop_i = meta["hop_i"]
                slot = meta["slot"]
                causal_enhanced_hop_preds[hop_i][slot] = {
                    "q": meta["q_raw"],
                    "prompt": q,
                    "ref": list(ans),
                    "pred": pred,
                    "forced_prefix": fp,
                }
            elif etype == "mh":
                metrics["accuracy"].append(is_ok)
                metrics["accuracy_pred"].append(
                    {
                        "q": meta["q_raw"],
                        "prompt": q,
                        "ref": list(ans),
                        "pred": pred,
                        "forced_prefix": fp,
                    }
                )
            elif etype == "cp":
                metrics["conflict_probe"].append(
                    {
                        "prompt": q,
                        "ref": [meta["target_new"]],
                        "pred": pred,
                        "forced_prefix": fp,
                    }
                )
            elif etype == "ce_cp":
                metrics["causal_enhanced_conflict_probe"].append(
                    {
                        "prompt": q,
                        "ref": [meta["target_new"]],
                        "pred": pred,
                        "forced_prefix": fp,
                    }
                )

        metrics["hop_wise"] = hop_metrics
        metrics["hop_wise_pred"] = hop_preds
        metrics["causal_enhanced_hop_wise_pred"] = causal_enhanced_hop_preds
        return metrics


# def compute_edit_quality(model, tokenizer, edit_item, hparams, test_generation=False):
#     metrics = {
#         "hop_wise": [],
#         "accuracy": [],
#     }
#     device = f"cuda:{hparams.device}"
#     for i in edit_item["new_single_hops"]:
#         temp_metrics = []
#         ans = i["answer_alias"]
#         ans.append(i["answer"])
#         temp_metrics.append(
#             check_answer(
#                 model,
#                 "Answer: " + i["cloze"],
#                 tokenizer,
#                 ans,
#                 device,
#                 max_new_tokens=10,
#             )
#         )
#         temp_metrics.append(
#             check_answer(
#                 model,
#                 "Question: " + i["question"] + "\nAnswer: The answer is",
#                 tokenizer,
#                 ans,
#                 device,
#                 max_new_tokens=10,
#             )
#         )
#         metrics["hop_wise"].append(temp_metrics)
#     answer = edit_item["new_answer_alias"]
#     answer.append(edit_item["new_answer"])
#     metrics["accuracy"].append(
#         check_answer(
#             model,
#             "Answer: " + edit_item["cloze_question"],
#             tokenizer,
#             answer,
#             device,
#             max_new_tokens=10,
#         )
#     )
#     metrics["accuracy"].append(
#         check_answer(
#             model,
#             "Question: " + edit_item["questions"][0] + "\nAnswer: The answer is",
#             tokenizer,
#             answer,
#             device,
#             max_new_tokens=10,
#         )
#     )
#     return metrics


@contextmanager
def edit_test_context(model):
    """Context manager: automatically manage model's evaluation context state"""
    model.eval()
    original_attn = getattr(model.config, "_attn_implementation")
    model.config.use_cache = True
    if not hasattr(model, "fast_generate"):  # If not an unsloth model
        compatible_attn = _get_compatible_paged_attn_implementation(model)
        if compatible_attn is not None:
            model.set_attn_implementation(compatible_attn)
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    try:
        yield
    finally:
        model.train()
        model.set_attn_implementation(original_attn)


def compute_edit_quality_standard(
    model, tokenizer, edit_item, hparams, test_generation=False
):
    """
    Compute edit quality metrics (standard version)
    """
    with edit_test_context(model):
        original_attn = getattr(model.config, "_attn_implementation")
        compatible_attn = _get_compatible_paged_attn_implementation(model)
        if compatible_attn is not None:
            model.set_attn_implementation(compatible_attn)

        chunk_size = getattr(hparams, "eval_chunk_size", None)

        metrics = {
            "hop_wise": [],
            "hop_wise_pred": [],
            "accuracy": [],
            "accuracy_pred": [],
            "conflict_probe": [],
        }
        # Collect all questions, generate in one batch
        all_questions = []
        all_answers = []
        all_meta = []

        hop_count = len(edit_item["new_single_hops"])
        hop_metrics = [[None, None, None] for _ in range(hop_count)]
        hop_preds = [[None, None, None] for _ in range(hop_count)]

        for hop_i, i in enumerate(edit_item["new_single_hops"]):
            ans = i["answer_alias"]
            ans.append(i["answer"])

            q1 = f"Now, {i['cloze']} ? Let's think step by step."
            q2 = f"Now, {i['question']} Let's think step by step."
            q3 = f"Now, {i['question']} Why? Let's think step by step."
            for slot, q in enumerate([q1, q2, q3]):
                all_questions.append(q)
                all_answers.append(ans)
                all_meta.append(
                    {
                        "type": "hop",
                        "hop_i": hop_i,
                        "slot": slot,
                        "q_raw": i["cloze"] if slot == 0 else i["question"],
                    }
                )

        answer = edit_item["new_answer_alias"]
        answer.append(edit_item["new_answer"])

        logger.info(f"\nMulti-hop question\n")
        for i in range(len(edit_item["questions"])):
            q = f"Now, {edit_item['questions'][i]} Let's think step by step."
            all_questions.append(q)
            all_answers.append(answer)
            all_meta.append(
                {
                    "type": "mh",
                    "mh_i": i,
                    "q_raw": edit_item["questions"][i],
                }
            )

        # conflict probe
        logger.info(f"\nConflict probe\n")
        for req in edit_item.get("requested_rewrite"):
            subject = req.get("subject")
            target_true = req.get("target_true").get("str")
            target_new = (
                req.get("target_new").get("str")
                if isinstance(req.get("target_new"), dict)
                else req.get("target_new")
            )

            cp_questions = [
                f"{req.get('question')}\n(A) {target_true} (B) {target_new}\nTell me about {subject} first, and then answer.",
                f"{req.get('question')}\n(A) {target_new} (B) {target_true}\nTell me about {subject} first, and then answer.",
                f"{req.get('prompt').format(subject)} {target_true} or {target_new}? Tell me about {subject} first, and then answer.",
                f"{req.get('prompt').format(subject)} {target_new} or {target_true}? Tell me about {subject} first, and then answer.",
                # Answer first, then tell me about
                f"{req.get('question')}\n(A) {target_true} (B) {target_new}\nAnswer first, and then tell me about {subject}.",
                f"{req.get('question')}\n(A) {target_new} (B) {target_true}\nAnswer first, and then tell me about {subject}.",
                f"{req.get('prompt').format(subject)} {target_true} or {target_new}? Answer first, and then tell me about {subject}.",
                f"{req.get('prompt').format(subject)} {target_new} or {target_true}? Answer first, and then tell me about {subject}.",
            ]
            for q in cp_questions:
                all_questions.append(q)
                all_answers.append([target_new])
                all_meta.append(
                    {
                        "type": "cp",
                        "target_new": target_new,
                    }
                )

        all_results = batch_check_answer(
            model,
            all_questions,
            tokenizer,
            all_answers,
            None,
            max_new_tokens=768,
            chunk_size=chunk_size,
        )

        for idx, (meta, (is_ok, pred)) in enumerate(zip(all_meta, all_results)):
            q = all_questions[idx]
            ans = all_answers[idx]
            if meta["type"] == "hop":
                hop_i = meta["hop_i"]
                slot = meta["slot"]
                logger.info("-" * 50)
                logger.info(f"Hop {hop_i} Q{slot + 1}")
                logger.info(f"Question: {q}")
                logger.info(f"Answer: {ans}")
                logger.info(f"Generated: {pred}")
                logger.info("-" * 50)
                hop_metrics[hop_i][slot] = is_ok
                hop_preds[hop_i][slot] = {
                    "q": meta["q_raw"],
                    "prompt": q,
                    "ref": list(ans),
                    "pred": pred,
                }
            elif meta["type"] == "mh":
                logger.info("-" * 50)
                logger.info("Multi-hop")
                logger.info(f"Question: {q}")
                logger.info(f"Answer: {ans}")
                logger.info(f"Generated: {pred}")
                logger.info("-" * 50)
                metrics["accuracy"].append(is_ok)
                metrics["accuracy_pred"].append(
                    {
                        "q": meta["q_raw"],
                        "prompt": q,
                        "ref": list(ans),
                        "pred": pred,
                    }
                )
            elif meta["type"] == "cp":
                logger.info("-" * 50)
                logger.info("Conflict probe")
                logger.info(f"Question: {q}")
                logger.info(f"Answer: {[meta['target_new']]}")
                logger.info(f"Generated: {pred}")
                logger.info("-" * 50)
                metrics["conflict_probe"].append(
                    {
                        "prompt": q,
                        "ref": [meta["target_new"]],
                        "pred": pred,
                    }
                )

        metrics["hop_wise"] = hop_metrics
        metrics["hop_wise_pred"] = hop_preds
        return metrics


def compute_edit_quality_causal_enhanced(
    model, tokenizer, edit_item, hparams, test_generation=False
):
    """
    Causal enhancement with articles.
    Normally only used for 2-hop MQuAKE-CF dataset, to verify the causal necessity of the pilot study
    """
    with edit_test_context(model):
        chunk_size = getattr(hparams, "eval_chunk_size", None)

        metrics = {
            "hop_wise": [],
            "hop_wise_pred": [],
            "accuracy": [],
            "accuracy_pred": [],
            "conflict_probe": [],
            "old_knowledge_probe": [],
            "causal_enhanced_hop_wise_pred": [],
            "causal_enhanced_conflict_probe": [],
        }
        # Collect all questions, generate in one batch
        all_questions = []
        all_answers = []
        all_meta = []

        hop_count = len(edit_item["new_single_hops"])
        hop_metrics = [[None, None, None] for _ in range(hop_count)]
        hop_preds = [[None, None, None] for _ in range(hop_count)]
        causal_enhanced_hop_preds = [[None, None, None] for _ in range(hop_count)]

        requested_rewrite = edit_item.get("requested_rewrite")
        articles_by_rewrite = []
        rewrite_targets = []
        for req in requested_rewrite:
            article = str(req.get("article")).strip()
            articles_by_rewrite.append(article)
            target_new = req.get("target_new")
            if isinstance(target_new, dict):
                target_new = target_new.get("str")
            target_new = str(target_new).strip()
            rewrite_targets.append(target_new)

        def _norm_answer(text: str) -> str:
            if text is None:
                return ""
            text = str(text).lower()
            text = re.sub(r"\s+", " ", text).strip()
            text = re.sub(r"[^\w\s]", "", text)
            return text

        def _prefix_article(q: str, article: str) -> str:
            if not article:
                return q
            return (
                "Use the following article as authoritative updated knowledge for answering the question. "
                "It is an absolute fact. Do not doubt whether it is true.\n\n"
                f"{article}\n\n{q}"
            )

        def _match_hop_article(ans_list: List[str]) -> str:
            norms = {_norm_answer(a) for a in ans_list if a}
            for idx, target in enumerate(rewrite_targets):
                if _norm_answer(target) in norms:
                    return articles_by_rewrite[idx]
            # raise ValueError(f"Failed to match article for ans_list: {ans_list}")
            return ""

        for hop_i, i in enumerate(edit_item["new_single_hops"]):
            ans = i["answer_alias"]
            ans.append(i["answer"])

            hop_article = _match_hop_article(ans)

            q1 = f"Now, {i['cloze']} ? Let's think step by step."
            q2 = f"Now, {i['question']} Let's think step by step."
            q3 = f"Now, {i['question']} Why? Let's think step by step."
            for slot, q in enumerate([q1, q2, q3]):
                all_questions.append(q)
                all_answers.append(ans)
                all_meta.append(
                    {
                        "type": "hop",
                        "hop_i": hop_i,
                        "slot": slot,
                        "q_raw": i["cloze"] if slot == 0 else i["question"],
                    }
                )

                ce_q = _prefix_article(q, hop_article)
                all_questions.append(ce_q)
                all_answers.append(ans)
                all_meta.append(
                    {
                        "type": "ce_hop",
                        "hop_i": hop_i,
                        "slot": slot,
                        "q_raw": i["cloze"] if slot == 0 else i["question"],
                    }
                )

        answer = edit_item["new_answer_alias"]
        answer.append(edit_item["new_answer"])

        logger.info(f"\nMulti-hop question\n")
        for i in range(len(edit_item["questions"])):
            q = f"Now, {edit_item['questions'][i]} Let's think step by step."
            all_questions.append(q)
            all_answers.append(answer)
            all_meta.append(
                {
                    "type": "mh",
                    "mh_i": i,
                    "q_raw": edit_item["questions"][i],
                }
            )

        # conflict probe
        logger.info(f"\nConflict probe\n")
        for req_i, req in enumerate(edit_item.get("requested_rewrite")):
            subject = req.get("subject")
            target_true = req.get("target_true").get("str")
            target_new = (
                req.get("target_new").get("str")
                if isinstance(req.get("target_new"), dict)
                else req.get("target_new")
            )
            cp_article = articles_by_rewrite[req_i]

            cp_questions = [
                f"{req.get('question')}\n(A) {target_true} (B) {target_new}\nTell me about {subject} first, and then answer.",
                f"{req.get('question')}\n(A) {target_new} (B) {target_true}\nTell me about {subject} first, and then answer.",
                f"{req.get('prompt').format(subject)} {target_true} or {target_new}? Tell me about {subject} first, and then answer.",
                f"{req.get('prompt').format(subject)} {target_new} or {target_true}? Tell me about {subject} first, and then answer.",
                # Answer first, then tell me about
                f"{req.get('question')}\n(A) {target_true} (B) {target_new}\nAnswer first, and then tell me about {subject}.",
                f"{req.get('question')}\n(A) {target_new} (B) {target_true}\nAnswer first, and then tell me about {subject}.",
                f"{req.get('prompt').format(subject)} {target_true} or {target_new}? Answer first, and then tell me about {subject}.",
                f"{req.get('prompt').format(subject)} {target_new} or {target_true}? Answer first, and then tell me about {subject}.",
            ]
            for q in cp_questions:
                all_questions.append(q)
                all_answers.append([target_new])
                all_meta.append(
                    {
                        "type": "cp",
                        "target_new": target_new,
                    }
                )

                ce_q = _prefix_article(q, cp_article)
                all_questions.append(ce_q)
                all_answers.append([target_new])
                all_meta.append(
                    {
                        "type": "ce_cp",
                        "target_new": target_new,
                    }
                )

            old_questions = [
                f"Now it is {target_new}. Before the update, {req.get('prompt').format(subject)} ? Let's think step by step.",
                f"Now the answer is {target_new}. Before the update, {req.get('question')} Let's think step by step.",
            ]
            for q in old_questions:
                all_questions.append(q)
                all_answers.append([target_true])
                all_meta.append(
                    {
                        "type": "old_cp",
                        "target_true": target_true,
                    }
                )

        all_results = batch_check_answer(
            model,
            all_questions,
            tokenizer,
            all_answers,
            None,
            max_new_tokens=768,
            chunk_size=chunk_size,
        )

        for idx, (meta, (is_ok, pred)) in enumerate(zip(all_meta, all_results)):
            q = all_questions[idx]
            ans = all_answers[idx]
            if meta["type"] == "hop":
                hop_i = meta["hop_i"]
                slot = meta["slot"]
                logger.info("-" * 50)
                logger.info(f"Hop {hop_i} Q{slot + 1}")
                logger.info(f"Question: {q}")
                logger.info(f"Answer: {ans}")
                logger.info(f"Generated: {pred}")
                logger.info("-" * 50)
                hop_metrics[hop_i][slot] = is_ok
                hop_preds[hop_i][slot] = {
                    "q": meta["q_raw"],
                    "prompt": q,
                    "ref": list(ans),
                    "pred": pred,
                }
            elif meta["type"] == "ce_hop":
                hop_i = meta["hop_i"]
                slot = meta["slot"]
                causal_enhanced_hop_preds[hop_i][slot] = {
                    "q": meta["q_raw"],
                    "prompt": q,
                    "ref": list(ans),
                    "pred": pred,
                }
            elif meta["type"] == "mh":
                logger.info("-" * 50)
                logger.info("Multi-hop")
                logger.info(f"Question: {q}")
                logger.info(f"Answer: {ans}")
                logger.info(f"Generated: {pred}")
                logger.info("-" * 50)
                metrics["accuracy"].append(is_ok)
                metrics["accuracy_pred"].append(
                    {
                        "q": meta["q_raw"],
                        "prompt": q,
                        "ref": list(ans),
                        "pred": pred,
                    }
                )
            elif meta["type"] == "cp":
                logger.info("-" * 50)
                logger.info("Conflict probe")
                logger.info(f"Question: {q}")
                logger.info(f"Answer: {[meta['target_new']]}")
                logger.info(f"Generated: {pred}")
                logger.info("-" * 50)
                metrics["conflict_probe"].append(
                    {
                        "prompt": q,
                        "ref": [meta["target_new"]],
                        "pred": pred,
                    }
                )
            elif meta["type"] == "ce_cp":
                metrics["causal_enhanced_conflict_probe"].append(
                    {
                        "prompt": q,
                        "ref": [meta["target_new"]],
                        "pred": pred,
                    }
                )
            elif meta["type"] == "old_cp":
                logger.info("-" * 50)
                logger.info("Old knowledge probe")
                logger.info(f"Question: {q}")
                logger.info(f"Answer: {[meta['target_true']]}")
                logger.info(f"Generated: {pred}")
                logger.info("-" * 50)
                metrics["old_knowledge_probe"].append(
                    {
                        "prompt": q,
                        "ref": [meta["target_true"]],
                        "pred": pred,
                    }
                )

        metrics["hop_wise"] = hop_metrics
        metrics["hop_wise_pred"] = hop_preds
        metrics["causal_enhanced_hop_wise_pred"] = causal_enhanced_hop_preds
        return metrics


compute_edit_quality = compute_edit_quality_causal_enhanced
# compute_edit_quality = compute_edit_quality_standard
# compute_edit_quality = compute_edit_quality_unedit_model_forced_decoding


def edit_mello(
    model,
    task_prompt,
    stop,
    tokenizer,
    edit_item,
    hparams,
    contriever,
    contriever_tokenizer,
    embs,
    new_facts,
    test_generation=False,
):
    start = time()
    device = f"cuda:{hparams.device}"

    metrics = {
        "hop_wise": [],
        "hop_wise_pred": [],
        "accuracy": [],
        "accuracy_pred": [],
        "conflict_probe": [],
        "causal_enhanced_hop_wise_pred": [],
        "causal_enhanced_conflict_probe": [],
    }

    hop_count = len(edit_item["new_single_hops"])
    hop_metrics = [[None, None, None] for _ in range(hop_count)]
    hop_preds = [[None, None, None] for _ in range(hop_count)]

    for hop_i, hop_item in enumerate(edit_item["new_single_hops"]):
        ans = list(hop_item["answer_alias"])
        ans.append(hop_item["answer"])

        q1 = f"Now, {hop_item['cloze']} ? Let's think step by step."
        q2 = f"Now, {hop_item['question']} Let's think step by step."
        q3 = f"Now, {hop_item['question']} Why? Let's think step by step."

        for slot, q in enumerate([q1, q2, q3]):
            result, prompt = mello(
                task_prompt,
                q,
                model,
                tokenizer,
                stop,
                contriever,
                contriever_tokenizer,
                embs,
                new_facts,
                ans,
                device,
            )
            hop_metrics[hop_i][slot] = result
            hop_preds[hop_i][slot] = {
                "q": q,
                "prompt": prompt,
                "ref": list(ans),
                "pred": "mello_inference",
            }
            logger.info(f"Hop {hop_i} Q{slot + 1}: {result}")

    answer = list(edit_item["new_answer_alias"])
    answer.append(edit_item["new_answer"])

    for i, q_raw in enumerate(edit_item["questions"]):
        q = f"Now, {q_raw} Let's think step by step."
        result, prompt = mello(
            task_prompt,
            q,
            model,
            tokenizer,
            stop,
            contriever,
            contriever_tokenizer,
            embs,
            new_facts,
            answer,
            device,
        )
        metrics["accuracy"].append(result)
        metrics["accuracy_pred"].append(
            {
                "q": q,
                "prompt": prompt,
                "ref": list(answer),
                "pred": "mello_inference",
            }
        )
        logger.info(f"Multi-hop Q{i}: {result}")

    for req in edit_item.get("requested_rewrite", []):
        subject = req.get("subject")
        target_true = req.get("target_true", {}).get("str", "")
        target_new = req.get("target_new")
        if isinstance(target_new, dict):
            target_new = target_new.get("str", "")

        cp_questions = [
            f"{req.get('question')}\n(A) {target_true} (B) {target_new}\nTell me about {subject} first, and then answer.",
            f"{req.get('question')}\n(A) {target_new} (B) {target_true}\nTell me about {subject} first, and then answer.",
            f"{req.get('prompt', '').format(subject)} {target_true} or {target_new}? Tell me about {subject} first, and then answer.",
            f"{req.get('prompt', '').format(subject)} {target_new} or {target_true}? Tell me about {subject} first, and then answer.",
            f"{req.get('question')}\n(A) {target_true} (B) {target_new}\nAnswer first, and then tell me about {subject}.",
            f"{req.get('question')}\n(A) {target_new} (B) {target_true}\nAnswer first, and then tell me about {subject}.",
            f"{req.get('prompt', '').format(subject)} {target_true} or {target_new}? Answer first, and then tell me about {subject}.",
            f"{req.get('prompt', '').format(subject)} {target_new} or {target_true}? Answer first, and then tell me about {subject}.",
        ]
        for cp_q in cp_questions:
            result, prompt = mello(
                task_prompt,
                cp_q,
                model,
                tokenizer,
                stop,
                contriever,
                contriever_tokenizer,
                embs,
                new_facts,
                [target_new],
                device,
            )
            metrics["conflict_probe"].append(
                {
                    "prompt": cp_q,
                    "ref": [target_new],
                    "pred": "mello_inference",
                    "result": result,
                }
            )

    metrics["hop_wise"] = hop_metrics
    metrics["hop_wise_pred"] = hop_preds

    final_metrics = {
        "case_id": edit_item["case_id"],
        "requested_rewrite": edit_item["requested_rewrite"],
        "time": time() - start,
        "post": metrics,
    }
    return model, final_metrics


# def cake(original_model, tokenizer, item, hparams, test_generation=False):
#     target_modules = [
#         "q_proj",
#         "v_proj",
#         "k_proj",
#         "o_proj",
#         "up_proj",
#         "down_proj",
#         "gate_proj",
#     ]
#     model = create_lora_model(original_model, target_modules=target_modules)
#     # original_model = original_model.to(device)
#     model.enable_input_require_grads()
#     train_examples = []
#     item_case_examples = []
#     learning_examples = []
#     for rewrite in item["requested_rewrite"]:
#         prompt = rewrite["prompt"].format(rewrite["subject"])
#         target = rewrite["target_new"]["str"]
#         item_case_examples.append({"text": prompt, "target": target})
#         if "rephrase_prompt" in rewrite:
#             for rewrite_item in rewrite["rephrase_prompt"]:
#                 item_case_examples.append(
#                     {"text": rewrite_item["question"], "target": rewrite_item["answer"]}
#                 )
#         if "learning_prompt" in rewrite:
#             for learning_item in rewrite["learning_prompt"]:
#                 learning_examples.append(
#                     {
#                         "text": learning_item["question"],
#                         "target": learning_item["answer"],
#                     }
#                 )
#     train_examples.append(
#         {
#             "item_case_examples": item_case_examples,
#             "learning_examples": learning_examples,
#         }
#     )
#     train_dataset = Dataset.from_list(train_examples)
#     train_dataset = train_dataset.map(
#         preprocess_function,
#         batched=True,
#         remove_columns=train_dataset.column_names,
#         fn_kwargs={"tokenizer": tokenizer},
#     )

#     training_args = TrainingArguments(
#         output_dir=f"./output/",
#         overwrite_output_dir=True,
#         num_train_epochs=40,
#         per_device_train_batch_size=4,
#         learning_rate=1e-4,
#         save_strategy="no",
#         bf16=True,
#         logging_steps=10,
#         report_to="none",
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#     )
#     start = time()
#     trainer.train()
#     exec_time = time() - start
#     metrics = {
#         "case_id": item["case_id"],
#         "requested_rewrite": item["requested_rewrite"],
#         "time": exec_time,
#         "post": compute_edit_quality(
#             model, tokenizer, item, hparams, test_generation=test_generation
#         ),
#     }
#     model = model.unload()
#     del model.peft_config

#     return model, metrics


def cake(
    original_model,
    tokenizer,
    item,
    hparams,
    test_generation=False,
    skip_post_eval=False,
):
    target_modules = [
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
    ]
    model = create_lora_model(original_model, target_modules=target_modules)
    # original_model = original_model.to(device)
    model.enable_input_require_grads()
    train_examples = []
    item_case_examples = []
    learning_examples = []
    for rewrite in item["requested_rewrite"]:
        prompt = rewrite["prompt"].format(rewrite["subject"])
        target = rewrite["target_new"]["str"]
        item_case_examples.append({"text": prompt, "target": target})
        if "rephrase_prompt" in rewrite:
            for rewrite_item in rewrite["rephrase_prompt"]:
                item_case_examples.append(
                    {"text": rewrite_item["question"], "target": rewrite_item["answer"]}
                )
        if "learning_prompt" in rewrite:
            for learning_item in rewrite["learning_prompt"]:
                learning_examples.append(
                    {
                        "text": learning_item["question"],
                        "target": learning_item["answer"],
                    }
                )
    train_examples.append(
        {
            "item_case_examples": item_case_examples,
            "learning_examples": learning_examples,
        }
    )
    train_dataset = Dataset.from_list(train_examples)
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        fn_kwargs={"tokenizer": tokenizer},
    )

    training_args = TrainingArguments(
        # output_dir=f"./output/",
        # overwrite_output_dir=True,
        num_train_epochs=40,
        per_device_train_batch_size=4,
        learning_rate=1e-4,
        save_strategy="no",
        bf16=True,
        logging_steps=10,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    start = time()
    trainer.train()
    exec_time = time() - start

    model.eval()
    model.config.use_cache = True
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    if skip_post_eval:
        metrics = {
            "case_id": item["case_id"],
            "requested_rewrite": item["requested_rewrite"],
            "time": exec_time,
        }
        return model, metrics

    metrics = {
        "case_id": item["case_id"],
        "requested_rewrite": item["requested_rewrite"],
        "time": exec_time,
        "post": compute_edit_quality(
            model, tokenizer, item, hparams, test_generation=test_generation
        ),
    }

    model = model.unload()
    del model.peft_config

    return model, metrics


def edit(
    model,
    tokenizer,
    edit_item,
    hparams,
    alg_name,
    apply_algo,
    test_generation=False,
    datatype=None,
):
    # all_metrics = []
    start = time()
    requests = edit_item["requested_rewrite"]
    for i in requests:
        if isinstance(i.get("target_new"), dict):
            i["target_new"] = i["target_new"].get("str")
    if hasattr(hparams, "batch_size"):
        hparams.batch_size = len(requests)
    edited_model, weights_copy = apply_algo(
        model,
        tokenizer,
        requests,
        hparams,
        copy=False,
        return_orig_weights=True,
        keep_original_weight=True,
        edit_item=edit_item,
    )
    exec_time = time() - start
    # start = time()
    # chunk_metrics = []
    metrics = {
        "case_id": edit_item["case_id"],
        "requested_rewrite": requests,
        "time": exec_time,
        "post": compute_edit_quality(
            edited_model, tokenizer, edit_item, hparams, test_generation=test_generation
        ),
    }
    # chunk_metrics.append(metrics)
    if alg_name == "KN" or alg_name == "GRACE" or alg_name == "WISE":
        with torch.no_grad():
            weights_copy()
    elif (
        alg_name == "LoRA"
        or alg_name == "QLoRA"
        or alg_name == "DPO"
        or alg_name == "CausalEdit"
    ):
        edited_model = edited_model.unload()
        del edited_model.peft_config
    elif alg_name == "MELO":
        model = edited_model
    elif alg_name == "AlphaEdit":
        # save_dir = edit_item["requested_rewrite"][0].get("save_dir")
        # if save_dir:
        #     alphaedit_dir = os.path.join(save_dir, "alphaedit_model")
        #     os.makedirs(alphaedit_dir, exist_ok=True)
        #     edited_weights = {}
        #     for layer in hparams.layers:
        #         weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        #         edited_weights[weight_name] = (
        #             nethook.get_parameter(edited_model, weight_name).detach().cpu()
        #         )
        #     model_name_safe = hparams.model_name.replace("/", "_")
        #     datatype_str = datatype if datatype else "unknown"
        #     weights_filename = f"{datatype_str}_{model_name_safe}_edited_weights.pt"
        #     torch.save(edited_weights, os.path.join(alphaedit_dir, weights_filename))
        #     requests_filename = f"{datatype_str}_{model_name_safe}_requests.json"
        #     with open(os.path.join(alphaedit_dir, requests_filename), "w") as f:
        #         json.dump(requests, f, ensure_ascii=False, indent=4)
        #     logger.info(f"AlphaEdit weights saved to: {alphaedit_dir}")
        with torch.no_grad():
            for k, v in weights_copy.items():
                param = nethook.get_parameter(model, k)
                param[...] = v.to(param.device)
    elif alg_name == "PreEdited":
        edited_model = model
    else:
        with torch.no_grad():
            for k, v in weights_copy.items():
                param = nethook.get_parameter(model, k)
                param[...] = v.to(param.device)
    return edited_model, metrics


def edit_batch(
    model,
    tokenizer,
    batch_requests,
    all_eval_items,
    hparams,
    alg_name,
    apply_algo,
    test_generation=False,
    datatype=None,
    batch_edit_item=None,
):
    start = time()
    requests = deepcopy(batch_requests)
    weights_copy = None
    edit_item = batch_edit_item or {
        "case_id": "batch_edit",
        "requested_rewrite": requests,
    }
    if alg_name == "CAKE":
        edited_model, metrics = cake(
            model,
            tokenizer,
            edit_item,
            hparams,
            test_generation=test_generation,
            skip_post_eval=True,
        )
        exec_time = metrics["time"]
    else:
        for request in requests:
            if isinstance(request.get("target_new"), dict):
                request["target_new"] = request["target_new"].get("str")

        edited_model, weights_copy = apply_algo(
            model,
            tokenizer,
            requests,
            hparams,
            copy=False,
            return_orig_weights=True,
            keep_original_weight=True,
            edit_item=edit_item,
        )
        exec_time = time() - start

    current_batch_size = len(edit_item["requested_rewrite"])
    if current_batch_size == 90:
        save_batch_edit_artifacts(
            edited_model,
            edit_item,
            hparams,
            alg_name,
            all_eval_items=all_eval_items,
            weights_copy=weights_copy,
        )
    else:
        logger.info(
            "Skip saving batch edit artifacts because batch_size=%s (only save when batch_edit=90).",
            current_batch_size,
        )

    all_metrics = []
    for eval_item in all_eval_items:
        metrics = compute_edit_quality(
            edited_model,
            tokenizer,
            eval_item,
            hparams,
            test_generation=test_generation,
        )
        all_metrics.append(
            {
                "case_id": eval_item["case_id"],
                "requested_rewrite": eval_item["requested_rewrite"],
                "time": exec_time,
                "post": metrics,
            }
        )

    if alg_name in {"CausalEdit", "CAKE"}:
        edited_model = edited_model.unload()
        if hasattr(edited_model, "peft_config"):
            del edited_model.peft_config
    elif alg_name in {"AlphaEdit", "MEMIT", "EMMET"}:
        with torch.no_grad():
            for k, v in weights_copy.items():
                param = nethook.get_parameter(model, k)
                param[...] = v.to(param.device)
    else:
        raise ValueError(f"Unsupported batch edit: {alg_name}")

    return edited_model, all_metrics


def edit_rome(model, tokenizer, edit_item, hparams, apply_algo, test_generation=False):
    # all_metrics = []
    start = time()
    requests = edit_item["requested_rewrite"]
    logger.info(requests)
    for i in requests:
        i["target_new"] = i["target_new"]["str"]
    origin_weights_copy = None
    for index, request in enumerate(requests):
        edited_model, weights_copy = apply_algo(
            model, tokenizer, [request], hparams, copy=False, return_orig_weights=True
        )
        if index == 0:
            origin_weights_copy = weights_copy
    exec_time = time() - start
    metrics = {
        "case_id": edit_item["case_id"],
        "requested_rewrite": requests,
        "time": exec_time,
        "post": compute_edit_quality(
            edited_model, tokenizer, edit_item, hparams, test_generation=test_generation
        ),
    }
    with torch.no_grad():
        for k, v in origin_weights_copy.items():
            param = nethook.get_parameter(model, k)
            param[...] = v.to(param.device)
    return edited_model, metrics


def edit_ifmet(
    model, tokenizer, edit_item, hparams_s, hparams_d, apply_algo, test_generation=False
):
    start = time()
    requests = edit_item["requested_rewrite"]
    ifmet_requests = []
    for i in requests:
        i["target_new"] = i["target_new"]["str"]
        if len(i.get("ifmet_question", [])) != 0:
            j = {}
            j["prompt"] = i["ifmet_question"][0]
            j["target_new"] = i["target_new"]
            j["subject"] = j["prompt"]
            j["question"] = i["question"]
            ifmet_requests.append(j)
    hparams_s.batch_size = len(requests)
    hparams_d.batch_size = len(ifmet_requests)
    origin_weights_copy = None
    edited_model, weights_copy = apply_algo(
        model, tokenizer, requests, hparams_s, copy=False, return_orig_weights=True
    )
    origin_weights_copy = weights_copy
    if len(ifmet_requests) > 0:
        edited_model, weights_copy = apply_algo(
            edited_model,
            tokenizer,
            ifmet_requests,
            hparams_d,
            copy=False,
            return_orig_weights=True,
        )
    exec_time = time() - start
    metrics = {
        "case_id": edit_item["case_id"],
        "requested_rewrite": requests,
        "time": exec_time,
        "post": compute_edit_quality(
            edited_model,
            tokenizer,
            edit_item,
            hparams_s,
            test_generation=test_generation,
        ),
    }
    with torch.no_grad():
        if len(ifmet_requests) > 0:
            for k, v in weights_copy.items():
                param = nethook.get_parameter(model, k)
                param[...] = v.to(param.device)
        for k, v in origin_weights_copy.items():
            param = nethook.get_parameter(model, k)
            param[...] = v.to(param.device)
    return edited_model, metrics


def edit_wise(
    model,
    tokenizer,
    edit_item,
    hparams,
    loc_data,
    loc_index,
    apply_algo,
    test_generation=False,
):
    start = time()
    requests = edit_item["requested_rewrite"]
    for i, item in enumerate(requests):
        item["prompt"] = item["prompt"].format(item["subject"])
        item["target_new"] = item["target_new"]["str"]
        item.update(
            {
                "loc_prompt": loc_data[loc_index + i]["loc"]
                + " "
                + loc_data[loc_index + i]["loc_ans"]
            }
        )
    loc_index = loc_index + len(requests)
    hparams.batch_size = len(requests)
    edited_model, weights_copy = apply_algo(
        model,
        tokenizer,
        requests,
        hparams,
        copy=False,
        return_orig_weights=True,
        keep_original_weight=True,
    )
    eval_model = edited_model.model if hasattr(edited_model, "model") else edited_model
    exec_time = time() - start
    metrics = {
        "case_id": edit_item["case_id"],
        "requested_rewrite": requests,
        "time": exec_time,
        "post": compute_edit_quality(
            eval_model, tokenizer, edit_item, hparams, test_generation=test_generation
        ),
    }
    with torch.no_grad():
        weights_copy()
    return metrics, loc_index


def get_real_model_path(
    repo_id,
    base_cache_path=None,
):
    import glob

    """
    Locate the snapshots path containing config.json based on the given cache root directory and Repo ID
    """
    if base_cache_path is None:
        base_cache_path = os.getenv(
            "HF_HOME", os.path.join(os.path.dirname(__file__), "../huggface_cache")
        )
    # Convert "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" to "models--unsloth--Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    formatted_repo = "models--" + repo_id.replace("/", "--")

    # Build search path for snapshots
    # Note: ensure base_cache_path points to your 'huggface_cache' folder
    search_pattern = os.path.join(
        base_cache_path, "hub", formatted_repo, "snapshots", "*"
    )

    snapshots = glob.glob(search_pattern)
    if not snapshots:
        raise OSError(
            f"No model files found under {search_pattern}, please check if the HF_HOME path is correct."
        )

    # Return the latest (or first) hash folder
    return snapshots[0]
