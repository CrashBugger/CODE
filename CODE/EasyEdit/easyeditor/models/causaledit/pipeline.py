import signal
import sys
import tempfile
import shutil
from tqdm import tqdm
import re
import math
import hashlib
import json
import os
import logging
import html
import time
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from filelock import FileLock

logger = logging.getLogger("mine")
import unsloth
from unsloth import FastLanguageModel
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset, concatenate_datasets
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    inject_adapter_in_model,
    set_peft_model_state_dict,
)
from .minillm.minillm_config import MiniLLMConfig
from .minillm.minillm_trainer import MiniLLMTrainer
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer
from torch.cuda import empty_cache
from transformers import TrainerCallback
import swanlab
import random
import datetime
from peft import PeftModel
import uuid

from vllm import SamplingParams, TokensPrompt

# Global cache dict for caching _rejection_sampling results
_rejection_sampling_cache: Dict[str, str] = {}
_EDITING_MODE = "causal"

_LATENT_THINKING_REJECT_PATTERNS = [
    re.compile(r"<\/?think(?:ing)?>", re.IGNORECASE),
    re.compile(r"\bthe answer now is\b", re.IGNORECASE),
    re.compile(r"\bcheat sheet\b", re.IGNORECASE),
    re.compile(r"\buser provided\b", re.IGNORECASE),
    re.compile(r"\bprovided an article\b", re.IGNORECASE),
    re.compile(r"\breference article\b", re.IGNORECASE),
    re.compile(r"\bi was told\b", re.IGNORECASE),
    re.compile(r"\bcurrent knowledge\b", re.IGNORECASE),
    re.compile(r"\bpublicly available information\b", re.IGNORECASE),
    re.compile(r"\bquick search\b", re.IGNORECASE),
    re.compile(r"\bi did not find\b", re.IGNORECASE),
    re.compile(r"\bremains unclear\b", re.IGNORECASE),
    re.compile(r"\bmay still be accurate\b", re.IGNORECASE),
    re.compile(r"\bpublicly stated\b", re.IGNORECASE),
    re.compile(r"\bpublicly expressed\b", re.IGNORECASE),
    re.compile(r"\baccording to the recent developments\b", re.IGNORECASE),
    re.compile(r"\bthis change is likely due to\b", re.IGNORECASE),
    re.compile(r"\bhowever, considering recent updates\b", re.IGNORECASE),
    re.compile(r"\bhowever, with the current context\b", re.IGNORECASE),
    re.compile(r"\bto provide a precise answer\b", re.IGNORECASE),
    re.compile(r"\bavailable information\b", re.IGNORECASE),
]


def _get_cache_key_explict(model_name, question: str, reference: str) -> str:
    """
    Generate cache key based on model name, question, and reference (article)

    Args:
        model: Model object
        question: Question
        reference: Reference article

    Returns:
        Cache key string
    """

    # Combine key and compute hash
    hash_ref = hashlib.shake_128(reference.encode("utf-8")).hexdigest(10)
    return f"{model_name.lower()}|{question.lower()}|{hash_ref}"


def _sanitize_for_filename(value: str) -> str:
    sanitized = re.sub(r'[\\/:*?"<>|\s]+', "_", value.strip())
    return sanitized.strip("._") or "unknown"


def _get_model_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def _is_probably_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    oom_markers = [
        "out of memory",
        "cuda out of memory",
        "cublas_status_alloc_failed",
        "cuda error: out of memory",
        "hip out of memory",
    ]
    return any(marker in message for marker in oom_markers)


def _apply_chat_template_with_native_thinking(tokenizer, messages, **kwargs):
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=True, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def _uses_native_thinking_template(tokenizer) -> bool:
    tokenizer_name = getattr(tokenizer, "name_or_path", "")
    tokenizer_name = tokenizer_name.lower()
    return "qwen3" in tokenizer_name


def _render_answer_text(answer: str) -> str:
    return f"The answer now is {answer}."


def _render_completion_with_think(thought: str, answer: str, *, tokenizer: Any) -> str:
    """
    Render complete response text with thinking process

    Combines the thinking process and final answer into text that conforms
    to the model output format.
    Output format: <think_block>\n<answer_text>

    Args:
        thought: Thinking process text
        answer: Final answer text
        tokenizer: Tokenizer, used to determine whether to add thinking tags

    Returns:
        Complete response text, e.g. "<thought>\n</think_part>\nThe answer now is <answer>."
    """

    def _render_think_block(thought: str, *, tokenizer: Any, include_open_tag: bool | None = None) -> str:
        """
        Render thinking block

        Decide whether to add thinking tags based on tokenizer type (e.g. Qwen3
        natively supports thinking template, no extra tags needed; other models
        require manual tag addition).

        Args:
            thought: Thinking process text
            tokenizer: Tokenizer
            include_open_tag: Whether to add thinking tags, auto-detect when None

        Returns:
            Thinking block text, e.g. "<thought>\n" or "\n<thought>\n"
        """
        cleaned_thought = thought.strip()
        if include_open_tag is None:
            include_open_tag = not _uses_native_thinking_template(tokenizer)
        if include_open_tag:
            return f"<think>\n{cleaned_thought}\n</think>"
        return f"{cleaned_thought}\n</think>"

    think_part = _render_think_block(thought, tokenizer=tokenizer)
    return f"{think_part}\n{_render_answer_text(answer)}"


def _load_cache_from_disk(
    cache_dir: str = "./cache/causaledit",
    cache_filename: str = "rejection_sampling_cache.json",
):
    """
    Load cache from disk (with file lock for multi-process safe reads)

    Args:
        cache_dir: Cache directory path
        cache_filename: Cache filename
    """
    global _rejection_sampling_cache
    cache_file = os.path.join(cache_dir, cache_filename)
    lock_file = cache_file + ".lock"
    lock = FileLock(lock_file)

    with lock:
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    _rejection_sampling_cache = json.load(f)
                logger.info(f"Loaded cache: {len(_rejection_sampling_cache)} records")
            except Exception as e:
                logger.error(f"Failed to load cache: {e}")
                _rejection_sampling_cache = {}
        else:
            _rejection_sampling_cache = {}


def _save_cache_to_disk(
    cache_dir: str = "./cache/causaledit",
    cache_filename: str = "rejection_sampling_cache.json",
):
    """
    Save cache to disk (with file lock for multi-process safe writes)

    Before writing, reads the latest data from disk and merges it with the
    current in-memory data to avoid data loss from concurrent multi-process writes.
    Uses atomic writes (write to temp file then rename) and registers signal
    handlers to prevent Ctrl+C from interrupting writes and corrupting files.

    Args:
        cache_dir: Cache directory path
        cache_filename: Cache filename
    """
    global _rejection_sampling_cache
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, cache_filename)
    lock_file = cache_file + ".lock"
    lock = FileLock(lock_file)

    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)

    def _delay_signal(signum, frame):
        logger.info("Received interrupt signal, waiting for cache save to complete before exiting...")

    signal.signal(signal.SIGINT, _delay_signal)
    signal.signal(signal.SIGTERM, _delay_signal)

    try:
        with lock:
            existing = {}
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to read existing cache, will overwrite: {e}")
                    existing = {}

            existing.update(_rejection_sampling_cache)

            try:
                fd, temp_path = tempfile.mkstemp(dir=cache_dir, suffix=".tmp", prefix="atomic_")
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        json.dump(existing, f, ensure_ascii=False, indent=2)
                        f.flush()
                        os.fsync(f.fileno())
                    shutil.move(temp_path, cache_file)
                    temp_path = None
                except Exception:
                    if temp_path and os.path.exists(temp_path):
                        try:
                            os.unlink(temp_path)
                        except OSError:
                            pass
                    raise
                logger.info(f"Saved cache: {len(existing)} records (added {len(_rejection_sampling_cache)} new)")
            except Exception as e:
                logger.error(f"Failed to save cache: {e}")
    finally:
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)


def _find_token_span(tokenizer, full_text, search_text, occurrence=-1):
    """
    Find the token index range of search_text within the token sequence of full_text.

    Args:
        tokenizer: HuggingFace tokenizer object
        full_text: Complete paragraph text
        search_text: Sub-string to search for
        occurrence: If search_text appears multiple times, specify which occurrence to find (default -1, i.e. the last occurrence)

    Returns:
        (start_token_idx, end_token_idx): Half-open interval [start, end)
        e.g.: tokens[start:end] is the token list corresponding to the sub-string
    """

    # 1. Find the character position of search_text in the string (Character Indices)
    start_char = -1
    current_occurrence = 1

    # If occurrence is -1, find the last occurrence
    if occurrence == -1:
        # First compute all occurrence positions
        all_positions = []
        search_start = 0
        while True:
            pos = full_text.find(search_text, search_start)
            if pos == -1:
                break
            all_positions.append(pos)
            search_start = pos + 1

        if not all_positions:
            raise ValueError(f"'{search_text}' not found in text")

        # Use the last occurrence position
        start_char = all_positions[-1]
    else:
        # Handle the case of specified occurrence count
        search_start = 0
        while current_occurrence <= occurrence:
            start_char = full_text.find(search_text, search_start)
            if start_char == -1:
                raise ValueError(f"'{search_text}' not found in text (occurrence {occurrence})")
            search_start = start_char + 1
            current_occurrence += 1
        # Restore the last found start position
        start_char = search_start - 1

    end_char = start_char + len(search_text)

    # 2. Encode the entire text and get Offset Mapping
    # return_offsets_mapping=True is key, but in FastTokenizer,
    # it's usually more convenient to call char_to_token directly via the encoding object
    encoding = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)  # By default, full_text already has special tokens added

    # 3. Map character indices to Token indices
    # char_to_token(char_index) returns the index of the token ID that the character belongs to

    # Find the start token
    # Sometimes start_char is a space, which may cause mapping issues; do a simple forward search here
    token_start_index = encoding.char_to_token(start_char)
    while token_start_index is None and start_char < end_char:
        start_char += 1
        token_start_index = encoding.char_to_token(start_char)

    # Find the end token
    # Note: we want the token containing the last character
    token_end_index = encoding.char_to_token(end_char - 1)
    while token_end_index is None and end_char > start_char:
        end_char -= 1
        token_end_index = encoding.char_to_token(end_char - 1)

    if token_start_index is None or token_end_index is None:
        raise ValueError("Failed to map character range to tokens, possibly at a special boundary")

    # Return half-open interval (Python slice style)
    return token_start_index, token_end_index + 1


@torch.inference_mode()
def _get_perplexity_batch(model_to_use, tokenizer_to_use, messages_list, completion_list, target_part_list):
    """
    Batch compute perplexity of p(completion | messages)
    Use chat template to properly format inputs

    Args:
        model_to_use: Model for computation
        tokenizer_to_use: Corresponding tokenizer
        messages_list: List of message lists (batch_size items)
        completion_list: Completion text list (batch_size items)
        target_part_list: Target part text list (batch_size items)

    Returns:
        perplexities: Perplexity list (batch_size items)
    """
    batch_size = len(messages_list)
    device = next(model_to_use.parameters()).device

    # Format all inputs using chat template
    full_texts = []
    for messages, completion in zip(messages_list, completion_list):
        prompt = _apply_chat_template_with_native_thinking(
            tokenizer_to_use,
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = prompt + completion
        full_texts.append(full_text)

    # Batch encode input texts
    inputs = tokenizer_to_use(full_texts, return_tensors="pt", padding=True, padding_side="right")
    inputs = inputs.to(device)

    # Get model output logits (batch inference)
    with torch.no_grad():
        outputs = model_to_use(**inputs)
        logits = outputs.logits

    # Get input token IDs
    input_ids = inputs.input_ids

    # Compute log probability for each token (batch operation)
    log_probs = torch.log_softmax(logits, dim=-1)

    # Batch compute log probability of the token at each position
    # For each position i, compute the probability of token i+1
    # Use gather for batch retrieval
    batch_size, seq_len, vocab_size = logits.shape

    # Create target token IDs (shift right by 1)
    # Note: the last position has no next token, so the last column is set to 0 (will be ignored later)
    target_token_ids = torch.zeros_like(input_ids)
    target_token_ids[:, :-1] = input_ids[:, 1:]

    # Use gather for batch retrieval of log probabilities
    # log_probs shape: [batch_size, seq_len, vocab_size]
    # target_token_ids shape: [batch_size, seq_len]
    # We need to get the log prob of the corresponding token for each batch and position
    token_log_probs = torch.gather(log_probs, dim=-1, index=target_token_ids.unsqueeze(-1)).squeeze(-1)  # [batch_size, seq_len]

    # Only keep valid log probs for the first seq_len-1 positions (the last position has no next token)
    token_log_probs = token_log_probs[:, :-1]  # [batch_size, seq_len-1]

    # Compute perplexity for the target part of each sample separately
    perplexities = []
    for idx in range(batch_size):
        # Find the token range of the target part for this sample
        full_text = full_texts[idx]
        target_part = target_part_list[idx]

        start_idx, end_idx = _find_token_span(tokenizer_to_use, full_text, target_part)

        # Compute perplexity of the target part
        # Note: token_log_probs has one fewer token than tokens (since the last token has no next token)
        target_log_probs = token_log_probs[idx, start_idx - 1 : end_idx - 1]
        perplexity = math.exp(-target_log_probs.sum().item() / (end_idx - start_idx))
        perplexities.append(perplexity)

    return perplexities


@contextmanager
def _my_context_manager(model):
    model.eval()
    if hasattr(model, "disable_adapter_layers"):
        model.disable_adapter_layers()
    with torch.inference_mode():
        yield
    if hasattr(model, "enable_adapter_layers"):
        model.enable_adapter_layers()
    model.train()


@contextmanager
def _temporary_disable_adapter(model):
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            yield
        return

    if hasattr(model, "disable_adapter_layers"):
        model.disable_adapter_layers()
        try:
            yield
        finally:
            model.enable_adapter_layers()
        return

    raise ValueError("Model does not support adapter layers")


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _set_adapter_requires_grad(model, adapter_name: str, requires_grad: bool):
    raw_model = _unwrap_model(model)
    if adapter_name not in getattr(raw_model, "peft_config", {}):
        return
    raw_model.set_requires_grad(adapter_name, requires_grad=requires_grad)


def _has_adapter(model, adapter_name: str) -> bool:
    raw_model = _unwrap_model(model)
    return adapter_name in getattr(raw_model, "peft_config", {})


def _activate_adapter(model, adapter_name: str, *, trainable: bool):
    raw_model = _unwrap_model(model)
    raw_model.set_adapter(adapter_name)
    _set_adapter_requires_grad(raw_model, adapter_name, requires_grad=trainable)
    return model


def _activate_student_teacher_state(model, student_adapter_name: str = "default", teacher_adapter_name: str = "teacher"):
    _activate_adapter(model, student_adapter_name, trainable=True)
    if _has_adapter(model, teacher_adapter_name):
        _set_adapter_requires_grad(model, teacher_adapter_name, requires_grad=False)
    return model


def _snapshot_adapter(model, adapter_name: str = "default"):
    raw_model = _unwrap_model(model)
    if adapter_name not in getattr(raw_model, "peft_config", {}):
        raise ValueError(f"Adapter '{adapter_name}' not found on model")
    return {
        "config": deepcopy(raw_model.peft_config[adapter_name]),
        "state_dict": get_peft_model_state_dict(raw_model, adapter_name=adapter_name),
    }


@contextmanager
def _temporary_injected_adapter(
    model,
    *,
    adapter_snapshot: dict[str, Any],
    adapter_name: str,
    fallback_adapter_name: str = "default",
):
    raw_model = _unwrap_model(model)
    if _has_adapter(raw_model, adapter_name):
        raise RuntimeError(f"Temporary adapter '{adapter_name}' already exists on the model")

    try:
        inject_adapter_in_model(
            peft_config=deepcopy(adapter_snapshot["config"]),
            model=raw_model,
            adapter_name=adapter_name,
        )
        set_peft_model_state_dict(
            raw_model,
            peft_model_state_dict=adapter_snapshot["state_dict"],
            adapter_name=adapter_name,
        )
        _set_adapter_requires_grad(raw_model, adapter_name, requires_grad=False)
        raw_model.set_adapter(adapter_name)
        yield
    finally:
        raw_model.set_adapter(fallback_adapter_name)
        raw_model.delete_adapter(adapter_name)
        raw_model.set_adapter(fallback_adapter_name)


def _ensure_unsloth_training_mode(model, use_gradient_checkpointing=True):
    raw_model = _unwrap_model(model)

    FastLanguageModel.for_training(raw_model, use_gradient_checkpointing=use_gradient_checkpointing)
    return model


def _extract_thinking(text: str):
    text = text.strip()
    for pattern in (r"<think>(.*?)</think>",):
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            cleaned = re.sub(r"</?think(?:ing)?>", "", match.group(1), flags=re.IGNORECASE)
            cleaned = re.split(r"\bThe answer now is\b", cleaned, maxsplit=1, flags=re.IGNORECASE)[0]
            cleaned = cleaned.strip()
            return cleaned or None
    closing_tag_match = re.search(r"^(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    if closing_tag_match:
        cleaned = re.sub(r"</?think(?:ing)?>", "", closing_tag_match.group(1), flags=re.IGNORECASE)
        cleaned = re.split(r"\bThe answer now is\b", cleaned, maxsplit=1, flags=re.IGNORECASE)[0]
        cleaned = cleaned.strip()
        return cleaned or None
    cleaned = re.split(r"\bThe answer now is\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
    cleaned = re.sub(r"</?think(?:ing)?>", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned or None


def _is_valid_latent_thinking(text: str) -> bool:
    if not text:
        return False
    if len(text.split()) < 5:
        return False
    return not any(pattern.search(text) for pattern in _LATENT_THINKING_REJECT_PATTERNS)


def _normalize_requests(requests: List[Dict]) -> List[Dict]:
    normalized = []
    for r in requests:
        subject = r.get("subject")
        question = r.get("question")
        prompt_template = r.get("prompt")
        question_cloze = f"{prompt_template.format(subject)} ?" if prompt_template is not None and subject is not None else None
        article = r.get("article")
        old_answer = r.get("target_true")["str"] if not (isinstance(r.get("target_true"), str)) else r.get("target_true")
        new_answer = r.get("target_new")["str"] if not (isinstance(r.get("target_new"), str)) else r.get("target_new")
        if subject is None or question is None or article is None or new_answer is None:
            raise ValueError("CausalEdit requires subject/question/target_new/article in each request")
        normalized.append(
            {
                "subject": subject,
                "prompt": prompt_template,
                "question": question,
                "reference": article,
                "new_answer": new_answer,
                "old_answer": old_answer,
                "question_cloze": question_cloze,
                "question_source": "base",
                "source_idx": None,
            }
        )
    return normalized


def _expand_stage3_requests(requests: List[Dict], hparams=None) -> List[Dict]:
    expanded = []
    source_counts = {"base": 0, "rephrase": 0, "learning": 0}
    variant_counts = {
        "aligned_what_changed": 0,
        "aligned_think_step": 0,
        "aligned_direct": 0,
        "stage2_sft": 0,
    }

    for r in requests:
        subject = r.get("subject")
        base_question = r.get("question")
        prompt_template = r.get("prompt")
        article = r.get("article", r.get("reference"))
        old_answer = r.get("old_answer")
        if old_answer is None and r.get("target_true") is not None:
            old_answer = r.get("target_true")["str"] if not isinstance(r.get("target_true"), str) else r.get("target_true")
        new_answer = r.get("new_answer")
        if new_answer is None and r.get("target_new") is not None:
            new_answer = r.get("target_new")["str"] if not isinstance(r.get("target_new"), str) else r.get("target_new")
        if subject is None or base_question is None or article is None:
            raise ValueError("CausalEdit stage3 requires subject/question/article in each request")

        def _render_student_user_content(question: str, prompt_variant: str) -> str:
            if prompt_variant == "aligned_what_changed":  # Same distribution as test, with stronger directional hint for why (-> why did it change?)
                return f"{question} What changed, and why? Let's think step by step."
            if prompt_variant == "aligned_think_step":  # Same distribution as test
                return f"{question} Let's think step by step."
            if prompt_variant == "aligned_direct":  # Preserve direct-answer capability
                return f"{question} Direct answer:"
            if prompt_variant == "stage2_sft":  # Consistent with SFT cold start data distribution, always keep some rollouts guided
                return _build_stage2_sft_user_content(question)
            raise ValueError(f"Unsupported prompt_variant: {prompt_variant}")

        def _append_variant(
            *,
            question: str,
            question_source: str,
            source_idx: int | None,
            prompt_variant: str,
        ) -> None:
            student_user_content = _render_student_user_content(question, prompt_variant)
            teacher_messages = _build_latent_thinking_messages(
                question=question,
                new_answer=new_answer,
                old_answer=old_answer,
                reference=article,
                user_content=student_user_content,
                system_question=base_question,
            )
            expanded.append(
                {
                    "subject": subject,
                    "prompt": prompt_template,
                    "question": question,
                    "reference": article,
                    "new_answer": new_answer,
                    "old_answer": old_answer,
                    "question_source": question_source,
                    "source_idx": source_idx,
                    "prompt_variant": prompt_variant,
                    "student_prompt_msg": [{"role": "user", "content": student_user_content}],
                    "teacher_prompt_msg": teacher_messages,
                }
            )
            source_counts[question_source] += 1
            variant_counts[prompt_variant] += 1

        questions_to_expand = [
            (base_question, "base", None),
            (f"{prompt_template.format(subject)} ?", "base", None),
        ]
        if _EDITING_MODE == "causal":
            prompt_variants = [
                "aligned_what_changed",
                "aligned_what_changed",
                "aligned_think_step",
                "aligned_think_step",
                "aligned_direct",
                "aligned_direct",
                # "aligned_direct",
                "stage2_sft",
            ]
        else:
            prompt_variants = [
                "aligned_think_step",
                "aligned_think_step",
                "aligned_direct",
                "aligned_direct",
                "aligned_direct",
                "stage2_sft",
            ]

        for question, question_source, source_idx in questions_to_expand:
            for prompt_variant in prompt_variants:
                _append_variant(
                    question=question,
                    question_source=question_source,
                    source_idx=source_idx,
                    prompt_variant=prompt_variant,
                )
        if hparams.stage3_cake_data:
            # ----------------------------------------------------------------
            # cake-related
            cake_prompt_variants = [
                "aligned_what_changed",
                "aligned_what_changed",
                "aligned_think_step",
                "aligned_think_step",
                "aligned_direct",
                "aligned_direct",
                "stage2_sft",
            ]
            # cake dataset augmentation
            for idx, item in enumerate(r.get("rephrase_prompt", [])):
                for prompt_variant in cake_prompt_variants:
                    question = item.get("question")
                    _append_variant(
                        question=question,
                        question_source="rephrase",
                        source_idx=idx,
                        prompt_variant=prompt_variant,
                    )

            for idx, item in enumerate(r.get("learning_prompt", [])):
                for prompt_variant in cake_prompt_variants:
                    question = item.get("question")
                    _append_variant(
                        question=question,
                        question_source="learning",
                        source_idx=idx,
                        prompt_variant=prompt_variant,
                    )
            # cake-related
            # ----------------------------------------------------------------

    logger.info(
        "Expanded CausalEdit stage3 requests: "
        f"total={len(expanded)}, "
        f"base={source_counts['base']}, "
        f"rephrase={source_counts['rephrase']}, "
        f"learning={source_counts['learning']}, "
        f"aligned_what_changed={variant_counts['aligned_what_changed']}, "
        f"aligned_think_step={variant_counts['aligned_think_step']}, "
        f"aligned_direct={variant_counts['aligned_direct']}, "
        f"stage2_sft={variant_counts['stage2_sft']}"
    )
    return expanded


def _compute_latent_rewards_batch(model, tokenizer, question, latent_thoughts, new_answer):
    """
    Batch compute implicit thinking rewards

    Args:
        model: Model
        tokenizer: Tokenizer
        question: Question
        latent_thoughts: List of latent thinking contents
        new_answer: New answer

    Returns:
        rewards: List of reward values
    """
    messages_list = []
    completion_list = []
    target_part_list = []

    for latent_thought in latent_thoughts:
        user_instruction = (
            f'{question} You need to think before answering the question (place it in the <think> tag), and end with "The answer now is [YOUR ANSWER]".'
        )
        messages_q_z = [
            {
                "role": "user",
                "content": user_instruction,
            },
        ]
        full_completion = _render_completion_with_think(
            latent_thought,
            new_answer,
            tokenizer=tokenizer,
        )
        target_part = _render_answer_text(new_answer)

        messages_list.append(messages_q_z)
        completion_list.append(full_completion)
        target_part_list.append(target_part)

    # Batch compute perplexity
    with _my_context_manager(model):
        perplexities = _get_perplexity_batch(
            model_to_use=model,
            tokenizer_to_use=tokenizer,
            messages_list=messages_list,
            completion_list=completion_list,
            target_part_list=target_part_list,
        )

    # Convert perplexities to rewards
    rewards = [torch.exp(-torch.log(torch.tensor(p))).item() for p in perplexities]
    return rewards


class UnslothGenerationBatchController:
    def __init__(self, initial_batch_size: Optional[int] = None) -> None:
        self.current_batch_size = max(1, int(initial_batch_size)) if initial_batch_size is not None else None
        self.growth_factor = 1.5

    def get_batch_size(self, requested_batch_size: int) -> int:
        requested_batch_size = max(1, int(requested_batch_size))
        if self.current_batch_size is None:
            self.current_batch_size = requested_batch_size
        return min(requested_batch_size, self.current_batch_size)

    def report_success(self, successful_batch_size: int) -> None:
        successful_batch_size = max(1, int(successful_batch_size))
        if self.current_batch_size is None:
            self.current_batch_size = successful_batch_size
        else:
            self.current_batch_size = max(
                successful_batch_size + 1,
                int(successful_batch_size * self.growth_factor),
            )

    def report_oom(self, failed_batch_size: int) -> int:
        failed_batch_size = max(1, int(failed_batch_size))
        next_batch_size = max(1, int(failed_batch_size * 0.8))
        self.current_batch_size = next_batch_size
        return next_batch_size


class RewardBatchController:
    def __init__(self, initial_batch_size: Optional[int] = None) -> None:
        self.current_batch_size = max(1, int(initial_batch_size)) if initial_batch_size is not None else None
        self.growth_factor = 1.5

    def get_batch_size(self, requested_batch_size: int) -> int:
        requested_batch_size = max(1, int(requested_batch_size))
        if self.current_batch_size is None:
            self.current_batch_size = requested_batch_size
        return min(requested_batch_size, self.current_batch_size)

    def report_success(self, successful_batch_size: int) -> None:
        successful_batch_size = max(1, int(successful_batch_size))
        if self.current_batch_size is None:
            self.current_batch_size = successful_batch_size
        else:
            self.current_batch_size = max(
                successful_batch_size + 1,
                int(successful_batch_size * self.growth_factor),
            )

    def report_oom(self, failed_batch_size: int) -> int:
        failed_batch_size = max(1, int(failed_batch_size))
        next_batch_size = max(1, int(failed_batch_size * 0.8))
        self.current_batch_size = next_batch_size
        return next_batch_size


def _compute_best_latent_thought(
    *,
    model: Any,
    tokenizer: Any,
    question: str,
    new_answer: str,
    latent_thoughts: List[str],
    reward_batch_controller: RewardBatchController,
) -> tuple[int, List[float]]:
    rewards = []
    remaining_rewards = len(latent_thoughts)
    reward_batch_size = reward_batch_controller.get_batch_size(len(latent_thoughts))
    while remaining_rewards > 0:
        current_batch_size = min(remaining_rewards, reward_batch_size)
        batch_latent_thoughts = latent_thoughts[len(rewards) : len(rewards) + current_batch_size]
        try:
            batch_rewards = _compute_latent_rewards_batch(
                model=model,
                tokenizer=tokenizer,
                question=question,
                latent_thoughts=batch_latent_thoughts,
                new_answer=new_answer,
            )
            rewards.extend(batch_rewards)
            remaining_rewards -= current_batch_size
            reward_batch_controller.report_success(current_batch_size)
        except RuntimeError as exc:
            if not _is_probably_oom_error(exc) or current_batch_size <= 1:
                raise
            next_batch_size = reward_batch_controller.report_oom(current_batch_size)
            reward_batch_size = next_batch_size
            logger.warning(
                "Stage2 reward batch_size=%s OOM; fallback to %s and retry. error=%s",
                current_batch_size,
                next_batch_size,
                exc,
            )
            torch.cuda.empty_cache()

    best_idx = max(range(len(rewards)), key=lambda j: rewards[j])
    return best_idx, rewards


def _build_latent_thinking_system_prompt(
    *,
    anchor_question: str,
    new_answer: str,
    old_answer: str,
    reference: str,
) -> str:
    if _EDITING_MODE == "causal":
        cheat_sheet_lines = [f"- **Question**: {anchor_question}"]
        cheat_sheet_lines.append(f"- **Old Fact**: {old_answer}")
        cheat_sheet_lines.append(f"- **Target New Answer**: {new_answer}")
        cheat_sheet_lines.append(f"- **Reference Article**: <article>{reference}</article>")
        cheat_sheet = "\n".join(cheat_sheet_lines)
        return f"""You are an AI generating reasoning paths for a knowledge update.

### TASK
You will receive a QUESTION. I am providing you with the TARGET ANSWER below. 
Your job is to generate a short causal update reasoning chain that leads to this answer.

Treat the reference as your updated internal knowledge.
Do NOT mention the article, source, search, user, or hidden context.
Do NOT express uncertainty, hedging, public-information wording, or that the answer is unclear.

### HIDDEN CONTEXT Cheat Sheet (Use this to guide your thought, but DO NOT mention you were told this. The information below is ABSOLUTELY CORRECT.)
{cheat_sheet}
"""

    cheat_sheet_lines = [f"- **Question**: {anchor_question}"]
    cheat_sheet_lines.append(f"- **Target New Answer**: {new_answer}")
    cheat_sheet_lines.append(f"- **Previous Answer (for disambiguation only)**: {old_answer}")
    cheat_sheet = "\n".join(cheat_sheet_lines)
    return f"""You are an AI generating short reasoning paths for an updated answer.

### TASK
You will receive a QUESTION. I am providing you with the TARGET ANSWER below.
Your job is to use the TARGET ANSWER to answer the question.

Do NOT mention any article, source, search, user, or hidden context.
Do NOT fabricate causes, transitions, or historical background.

### HIDDEN CONTEXT Cheat Sheet (Use this to guide your thought, but DO NOT mention you were told this. The information below is ABSOLUTELY CORRECT.)
{cheat_sheet}
"""


def _build_latent_thinking_messages(
    question: str,
    new_answer: str,
    old_answer: str | None,
    reference: str,
    tokenizer: Any | None = None,
    user_content: str | None = None,
    system_question: str | None = None,
):
    if _EDITING_MODE == "causal":
        one_shot_example = """
### Format Example
<think>
Old fact: ...
Reason for change: ...
Updated fact: ...
Derivation: ...
</think>
The answer now is [TARGET ANSWER]
"""
    else:
        one_shot_example = """
### Format Example
<think>
Updated answer: ...
</think>
The answer now is [TARGET ANSWER]
"""

    if system_question is None:
        system_question = question
    system_prompt = _build_latent_thinking_system_prompt(
        anchor_question=system_question,
        new_answer=new_answer,
        old_answer=old_answer,
        reference=reference,
    )

    # 3. User Input: place the structured output requirements for this task
    if user_content is None:
        if _EDITING_MODE == "causal":
            user_content = f"""Question: {question}

{one_shot_example}

Generate the reasoning inside <think>...</think> using exactly these parts:
- Old fact: ...
- Reason for change: ...
- Updated fact: ...
- Derivation: ...

Only include these parts in <think>.
The "Reason for change" part may include more details to explain why the old fact changed.
Then end exactly with:
The answer now is [the target answer]"""
        else:
            user_content = f"""Question: {question}

{one_shot_example}

Generate the reasoning inside <think>...</think> using exactly these parts:
- Updated answer: ...

Only include these parts in <think>.
Do not mention causes, changes, sources, articles, or hidden context.
Then end exactly with:
The answer now is [the target answer]"""

    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    return prompt


def _unsloth_generate_candidates(
    *,
    model: Any,
    tokenizer: Any,
    question: str,
    new_answer: str,
    old_answer: str | None,
    reference: str,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    batch_controller: Optional[UnslothGenerationBatchController] = None,
) -> List[str]:
    messages = _build_latent_thinking_messages(
        question=question,
        new_answer=new_answer,
        old_answer=old_answer,
        reference=reference,
        tokenizer=tokenizer,
    )
    model.eval()
    original_padding_side = getattr(tokenizer, "padding_side", "right")
    try:
        tokenizer.padding_side = "left"
        prompt_inputs = _apply_chat_template_with_native_thinking(
            tokenizer,
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(_get_model_device(model))
    finally:
        tokenizer.padding_side = original_padding_side

    generated_texts: List[str] = []
    remaining = num_samples
    generation_batch_size = batch_controller.get_batch_size(num_samples) if batch_controller is not None else num_samples
    while remaining > 0:
        current_batch_size = min(remaining, generation_batch_size)
        try:
            with torch.inference_mode():
                if SamplingParams is None or TokensPrompt is None:
                    raise RuntimeError("Unsloth fast_generate requires vllm SamplingParams/TokensPrompt.")

                single_prompt_ids = prompt_inputs["input_ids"][0].tolist()
                prompt_token_ids_batch = [TokensPrompt(prompt_token_ids=list(single_prompt_ids)) for _ in range(current_batch_size)]
                sampling_params = SamplingParams(
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_new_tokens,
                    detokenize=True,
                )
                outputs = model.fast_generate(
                    prompt_token_ids_batch,
                    sampling_params=sampling_params,
                )
        except RuntimeError as exc:
            if batch_controller is None or not _is_probably_oom_error(exc) or current_batch_size <= 1:
                raise
            next_batch_size = batch_controller.report_oom(current_batch_size)
            generation_batch_size = next_batch_size
            logger.warning(
                "Stage2 fast_generate batch_size=%s OOM; fallback to %s and retry. error=%s",
                current_batch_size,
                next_batch_size,
                exc,
            )
            torch.cuda.empty_cache()
            continue

        generated_texts.extend(output.outputs[0].text for output in outputs)
        remaining -= current_batch_size
        if batch_controller is not None:
            batch_controller.report_success(current_batch_size)
        del outputs

    return generated_texts


def _select_best_latent_thought_from_candidates(
    *,
    model: Any,
    tokenizer: Any,
    question: str,
    new_answer: str,
    candidate_texts: List[str],
    reward_batch_controller: RewardBatchController,
    replica_idx: int | None = None,
) -> str:
    latent_thoughts = []
    for text in candidate_texts:
        z = _extract_thinking(text)
        if z is not None and _is_valid_latent_thinking(z):
            latent_thoughts.append(z)

    if not latent_thoughts:
        replica_note = f", replica_idx={replica_idx}" if replica_idx is not None else ""
        raise ValueError(f"No valid latent thoughts generated under <think> format. question={question}{replica_note}")

    best_idx, rewards = _compute_best_latent_thought(
        model=model,
        tokenizer=tokenizer,
        question=question,
        new_answer=new_answer,
        latent_thoughts=latent_thoughts,
        reward_batch_controller=reward_batch_controller,
    )
    logger.info(
        "Stage2 rejection_sampling: replica_idx=%s best_idx=%s best_reward=%.6f mean_reward=%.6f",
        replica_idx,
        best_idx,
        rewards[best_idx],
        sum(rewards) / len(rewards),
    )
    return latent_thoughts[best_idx]


def _resolve_latent_thoughts_for_request(
    *,
    model: Any,
    tokenizer: Any,
    request: Dict[str, Any],
    hparams: Any,
    generation_batch_controller: UnslothGenerationBatchController,
    reward_batch_controller: RewardBatchController,
) -> tuple[Dict[int, str], int]:
    replica_indices = list(range(hparams.stage2_data_replica_times))
    cache_keys = {
        replica_idx: _get_cache_key_explict(
            model.___my_real_repod_id,
            f"{request['question']}_{str(replica_idx)}",
            request["reference"],
        )
        for replica_idx in replica_indices
    }

    z_star_by_replica = {}
    remaining_missing = []
    for replica_idx in tqdm(replica_indices, desc=f"Processing {request['subject']}"):
        cache_key = cache_keys[replica_idx]
        if cache_key in _rejection_sampling_cache:
            z_star_by_replica[replica_idx] = _rejection_sampling_cache[cache_key]["z_star"]
        else:
            remaining_missing.append(replica_idx)

    cache_misses = len(remaining_missing)
    if not remaining_missing:
        return z_star_by_replica, cache_misses

    total_to_generate = len(remaining_missing) * hparams.stage2_num_samples
    candidates = _unsloth_generate_candidates(
        model=model,
        tokenizer=tokenizer,
        question=request["question"],
        new_answer=request["new_answer"],
        old_answer=request.get("old_answer"),
        reference=request["reference"],
        num_samples=total_to_generate,
        max_new_tokens=hparams.stage2_max_new_tokens,
        temperature=0.9,
        top_p=0.95,
        batch_controller=generation_batch_controller,
    )

    for i, replica_idx in enumerate(remaining_missing):
        chunk = candidates[i * hparams.stage2_num_samples : (i + 1) * hparams.stage2_num_samples]
        z_star = _select_best_latent_thought_from_candidates(
            model=model,
            tokenizer=tokenizer,
            question=request["question"],
            new_answer=request["new_answer"],
            candidate_texts=chunk,
            reward_batch_controller=reward_batch_controller,
            replica_idx=replica_idx,
        )
        cache_key = cache_keys[replica_idx]
        _rejection_sampling_cache[cache_key] = {
            "z_star": z_star,
            "new_answer": request["new_answer"],
            "old_answer": request.get("old_answer"),
            "question": request["question"],
        }
        z_star_by_replica[replica_idx] = z_star

    return z_star_by_replica, cache_misses


def _sample_noncausal_latent_thoughts(
    model,
    tokenizer,
    question: str,
    new_answer: str,
    num_samples: int,
    max_new_tokens: int,
):
    model.config.use_cache = True
    model.eval()

    messages = _build_latent_thinking_messages(
        question=question,
        new_answer=new_answer,
        old_answer=None,
        reference="",
        tokenizer=tokenizer,
    )
    inputs = _apply_chat_template_with_native_thinking(
        tokenizer,
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.inference_mode():
        if SamplingParams is None or TokensPrompt is None:
            raise RuntimeError("Unsloth fast_generate requires vllm SamplingParams/TokensPrompt.")
        prompt_token_ids = inputs["input_ids"][0].tolist()
        prompt_token_ids_batch = [TokensPrompt(prompt_token_ids=list(prompt_token_ids)) for _ in range(num_samples)]
        sampling_params = SamplingParams(
            temperature=0.9,
            top_p=0.95,
            max_tokens=max_new_tokens,
            detokenize=True,
        )
        outputs = model.fast_generate(
            prompt_token_ids_batch,
            sampling_params=sampling_params,
        )
        generated_texts = [output.outputs[0].text for output in outputs]
    latent_thoughts = []
    for t in generated_texts:
        z = _extract_thinking(t)
        if z is not None:
            latent_thoughts.append(z)

    model.train()
    return latent_thoughts


def _prepare_sft_dataset(sft_data_list, tokenizer):
    def format_sft_dataset(example):
        q = example["question"]
        z = example["z_star"]
        a = example["new_answer"]
        user_content = _build_stage2_sft_user_content(q)
        prompt = [
            {
                "role": "user",
                "content": user_content,
            },
        ]

        completion = [
            {
                "role": "assistant",
                "content": _render_completion_with_think(
                    z,
                    a,
                    tokenizer=tokenizer,
                ),
            }
        ]

        return {"prompt": prompt, "completion": completion}

    dataset = Dataset.from_list(sft_data_list)
    dataset = dataset.map(format_sft_dataset, remove_columns=["question", "z_star", "new_answer"])
    logger.info(f"SFT data example:\n{dataset[0]}")
    return dataset


def _build_stage2_sft_user_content(question: str) -> str:
    if _EDITING_MODE == "causal":
        one_shot_example = """
### Format Example
<think>
Old fact: ...
Reason for change: ...
Updated fact: ...
Derivation: ...
</think>
The answer now is [TARGET ANSWER]
"""
        return f"""Question: {question}

{one_shot_example}

Generate the reasoning inside <think>...</think> using exactly these parts:
- Old fact: ...
- Reason for change: ...
- Updated fact: ...
- Derivation: ...

Only include these parts in <think>.
The "Reason for change" part may include more details to explain why the old fact changed.
Then end exactly with:
The answer now is [the target answer]"""

    one_shot_example = """
### Format Example
<think>
Updated answer: ...
</think>
The answer now is [TARGET ANSWER]
"""
    return f"""Question: {question}

{one_shot_example}

Generate the reasoning inside <think>...</think> using exactly these parts:
- Updated answer: ...

Only include these parts in <think>.
Do not mention causes, changes, sources, articles, or hidden context.
Then end exactly with:
The answer now is [the target answer]"""


def _get_lora_model(hparams, model):
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    from unsloth import FastLanguageModel

    model = FastLanguageModel.get_peft_model(
        model,
        r=hparams.stage2_lora_rank,
        target_modules=target_modules,
        lora_alpha=hparams.stage2_lora_rank * 2,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    model.print_trainable_parameters()
    return model


@dataclass
class _ContextDistillationDataCollator:
    tokenizer: Any
    max_length: int = 1024

    def __call__(self, features: List[Dict[str, Any]]):
        self.tokenizer.padding_side = "left"
        teacher_prompts = [f["teacher_prompt_msg"] for f in features]
        teacher_encodings = _apply_chat_template_with_native_thinking(
            self.tokenizer,
            teacher_prompts,
            padding=True,
            padding_side="left",
            return_tensors="pt",
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
        )
        batch_list = []
        for i in range(len(features)):
            batch_list.append(
                {
                    "prompt": features[i]["student_prompt_msg"],
                    "teacher_prompt_ids": teacher_encodings["input_ids"][i],
                    "teacher_prompt_mask": teacher_encodings["attention_mask"][i],
                }
            )
        return batch_list


def _expand_token_level_advantages_for_unsloth(
    *,
    advantages: torch.Tensor,
    prompt_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    max_left_pad: Any,
) -> torch.Tensor:
    if advantages.dim() != 2:
        return advantages
    if advantages.shape != completion_mask.shape:
        raise ValueError(
            "Token-level advantages must align with the unexpanded completion mask: "
            f"advantages={tuple(advantages.shape)} completion_mask={tuple(completion_mask.shape)}"
        )

    if isinstance(max_left_pad, torch.Tensor):
        max_left_pad = int(max_left_pad.reshape(-1)[0].item()) if max_left_pad.numel() > 0 else 0
    else:
        max_left_pad = int(max_left_pad or 0)

    prompt_mask_i64 = prompt_mask.to(dtype=torch.int64)
    if max_left_pad <= 0:
        max_left_pad = int((prompt_mask.shape[1] - prompt_mask_i64.sum(dim=1)).max().item())
    if max_left_pad <= 0:
        return advantages

    completion_mask_i64 = completion_mask.to(dtype=torch.int64)
    left_pad_tokens_per_prompt = prompt_mask.shape[1] - prompt_mask_i64.sum(dim=1)
    valid_lengths = completion_mask_i64.sum(dim=1)
    expanded_advantages = advantages.new_zeros((advantages.shape[0], advantages.shape[1] + max_left_pad))

    prefix_lengths = (max_left_pad - left_pad_tokens_per_prompt).to(dtype=torch.int64)
    for row_idx in range(advantages.shape[0]):
        valid_len = int(valid_lengths[row_idx].item())
        if valid_len == 0:
            continue
        prefix_len = int(prefix_lengths[row_idx].item())
        expanded_advantages[row_idx, prefix_len : prefix_len + valid_len] = advantages[row_idx, :valid_len]
    return expanded_advantages


def _calculate_pad_tokens_in_prompt(
    input_ids: torch.Tensor,
    logits_to_keep: int,
    pad_token_id: int,
) -> torch.Tensor:
    if logits_to_keep >= input_ids.shape[1]:
        raise ValueError("logits_to_keep must be smaller than the sequence length.")
    prompt_section = input_ids[:, :-logits_to_keep]
    return (prompt_section == pad_token_id).sum(dim=1)


def _create_completion_attention_mask(
    completion_input_ids: torch.Tensor,
    left_pad_tokens_per_prompt: torch.Tensor,
    max_left_pad: int,
    pad_token_id: int,
) -> torch.Tensor:
    device = completion_input_ids.device
    num_tokens_to_mask = max_left_pad - left_pad_tokens_per_prompt
    indices = torch.arange(completion_input_ids.shape[1], device=device).unsqueeze(0)
    shift_mask = indices >= num_tokens_to_mask.unsqueeze(1)
    non_padding_mask = completion_input_ids != pad_token_id
    return shift_mask & non_padding_mask


def _left_pack_padding(tensor: torch.Tensor, pad_id: int) -> torch.Tensor:
    mask = tensor != pad_id
    sorted_indices = torch.argsort(mask, dim=1, descending=True, stable=True)
    return torch.gather(tensor, 1, sorted_indices)


def _align_logprobs_with_mask(
    logprob_tensor: torch.Tensor,
    attention_mask: torch.Tensor,
    pad_value: float = 0.0,
) -> torch.Tensor:
    device = logprob_tensor.device
    batch_size, logprob_seq_len = logprob_tensor.shape
    padded_logprobs = torch.full(
        attention_mask.shape,
        fill_value=pad_value,
        dtype=logprob_tensor.dtype,
        device=device,
    )
    left_pad_counts = torch.argmax(attention_mask.to(torch.int64), dim=1)
    cols = torch.arange(logprob_seq_len, device=device)
    dest_indices = left_pad_counts.unsqueeze(1) + cols
    row_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(dest_indices)
    valid_mask = dest_indices < attention_mask.shape[1]
    padded_logprobs[row_indices[valid_mask], dest_indices[valid_mask]] = logprob_tensor[valid_mask]
    return padded_logprobs


def _selective_log_softmax_local(
    logits: torch.Tensor,
    index: torch.Tensor,
) -> torch.Tensor:
    logits = logits.to(torch.float32)
    selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    return selected_logits - torch.logsumexp(logits, dim=-1)


def _collapse_packed_token_values_for_unsloth(
    *,
    values: torch.Tensor,
    prompt_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    max_left_pad: int,
) -> torch.Tensor:
    prompt_mask_i64 = prompt_mask.to(dtype=torch.int64)
    completion_mask_i64 = completion_mask.to(dtype=torch.int64)
    left_pad_tokens_per_prompt = prompt_mask.shape[1] - prompt_mask_i64.sum(dim=1)
    valid_lengths = completion_mask_i64.sum(dim=1)
    collapsed = values.new_zeros(completion_mask.shape, dtype=values.dtype)
    prefix_lengths = (max_left_pad - left_pad_tokens_per_prompt).to(dtype=torch.int64)

    for row_idx in range(values.shape[0]):
        valid_len = int(valid_lengths[row_idx].item())
        if valid_len == 0:
            continue
        prefix_len = int(prefix_lengths[row_idx].item())
        collapsed[row_idx, :valid_len] = values[row_idx, prefix_len : prefix_len + valid_len]
    return collapsed


def _compute_unsloth_aligned_importance_ratio(
    *,
    old_per_token_logps: torch.Tensor | None,
    sampling_per_token_logps: torch.Tensor | None,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    pad_token_id: int,
    cap: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if old_per_token_logps is None or sampling_per_token_logps is None:
        return None, None

    logits_to_keep = completion_ids.size(1)
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    left_pad_tokens_per_prompt = _calculate_pad_tokens_in_prompt(input_ids, logits_to_keep, pad_token_id)
    max_left_pad = int(left_pad_tokens_per_prompt.max().item())

    if max_left_pad > 0:
        packed_input_ids = _left_pack_padding(input_ids, pad_token_id)
        packed_target_ids = packed_input_ids[:, -(logits_to_keep + max_left_pad) :]
        packed_completion_mask = _create_completion_attention_mask(
            packed_target_ids,
            left_pad_tokens_per_prompt,
            max_left_pad,
            pad_token_id,
        ).to(completion_mask.dtype)
        aligned_sampling_logps = _align_logprobs_with_mask(sampling_per_token_logps, packed_completion_mask)
    else:
        packed_target_ids = completion_ids
        packed_completion_mask = completion_mask.bool()
        aligned_sampling_logps = sampling_per_token_logps

    if old_per_token_logps.dim() == 3:
        packed_old_logits = old_per_token_logps[:, -(packed_target_ids.shape[1] + 1) :, :][:, :-1, :]
        packed_old_logps = _selective_log_softmax_local(packed_old_logits, packed_target_ids)
    elif old_per_token_logps.dim() == 2:
        packed_old_logps = old_per_token_logps[:, -packed_target_ids.shape[1] :]
    else:
        raise ValueError(f"Unsupported old_per_token_logps rank for unsloth-aligned IS correction: {old_per_token_logps.dim()}")

    packed_delta = torch.abs(packed_old_logps - aligned_sampling_logps)
    packed_ratio = torch.exp(packed_old_logps - aligned_sampling_logps)
    packed_ratio = torch.clamp(packed_ratio, max=cap)

    if max_left_pad > 0:
        collapsed_ratio = _collapse_packed_token_values_for_unsloth(
            values=packed_ratio,
            prompt_mask=prompt_mask,
            completion_mask=completion_mask,
            max_left_pad=max_left_pad,
        )
        collapsed_delta = _collapse_packed_token_values_for_unsloth(
            values=packed_delta,
            prompt_mask=prompt_mask,
            completion_mask=completion_mask,
            max_left_pad=max_left_pad,
        )
    else:
        collapsed_ratio = packed_ratio
        collapsed_delta = packed_delta

    return collapsed_ratio, collapsed_delta


class _MyMiniLLMTrainer(MiniLLMTrainer):
    def __init__(
        self,
        *args,
        data_collator,
        teacher_mode: str,
        teacher_adapter_snapshot: dict[str, Any] | None = None,
        student_adapter_name: str = "default",
        teacher_adapter_name: str = "teacher",
        **kwargs,
    ):
        class _UnusedTeacherModel(torch.nn.Module):
            def forward(self, *args, **kwargs):
                raise RuntimeError(
                    "The placeholder teacher model should never be called. "
                    "Teacher logits are computed by adapter switching in _MyMiniLLMTrainer.compute_loss()."
                )

        kwargs.setdefault("teacher_model", _UnusedTeacherModel())
        super().__init__(*args, **kwargs)
        self.data_collator = data_collator
        self.teacher_mode = teacher_mode
        self.teacher_adapter_snapshot = teacher_adapter_snapshot
        self.student_adapter_name = student_adapter_name
        self.teacher_adapter_name = teacher_adapter_name
        self.rkl_top_k = int(getattr(self.args, "rkl_top_k", 0) or 0)
        self.use_forward_kl = bool(getattr(self.args, "use_forward_kl", True))
        self.repetition_ngram_n = int(getattr(self.args, "repetition_ngram_n", 3) or 3)
        self.repetition_tail_window_size = int(getattr(self.args, "repetition_tail_window_size", 48) or 48)
        self.repetition_ngram_ratio_threshold = float(getattr(self.args, "repetition_ngram_ratio_threshold", 0.25))
        seq_temperature_cfg = getattr(self.args, "teacher_seq_ppl_weight_temperature", 1.0)
        token_threshold_cfg = getattr(self.args, "teacher_entropy_mask_threshold", None)
        if token_threshold_cfg is None:
            token_threshold_cfg = getattr(self.args, "teacher_entropy_mask_top_ratio", None)
        self.teacher_seq_ppl_weight_temperature = max(
            float(seq_temperature_cfg) if seq_temperature_cfg is not None else 1.0,
            1e-6,
        )
        self.teacher_entropy_mask_threshold = float(token_threshold_cfg) if token_threshold_cfg is not None else None
        self._teacher_token_entropy_threshold_auto_initialized = False
        self.latest_teacher_entropy_stats: dict[str, float] = {}
        self.latest_rkl_topk_stats: dict[str, float] = {}
        self.latest_repetition_stats: dict[str, float] = {}
        self._running_log_sums: dict[str, float] = {}
        self._running_log_counts: dict[str, int] = {}

    def _build_compact_topk_union_indices(
        self,
        teacher_topk_indices: torch.Tensor,
        student_topk_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Merge teacher / student top-k indices into a compact union.

        The returned union_indices / union_valid_mask both have shape [B, T, 2K],
        but only the first union_size positions are valid; the rest are marked
        invalid by union_valid_mask. This avoids constructing a dense bool mask
        for full-vocab logits.
        """
        if teacher_topk_indices.shape != student_topk_indices.shape:
            raise ValueError("teacher_topk_indices and student_topk_indices must share the same shape")

        combined_indices = torch.cat([teacher_topk_indices, student_topk_indices], dim=-1)
        flat_indices = combined_indices.reshape(-1, combined_indices.shape[-1])
        num_rows, max_union_width = flat_indices.shape

        union_indices = flat_indices.new_zeros((num_rows, max_union_width))
        union_valid_mask = torch.zeros((num_rows, max_union_width), dtype=torch.bool, device=flat_indices.device)
        union_size = torch.zeros((num_rows,), dtype=torch.long, device=flat_indices.device)

        for col_idx in range(max_union_width):
            current_indices = flat_indices[:, col_idx]
            if col_idx == 0:
                is_new_index = torch.ones(num_rows, dtype=torch.bool, device=flat_indices.device)
            else:
                duplicate_mask = union_valid_mask[:, :col_idx] & (union_indices[:, :col_idx] == current_indices.unsqueeze(1))
                is_new_index = ~duplicate_mask.any(dim=1)

            if not is_new_index.any():
                continue

            row_indices = torch.nonzero(is_new_index, as_tuple=False).squeeze(-1)
            target_slots = union_size[row_indices]
            union_indices[row_indices, target_slots] = current_indices[row_indices]
            union_valid_mask[row_indices, target_slots] = True
            union_size[row_indices] += 1

        union_indices = union_indices.view_as(combined_indices)
        union_valid_mask = union_valid_mask.view_as(combined_indices)
        union_size = union_size.view(*combined_indices.shape[:-1])
        return union_indices, union_valid_mask, union_size

    def _accumulate_running_stats(self, stats: dict[str, float]) -> None:
        for key, value in stats.items():
            if value is None:
                continue
            self._running_log_sums[key] = self._running_log_sums.get(key, 0.0) + float(value)
            self._running_log_counts[key] = self._running_log_counts.get(key, 0) + 1

    def _flush_running_stats(self) -> dict[str, float]:
        averaged = {}
        for key, total in self._running_log_sums.items():
            count = self._running_log_counts.get(key, 0)
            if count > 0:
                averaged[key] = total / count
        self._running_log_sums = {}
        self._running_log_counts = {}
        return averaged

    def _prepare_repetition_filter(
        self,
        *,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        n = max(int(self.repetition_ngram_n), 1)
        tail_window_size = max(int(self.repetition_tail_window_size), n)
        threshold = float(self.repetition_ngram_ratio_threshold)
        batch_size = completion_ids.shape[0]
        ratios = completion_ids.new_zeros(batch_size, dtype=torch.float32)
        valid_lengths = completion_mask.to(dtype=torch.int64).sum(dim=1)

        for batch_idx in range(batch_size):
            valid_len = int(valid_lengths[batch_idx].item())
            if valid_len < n:
                continue
            tail_start = max(0, valid_len - tail_window_size)
            token_seq = completion_ids[batch_idx, tail_start:valid_len].tolist()
            total_ngrams = len(token_seq) - n + 1
            ngrams = [tuple(token_seq[pos : pos + n]) for pos in range(total_ngrams)]
            unique_ngrams = len(set(ngrams))
            ratios[batch_idx] = 1.0 - (unique_ngrams / max(total_ngrams, 1))

        sequence_keep_mask = ratios <= threshold
        ratio_quantiles = torch.quantile(
            ratios.float(),
            torch.tensor([0.25, 0.5, 0.75], device=ratios.device),
        )
        self.latest_repetition_stats = {
            "repetition_ngram_n_cfg": float(n),
            "repetition_tail_window_size_cfg": float(tail_window_size),
            "repetition_ngram_ratio_threshold_cfg": threshold,
            "repetition_ngram_ratio_max": ratios.max().item(),
            "repetition_ngram_ratio_p75": ratio_quantiles[2].item(),
            "repetition_ngram_ratio_p50": ratio_quantiles[1].item(),
            "repetition_ngram_ratio_p25": ratio_quantiles[0].item(),
            "repetition_ngram_ratio_min": ratios.min().item(),
            "repetition_filter_ratio_actual": (~sequence_keep_mask).float().mean().item(),
        }
        self._accumulate_running_stats(self.latest_repetition_stats)
        return sequence_keep_mask

    @torch.no_grad()
    def _compute_teacher_sequence_ppl_weight(
        self,
        *,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        base_valid_mask = mask.bool()
        teacher_log_probs = F.log_softmax(teacher_logits.float(), dim=-1)
        teacher_log_probs_on_labels = torch.gather(teacher_log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        valid_token_count = base_valid_mask.float().sum(dim=-1).clamp_min(1.0)
        seq_nll = -(teacher_log_probs_on_labels * base_valid_mask.float()).sum(dim=-1) / valid_token_count
        seq_ppl = torch.exp(seq_nll)
        valid_seq_mask = base_valid_mask.any(dim=-1)
        valid_seq_ppl = seq_ppl[valid_seq_mask]
        length_normalized_logprob = -seq_nll
        if seq_ppl.numel() > 0:
            if self.num_generations > 0 and seq_ppl.numel() % self.num_generations == 0:
                grouped_scores = (length_normalized_logprob / self.teacher_seq_ppl_weight_temperature).view(-1, self.num_generations)
                grouped_weights = torch.softmax(grouped_scores, dim=-1)
                seq_weight = grouped_weights.reshape(-1)
                logger.info("in group ppl reweight")
            else:
                # seq_weight = torch.softmax(
                #     length_normalized_logprob / self.teacher_seq_ppl_weight_temperature,
                #     dim=0,
                # )
                raise NotImplementedError("not implemented")
        else:
            seq_weight = torch.ones_like(seq_ppl)

        if valid_seq_ppl.numel() > 0:
            seq_ppl_quantiles = torch.quantile(
                valid_seq_ppl.float(),
                torch.tensor([0.25, 0.5, 0.75], device=valid_seq_ppl.device),
            )
            self.latest_teacher_entropy_stats = {
                "teacher_seq_ppl_mean": valid_seq_ppl.mean().item(),
                "teacher_seq_ppl_max": valid_seq_ppl.max().item(),
                "teacher_seq_ppl_p25": seq_ppl_quantiles[0].item(),
                "teacher_seq_ppl_p50": seq_ppl_quantiles[1].item(),
                "teacher_seq_ppl_p75": seq_ppl_quantiles[2].item(),
                "teacher_seq_weight_mean": seq_weight.mean().item(),
                "teacher_seq_weight_max": seq_weight.max().item(),
            }
            self._accumulate_running_stats(self.latest_teacher_entropy_stats)

            valid_seq_weight = seq_weight[valid_seq_mask]
            seq_weight_quantiles = torch.quantile(
                valid_seq_weight.float(),
                torch.tensor([0.25, 0.5, 0.75], device=valid_seq_weight.device),
            )
            self.latest_teacher_entropy_stats = {
                "teacher_seq_weight_mean": valid_seq_weight.mean().item(),
                "teacher_seq_weight_max": valid_seq_weight.max().item(),
                "teacher_seq_weight_min": valid_seq_weight.min().item(),
                "teacher_seq_weight_p25": seq_weight_quantiles[0].item(),
                "teacher_seq_weight_p50": seq_weight_quantiles[1].item(),
                "teacher_seq_weight_p75": seq_weight_quantiles[2].item(),
            }
            self._accumulate_running_stats(self.latest_teacher_entropy_stats)
        return seq_weight.to(teacher_logits.dtype)

    @torch.no_grad()
    def _compute_teacher_entropy_weight(
        self,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        sequence_keep_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute teacher model entropy-based weights.

        This function computes per-token weights based on the teacher model's
        prediction uncertainty (entropy). Sequence-level teacher PPL weights
        are computed separately externally; here we only handle token-level
        entropy filtering. Also supports integration of repetition filtering.

        Args:
            teacher_logits: Teacher model logits output, shape [batch_size, seq_len, vocab_size]
            mask: Valid token mask, shape [batch_size, seq_len], 1 for valid, 0 for padding
            sequence_keep_mask: Sequence-level keep mask, shape [batch_size], True means keep the sample
                from repetition filtering; if None, no repetition filtering is applied

        Returns:
            torch.Tensor: Weight tensor, shape [batch_size, seq_len], values in [0, 1]
                0 means the token is filtered (excluded from loss computation), 1 means fully retained

        Note:
            This function updates self.latest_teacher_entropy_stats with various statistics
        """
        teacher_log_probs = F.log_softmax(teacher_logits.float(), dim=-1)
        teacher_probs = teacher_log_probs.exp()
        entropy = -(teacher_probs * teacher_log_probs).sum(dim=-1)
        keep_weight = torch.ones_like(entropy)

        base_valid_mask = mask.bool()
        keep_weight = keep_weight * base_valid_mask.float()

        valid_entropy = entropy[base_valid_mask]
        if not self._teacher_token_entropy_threshold_auto_initialized and valid_entropy.numel() > 0:
            if self.teacher_entropy_mask_threshold is None:
                self.teacher_entropy_mask_threshold = max(0.8, valid_entropy.mean().item())
            self._teacher_token_entropy_threshold_auto_initialized = True
        seq_keep_mask = sequence_keep_mask.bool() if sequence_keep_mask is not None else torch.ones(mask.shape[0], dtype=torch.bool, device=mask.device)
        token_valid_mask = base_valid_mask & seq_keep_mask.unsqueeze(-1)
        keep_weight = keep_weight * token_valid_mask.to(keep_weight.dtype)

        masked_token_ratio = 0.0

        valid_keep_weight = keep_weight[base_valid_mask]
        if valid_entropy.numel() > 0:
            entropy_quantiles = torch.quantile(
                valid_entropy.float(),
                torch.tensor([0.25, 0.5, 0.75], device=valid_entropy.device),
            )
            token_entropy_stats = {
                "teacher_entropy_mask_threshold_cfg": self.teacher_entropy_mask_threshold,
                "teacher_entropy_mask_ratio_actual": masked_token_ratio,
                "teacher_entropy_mean": valid_entropy.mean().item(),
                "teacher_entropy_max": valid_entropy.max().item(),
                "teacher_entropy_p25": entropy_quantiles[0].item(),
                "teacher_entropy_p50": entropy_quantiles[1].item(),
                "teacher_entropy_p75": entropy_quantiles[2].item(),
                "teacher_entropy_keep_ratio": valid_keep_weight.mean().item(),
            }
            self.latest_teacher_entropy_stats.update(token_entropy_stats)
            self._accumulate_running_stats(token_entropy_stats)

        return keep_weight.to(teacher_logits.dtype)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        running_stats = self._flush_running_stats()
        if running_stats:
            logs = {**logs, **running_stats}
        super().log(logs, start_time=start_time)

    def _single_step_decomposition_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        sequence_keep_mask: torch.Tensor | None = None,
        importance_ratio: torch.Tensor | None = None,
        reduction: str = "batchmean",
    ):
        """
        Compute Single Step Decomposition Loss.

        This function implements an entropy-weighted KL divergence loss based on
        the teacher model, used for knowledge distillation training. Supports
        optional repetition filtering, importance sampling correction, and Top-K
        restriction.

        Args:
            student_logits: Student model logits output, shape [batch_size, seq_len, vocab_size]
            teacher_logits: Teacher model logits output, shape [batch_size, seq_len, vocab_size]
            labels: Completion token ids, shape [batch_size, seq_len]
            mask: Valid token mask, shape [batch_size, seq_len], 1 for valid, 0 for padding
            sequence_keep_mask: Sequence-level keep mask, shape [batch_size], True means keep the sample
                for repetition filtering; if None, no filtering is applied
            importance_ratio: Importance sampling ratio, shape [batch_size, seq_len]
                for vLLM importance sampling correction; if None, no correction is applied
            reduction: Loss reduction method, "batchmean" (default) / "sum" / "mean"
                - batchmean: weighted average over batch
                - sum: direct sum
                - mean: simple average

        Returns:
            torch.Tensor: Scalar loss value
        """
        sequence_weight = self._compute_teacher_sequence_ppl_weight(
            teacher_logits=teacher_logits,
            labels=labels,
            mask=mask,
        )
        entropy_weight = mask.to(dtype=teacher_logits.dtype)
        entropy_weight = entropy_weight * sequence_keep_mask.unsqueeze(-1).to(entropy_weight.dtype)
        if self.rkl_top_k > 0:
            top_k = min(self.rkl_top_k, teacher_logits.shape[-1])
            teacher_topk_indices = teacher_logits.topk(top_k, dim=-1).indices
            student_topk_indices = student_logits.topk(top_k, dim=-1).indices
            union_indices, union_valid_mask, union_size = self._build_compact_topk_union_indices(
                teacher_topk_indices=teacher_topk_indices,
                student_topk_indices=student_topk_indices,
            )

            student_union_logits = torch.gather(student_logits, dim=-1, index=union_indices)
            teacher_union_logits = torch.gather(teacher_logits, dim=-1, index=union_indices)
            neg_inf = torch.finfo(student_union_logits.dtype).min
            student_union_logits = student_union_logits.masked_fill(~union_valid_mask, neg_inf)
            teacher_union_logits = teacher_union_logits.masked_fill(~union_valid_mask, neg_inf)
            student_union_log_probs = F.log_softmax(student_union_logits, dim=-1)
            teacher_union_log_probs = F.log_softmax(teacher_union_logits, dim=-1)
            if self.use_forward_kl:
                pointwise_kl = F.kl_div(
                    student_union_log_probs,
                    teacher_union_log_probs,
                    reduction="none",
                    log_target=True,
                )
            else:
                pointwise_kl = F.kl_div(
                    teacher_union_log_probs,
                    student_union_log_probs,
                    reduction="none",
                    log_target=True,
                )
            pointwise_kl = torch.where(
                union_valid_mask,
                pointwise_kl,
                torch.zeros((), dtype=pointwise_kl.dtype, device=pointwise_kl.device),
            ).sum(dim=-1)
            reg_loss = pointwise_kl
            union_size = union_size.clamp_min(1).to(reg_loss.dtype)
            valid_union_size = union_size[mask] if mask is not None else union_size
            self.latest_rkl_topk_stats = {}
            if valid_union_size.numel() > 0:
                self.latest_rkl_topk_stats = {
                    "union_size_mean": valid_union_size.float().mean().item(),
                }
                self._accumulate_running_stats(self.latest_rkl_topk_stats)
            # reg_loss = reg_loss / union_size
        else:
            self.latest_rkl_topk_stats = {}
            student_log_probs = F.log_softmax(student_logits, dim=-1)
            teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
            if self.use_forward_kl:
                reg_loss = F.kl_div(
                    student_log_probs,
                    teacher_log_probs,
                    reduction="none",
                    log_target=True,
                ).sum(dim=-1)
            else:
                reg_loss = F.kl_div(
                    teacher_log_probs,
                    student_log_probs,
                    reduction="none",
                    log_target=True,
                ).sum(dim=-1)

        weighted_reg_loss = reg_loss * entropy_weight
        weighted_reg_loss = weighted_reg_loss * sequence_weight.unsqueeze(-1).to(weighted_reg_loss.dtype)
        if importance_ratio is not None:
            weighted_reg_loss = weighted_reg_loss * importance_ratio.to(weighted_reg_loss.dtype)

        reg_loss = weighted_reg_loss[mask]
        total_weight = entropy_weight * sequence_weight.unsqueeze(-1).to(entropy_weight.dtype)
        weight_denom = total_weight[mask].sum().clamp_min(1e-8)

        if reduction == "batchmean":
            return reg_loss.sum() / weight_denom
        if reduction == "sum":
            return reg_loss.sum()
        if reduction == "mean":
            return reg_loss.mean()
        return reg_loss

    def _generate_and_score_completions(self, inputs: list[dict[str, torch.Tensor | Any]]):

        if _has_adapter(self.model, self.teacher_adapter_name):
            raise RuntimeError("Teacher adapter must not be attached during generation; vLLM should only see the student adapter.")
        _activate_adapter(self.model, self.student_adapter_name, trainable=True)
        output = super()._generate_and_score_completions(inputs)
        output["teacher_prompt_ids"] = torch.stack([i["teacher_prompt_ids"] for i in inputs]).to(output["completion_ids"].device)
        output["teacher_prompt_mask"] = torch.stack([i["teacher_prompt_mask"] for i in inputs]).to(output["completion_mask"].device)
        return output

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        _ensure_unsloth_training_mode(
            model,
            use_gradient_checkpointing=getattr(self.args, "gradient_checkpointing", True),
        )
        input_ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
        attention_mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
        teacher_input_ids = torch.cat([inputs["teacher_prompt_ids"], inputs["completion_ids"]], dim=1)
        teacher_attention_mask = torch.cat([inputs["teacher_prompt_mask"], inputs["completion_mask"]], dim=1)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        # Compute teacher output first so its inference-only activations do not overlap
        # with the student's autograd graph and inflate peak memory.
        if self.teacher_mode == "adapter":
            if self.teacher_adapter_snapshot is None:
                raise ValueError("Missing teacher_adapter_snapshot for adapter-based teacher mode")
            teacher_context = _temporary_injected_adapter(
                model,
                adapter_snapshot=self.teacher_adapter_snapshot,
                adapter_name=self.teacher_adapter_name,
                fallback_adapter_name=self.student_adapter_name,
            )
        elif self.teacher_mode == "base":
            teacher_context = _temporary_disable_adapter(model)
        else:
            raise ValueError(f"Unsupported teacher mode: {self.teacher_mode}")

        model.eval()
        with teacher_context:
            with torch.inference_mode():
                teacher_outputs = model(
                    input_ids=teacher_input_ids,
                    attention_mask=teacher_attention_mask,
                    use_cache=False,
                )
        _ensure_unsloth_training_mode(
            model,
            use_gradient_checkpointing=getattr(self.args, "gradient_checkpointing", True),
        )
        model.train()
        _activate_student_teacher_state(model, self.student_adapter_name, self.teacher_adapter_name)

        # Compute student output after teacher inference to reduce peak memory.
        student_outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        # Slice the logits for the generated tokens using the inputs["prompts"] lengths
        prompt_lengths = inputs["prompt_ids"].shape[1]
        teacher_prompt_lengths = inputs["teacher_prompt_ids"].shape[1]
        student_logits = student_outputs.logits[:, prompt_lengths - 1 : -1, :]
        teacher_logits = teacher_outputs.logits[:, teacher_prompt_lengths - 1 : -1, :]
        shifted_labels = input_ids[:, prompt_lengths:]

        # Apply temperature scaling
        student_logits = student_logits / self.kd_temperature
        teacher_logits = teacher_logits / self.kd_temperature

        mask = inputs["completion_mask"].bool()
        sequence_keep_mask = self._prepare_repetition_filter(
            completion_ids=inputs["completion_ids"],
            completion_mask=inputs["completion_mask"],
        )
        sequence_keep_mask = sequence_keep_mask.to(mask.device)

        if self.rkl_advantage:
            student_log_probs = F.log_softmax(student_logits, dim=-1)
            teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
            student_log_probs_on_labels = torch.gather(student_log_probs, dim=-1, index=shifted_labels.unsqueeze(-1)).squeeze(-1)
            teacher_log_probs_on_labels = torch.gather(teacher_log_probs, dim=-1, index=shifted_labels.unsqueeze(-1)).squeeze(-1)
            reverse_kl_advantage = self._compute_advantage(
                student_log_probs_on_labels=student_log_probs_on_labels,
                teacher_log_probs_on_labels=teacher_log_probs_on_labels,
                mask=mask,
            ).detach()

            reverse_kl_advantage = _expand_token_level_advantages_for_unsloth(
                advantages=reverse_kl_advantage,
                prompt_mask=inputs["prompt_mask"],
                completion_mask=inputs["completion_mask"],
                max_left_pad=inputs.get("max_left_pad", 0),
            )
            base_advantages = inputs["advantages"]
            if base_advantages.dim() == 1:
                base_advantages = base_advantages.unsqueeze(1)
            elif base_advantages.dim() == 2:
                base_advantages = _expand_token_level_advantages_for_unsloth(
                    advantages=base_advantages,
                    prompt_mask=inputs["prompt_mask"],
                    completion_mask=inputs["completion_mask"],
                    max_left_pad=inputs.get("max_left_pad", 0),
                )
            inputs["advantages"] = base_advantages + reverse_kl_advantage

        loss = None

        # Compute loss
        if self.single_step_decomposition:
            importance_ratio = None
            old_per_token_logps = inputs.get("old_per_token_logps")
            sampling_per_token_logps = inputs.get("sampling_per_token_logps")
            importance_delta = None
            if self.use_vllm and self.vllm_importance_sampling_correction:
                importance_ratio, importance_delta = _compute_unsloth_aligned_importance_ratio(
                    old_per_token_logps=old_per_token_logps,
                    sampling_per_token_logps=sampling_per_token_logps,
                    prompt_ids=inputs["prompt_ids"],
                    completion_ids=inputs["completion_ids"],
                    prompt_mask=inputs["prompt_mask"],
                    completion_mask=inputs["completion_mask"],
                    pad_token_id=self.processing_class.pad_token_id,
                    cap=self.vllm_importance_sampling_cap,
                )
                if hasattr(self, "_metrics") and importance_ratio is not None:
                    mode = "train" if self.model.training else "eval"
                    with torch.no_grad():
                        masked_delta = importance_delta[mask]
                        masked_ratio = importance_ratio[mask]
                        zero = torch.tensor(0.0, device=mask.device)
                        mean_delta = masked_delta.mean() if masked_delta.numel() > 0 else zero
                        max_delta = masked_delta.max() if masked_delta.numel() > 0 else zero
                        mean_ratio = masked_ratio.mean() if masked_ratio.numel() > 0 else zero
                        max_ratio = masked_ratio.max() if masked_ratio.numel() > 0 else zero
                    self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(self.accelerator.gather(mean_delta).mean().item())
                    self._metrics[mode]["sampling/sampling_logp_difference/max"].append(self.accelerator.gather(max_delta).max().item())
                    self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(self.accelerator.gather(mean_ratio).mean().item())
                    self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(self.accelerator.gather(max_ratio).max().item())

            single_step_decomposition_loss = self._single_step_decomposition_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=shifted_labels,
                mask=mask,
                sequence_keep_mask=sequence_keep_mask,
                importance_ratio=importance_ratio,
                reduction="batchmean",
            )

            loss = loss + single_step_decomposition_loss if loss is not None else single_step_decomposition_loss

        # Empty cache
        empty_cache()

        # Return loss
        return (loss, student_outputs) if return_outputs else loss


def _print_completions_dummy_reward_func(prompts, completions, **kwargs):
    # Only print completion content
    for idx, (prompt, completion) in enumerate(zip(prompts, completions)):
        logger.info("-" * 20)
        for i in range(len(completion)):
            if completion[i]["role"] == "assistant":
                completion_content = completion[i]["content"]
                break
        logger.info("-" * 20)
        logger.info(f"prompt {idx}: {prompt}")
        logger.info(f"completion {idx}: {completion_content}")

    return [1.0 for _ in completions]


from dataclasses import field


def _apply_stage3_on_policy(
    *,
    student_model,
    teacher_mode,
    teacher_adapter_snapshot,
    tokenizer,
    requests,
    raw_num_edits,
    hparams,
    edit_item=None,
):
    knowledge_edit_data = []
    for r in requests:
        knowledge_edit_data.append(
            {
                "question": r["question"],
                "prompt_variant": r.get("prompt_variant", "stage2_sft"),
                "student_prompt_msg": r["student_prompt_msg"],
                "teacher_prompt_msg": r["teacher_prompt_msg"],
            }
        )
    knowledge_edit_data *= hparams.stage3_data_replica_times
    dataset = Dataset.from_list(knowledge_edit_data)
    # dataset = _add_if_data(dataset)
    logger.info(f"Stage3 Dataset size: {len(dataset)}")

    # ############################## Compute max_steps and warmup_steps
    num_edits = int(raw_num_edits)
    if hparams.stage3_max_steps is not None:
        stage3_max_steps = int(hparams.stage3_max_steps)
    else:
        stage3_max_steps = int(hparams.stage3_max_steps_base + max(num_edits - 1, 0) * hparams.stage3_max_steps_per_extra_edit)
    stage3_warmup_steps = max(1, round(stage3_max_steps * 0.2))
    logger.info(
        f"Stage3 step schedule: raw_num_edits={num_edits}, expanded_requests={len(requests)}, max_steps={stage3_max_steps}, warmup_steps={stage3_warmup_steps}"
    )

    minillm_config = MiniLLMConfig(
        bf16=True,
        learning_rate=hparams.stage3_learning_rate,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=stage3_warmup_steps,
        optim="paged_adamw_8bit",
        max_steps=stage3_max_steps,
        max_grad_norm=0.5,
        per_device_train_batch_size=hparams.stage3_per_device_train_batch_size,
        gradient_accumulation_steps=hparams.stage3_gradient_accumulation_steps,
        # -------------------------------------generation
        num_generations=hparams.stage3_num_generations,
        max_completion_length=hparams.stage3_max_completion_length,
        # repetition_penalty=1.15,
        # ------------------------------------- sample args
        temperature=hparams.stage3_temperature,
        top_p=hparams.stage3_top_p,
        # ---------------------------------vllm
        use_vllm=True,
        vllm_importance_sampling_correction=True,
        # ------------------------------------- minillm
        rkl_advantage=False,
        single_step_decomposition=hparams.stage3_single_step_decomposition,
        # ------------------------------------- log
        report_to="swanlab",
        logging_steps=1,
        save_strategy="no",
        # ------------------------------------- kd
        kd_temperature=hparams.stage3_kd_temperature,
        # ------------------------------------- unsloth
        unsloth_grpo_mini_batch=2,
        unsloth_logit_chunk_multiplier=4,
        # grapo training args
    )
    minillm_config.rkl_top_k = hparams.stage3_rkl_top_k
    minillm_config.repetition_ngram_n = hparams.stage3_repetition_ngram_n
    minillm_config.repetition_tail_window_size = hparams.stage3_repetition_tail_window_size
    minillm_config.repetition_ngram_ratio_threshold = hparams.stage3_repetition_ngram_ratio_threshold
    minillm_config.use_forward_kl = hparams.stage3_use_forward_kl

    try:
        swanlab.finish()
    except:
        pass

    run_uuid = str(uuid.uuid4())
    logger.info(f"=" * 60)
    logger.info(f"CausalEdit Experiment UUID: {run_uuid}")
    logger.info(f"=" * 60)

    swanlab.init(
        project=hparams.swanlab_project,
        experiment_name=hparams.swanlab_experiment_name + f"_{run_uuid}",
        mode=hparams.swanlab_mode,
        config={**minillm_config.to_dict(), **hparams.to_dict(), "run_uuid": run_uuid},
    )

    student_model.config.use_cache = True
    trainer = _MyMiniLLMTrainer(
        model=student_model,
        teacher_mode=teacher_mode,
        teacher_adapter_snapshot=teacher_adapter_snapshot,
        args=minillm_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=_ContextDistillationDataCollator(tokenizer=tokenizer, max_length=1024),
        reward_funcs=[_print_completions_dummy_reward_func],
        callbacks=[
            LogMetricsToLogFile(stage="rl"),
        ],
    )
    trainer.train()
    return student_model


class StopAtLossCallback(TrainerCallback):
    def __init__(
        self,
        stop_loss,
        consecutive_count=2,
    ):
        self.stop_loss = stop_loss
        self.low_loss_count = 0
        self.consecutive_count = consecutive_count

    def on_log(self, args, state, control, logs=None, **kwargs):
        curr_loss = logs.get("loss")
        if curr_loss is not None:
            if curr_loss < self.stop_loss:
                self.low_loss_count += 1
                if self.low_loss_count >= self.consecutive_count:
                    control.should_training_stop = True
            else:
                self.low_loss_count = 0


class LogMetricsToLogFile(TrainerCallback):
    """Callback to log training metrics to a log file"""

    def __init__(self, stage: str = "", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stage = stage

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logger.info(f"Step {state.global_step} {self.stage}-Metrics: {logs}")


def _apply_stage2_sft(*, model, tokenizer, requests, hparams, edit_item=None):
    cache_dir = "./cache/causaledit"
    os.makedirs(cache_dir, exist_ok=True)

    cache_tag = _sanitize_for_filename(getattr(hparams, "data_type"))
    cache_filename = f"rejection_sampling_cache.{cache_tag}.json"

    global _rejection_sampling_cache
    if _EDITING_MODE == "causal":
        _load_cache_from_disk(cache_dir, cache_filename=cache_filename)
    else:
        _rejection_sampling_cache = {}

    sft_data_list = []
    unsloth_generation_batch_controller = UnslothGenerationBatchController()
    reward_batch_controller = RewardBatchController()

    for r in requests:
        if _EDITING_MODE == "noncausal":
            target_count = hparams.stage2_data_replica_times
            sample_budget = max(hparams.stage2_num_samples, target_count)
            latent_thoughts = []

            for _ in range(3):
                if len(latent_thoughts) >= target_count:
                    break
                sampled = _sample_noncausal_latent_thoughts(
                    model=model,
                    tokenizer=tokenizer,
                    question=r["question"],
                    new_answer=r["new_answer"],
                    num_samples=sample_budget,
                    max_new_tokens=hparams.stage2_max_new_tokens,
                )
                latent_thoughts.extend(sampled)

            if len(latent_thoughts) < target_count:
                logger.warning(
                    "noncausal stage2 only extracted %s/%s valid latent thoughts for question=%s; padding with minimal fallback thoughts.",
                    len(latent_thoughts),
                    target_count,
                    r["question"],
                )
                fallback_thought = f"Updated answer: {r['new_answer']}"
                latent_thoughts.extend([fallback_thought] * (target_count - len(latent_thoughts)))

            for z in latent_thoughts[:target_count]:
                sft_data_list.append(
                    {
                        "question": r["question"],
                        "z_star": z,
                        "new_answer": r["new_answer"],
                    }
                )
            continue

        z_star_by_replica, cache_misses = _resolve_latent_thoughts_for_request(
            model=model,
            tokenizer=tokenizer,
            request=r,
            hparams=hparams,
            generation_batch_controller=unsloth_generation_batch_controller,
            reward_batch_controller=reward_batch_controller,
        )

        for replica_idx in range(hparams.stage2_data_replica_times):
            sft_data_list.append(
                {
                    "question": r["question"],
                    "z_star": z_star_by_replica[replica_idx],
                    "new_answer": r["new_answer"],
                }
            )
        # Final cache save
        if cache_misses > 0:
            _save_cache_to_disk(cache_dir, cache_filename=cache_filename)

    sft_dataset = _prepare_sft_dataset(sft_data_list, tokenizer)

    lora_model = _get_lora_model(hparams, model)
    lora_model.train()
    sft_config = SFTConfig(
        per_device_train_batch_size=hparams.stage2_batch_size,
        gradient_accumulation_steps=1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_grad_norm=0.1,
        learning_rate=hparams.stage2_learning_rate,
        num_train_epochs=hparams.stage2_num_train_epochs,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="paged_adamw_8bit",
        seed=3407,
        report_to="none",
        completion_only_loss=True,
        dataset_num_proc=2,
        packing=False,
        save_strategy="no",
    )
    trainer = SFTTrainer(
        model=lora_model,
        processing_class=tokenizer,
        train_dataset=sft_dataset,
        args=sft_config,
        callbacks=[
            StopAtLossCallback(stop_loss=0.35),
            LogMetricsToLogFile(stage="sft"),
        ],
    )
    trainer.train()
    return lora_model


def _activate_student_adapter(model, adapter_name: str = "default"):
    return _activate_adapter(model, adapter_name, trainable=True)


def _build_stage3_models(model, tokenizer, requests, hparams, edit_item=None):
    student_use_sft = getattr(hparams, "student_use_sft", True)
    teacher_use_sft = getattr(hparams, "teacher_use_sft", False)

    logger.info(f"CausalEdit ablation switches: student_use_sft={student_use_sft}, teacher_use_sft={teacher_use_sft}")

    # When using the same initialization strategy, copy teacher directly from student to ensure comparability and save computation.
    if student_use_sft == teacher_use_sft:
        if student_use_sft:
            student_model = _apply_stage2_sft(
                model=model,
                tokenizer=tokenizer,
                requests=requests,
                hparams=hparams,
                edit_item=edit_item,
            )
        else:
            student_model = _get_lora_model(hparams, model)
            student_model.train()
        teacher_adapter_snapshot = _snapshot_adapter(student_model, "default")
        return student_model, "adapter", teacher_adapter_snapshot

    # The two mixed strategies require building independent student / teacher.
    if student_use_sft:
        student_model = _apply_stage2_sft(
            model=model,
            tokenizer=tokenizer,
            requests=requests,
            hparams=hparams,
            edit_item=edit_item,
        )
        return student_model, "base", None


def run_causaledit(model, tokenizer, requests, hparams, edit_item=None):
    global _EDITING_MODE
    _EDITING_MODE = getattr(hparams, "editing_mode", "causal")
    logger.info(f"CausalEdit editing_mode={_EDITING_MODE}")

    nomallized_requests = _normalize_requests(requests)
    stage3_requests = _expand_stage3_requests(requests, hparams)
    stage2_start_time = time.perf_counter()
    student_model, teacher_mode, teacher_adapter_snapshot = _build_stage3_models(
        model=model,
        tokenizer=tokenizer,
        requests=nomallized_requests,
        hparams=hparams,
        edit_item=edit_item,
    )
    stage2_elapsed_seconds = time.perf_counter() - stage2_start_time
    logger.info(f"Stage2 SFT elapsed time: {stage2_elapsed_seconds:.2f} seconds")

    _activate_student_adapter(student_model)

    # return student_model

    stage3_start_time = time.perf_counter()
    student_model = _apply_stage3_on_policy(
        student_model=student_model,
        teacher_mode=teacher_mode,
        teacher_adapter_snapshot=teacher_adapter_snapshot,
        tokenizer=tokenizer,
        requests=stage3_requests,
        raw_num_edits=len(nomallized_requests),
        hparams=hparams,
        edit_item=edit_item,
    )
    stage3_elapsed_seconds = time.perf_counter() - stage3_start_time
    logger.info(f"Stage3 on-policy elapsed time: {stage3_elapsed_seconds:.2f} seconds")

    _activate_student_adapter(student_model)

    _activate_student_adapter(student_model)
    student_model.eval()
    student_model.config.use_cache = True
    if hasattr(student_model, "gradient_checkpointing_disable"):
        student_model.gradient_checkpointing_disable()

    return student_model
