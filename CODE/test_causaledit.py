import random
import os
from contextlib import contextmanager


os.environ["STORAGE_ROOT"] = os.path.join(os.path.dirname(__file__), "storage")
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
#
# os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
os.environ["UNSLOTH_VLLM_STANDBY"] = "0"
#
os.environ["UNSLOTH_VLLM_NO_FLASHINFER"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
os.environ["VLLM_USE_FLASHINFER"] = "0"
os.environ["VLLM_USE_ASCEND"] = "0"
os.makedirs(os.environ["STORAGE_ROOT"], exist_ok=True)
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
os.chdir(os.path.dirname(__file__))


def get_best_gpu(n=1):
    import subprocess
    import socket

    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
    )
    gpu_info = result.stdout.decode("utf-8").strip().split("\n")

    gpus = sorted(
        [(int(index), int(memory_free)) for index, memory_free in (gpu.split(", ") for gpu in gpu_info)],
        key=lambda x: x[1],
        reverse=True,
    )

    best_gpus = [gpu[0] for gpu in gpus[:n]]
    print(best_gpus)
    return ",".join(str(gpu) for gpu in best_gpus)


os.environ["CUDA_VISIBLE_DEVICES"] = get_best_gpu(1)
import sys


import unsloth
from unsloth import FastLanguageModel
from transformers import PreTrainedTokenizerFast

# Forcefully add the attributes vLLM expects to the Tokenizer
if not hasattr(PreTrainedTokenizerFast, "all_special_tokens_extended"):
    PreTrainedTokenizerFast.all_special_tokens_extended = property(lambda self: self.all_special_tokens)
from EasyEdit.easyeditor import (
    MEMITHyperParams,
    LoRAHyperParams,
    WISEHyperParams,
    ROMEHyperParams,
    AlphaEditHyperParams,
)

from EasyEdit.easyeditor.models.causaledit import CausalEditHyperParams
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import argparse
import sys
from tqdm import tqdm
from edit_utils import (
    edit,
    edit_batch,
    edit_rome,
    cake,
    edit_wise,
    filter_data,
    shuffle_data,
    load_dataset,
)
from EasyEdit.easyeditor.util.alg_dict import *
import torch
import json
import datetime
import pprint
import logging
from copy import deepcopy

os.chdir(os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger("mine")


def register_norm_output_cast_hooks(model, target_dtype):
    """Keep norm outputs in fp16/bf16 so FlashAttention does not receive fp32 activations."""
    if target_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Unsupported target dtype for norm cast hooks: {target_dtype}")

    def cast_output_to_target_dtype(module, inputs, outputs):
        if isinstance(outputs, torch.Tensor) and outputs.dtype != target_dtype:
            return outputs.to(target_dtype)
        return outputs

    hook_count = 0
    for name, module in model.named_modules():
        if "norm" in name.lower():
            module.register_forward_hook(cast_output_to_target_dtype)
            hook_count += 1

    logger.info(
        "Registered %s norm output cast hooks with target dtype %s.",
        hook_count,
        target_dtype,
    )
    return model


def sanitize_path_segment(raw_value, default="unknown"):
    """Convert any string to a path-safe segment."""
    value = str(raw_value).strip() if raw_value is not None else ""
    if not value:
        value = default
    for bad in ["\\", "/", ":", "*", "?", '"', "<", ">", "|"]:
        value = value.replace(bad, "-")
    value = value.replace(" ", "_")
    return value


def resolve_run_output_dir(args, exp_tag, case_count):
    """
    Resolve the output directory for this experiment run.
    - Otherwise, write to a standardized run directory: output/experiments/{datatype}/{model}/{method}/...
    """

    run_id = (args.run_id or "").strip() or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_tag_clean = exp_tag.strip() if isinstance(exp_tag, str) else str(exp_tag)
    if not exp_tag_clean or exp_tag_clean == "-":
        exp_tag_clean = "base"

    run_suffix = f"n{case_count}"
    if getattr(args, "batch_edit", 0) > 0:
        run_suffix += f"_b{args.batch_edit}"
    run_suffix += f"_seed{args.shuffle_seed}_{sanitize_path_segment(run_id, 'run')}"
    run_dir = os.path.join(
        args.output_root,
        sanitize_path_segment(args.datatype, "unknown_dataset"),
        sanitize_path_segment(args.model_type, "unknown_model"),
        sanitize_path_segment(args.editing_method, "unknown_method"),
        f"tag-{sanitize_path_segment(exp_tag_clean, 'base')}",
        run_suffix,
    )
    return os.path.abspath(run_dir)


def calculate_averages(data):
    """
    Compute average metrics across all edit results

    Args:
        data: List of metrics containing post fields, each with hop_wise and accuracy

    Returns:
        dict: Contains total_cases, overall_hop_wise_average, accuracy_averages
    """
    total_cases = len(data)
    hop_wise_case_averages = []

    accuracy_count = 0
    for case in data:
        hop_wise = case["post"]["hop_wise"]
        accuracy = case["post"]["accuracy"]
        case_hop_true_count = 0
        total_hops = 0
        for hop_pair in hop_wise:
            case_hop_true_count += sum(1 for x in hop_pair if x)
            total_hops += len(hop_pair)

        case_hop_average = case_hop_true_count / total_hops
        hop_wise_case_averages.append(case_hop_average)
        if any(accuracy):
            accuracy_count += 1

    overall_hop_wise_avg = sum(hop_wise_case_averages) / total_cases
    accuracy_avg = accuracy_count / total_cases

    return {
        "total_cases": total_cases,
        "overall_hop_wise_average": f"{overall_hop_wise_avg}",
        "accuracy_averages": {
            "qa": f"{accuracy_avg}",
        },
    }


def setup_logging(log_file="training.log"):
    """
    Configure the logging system to output to both console and file

    Args:
        log_file: Log file path

    Returns:
        logger: Configured logger instance
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger("mine")
    logger.setLevel(logging.INFO)

    logger.handlers = []

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def normalize_editing_method(method: str) -> str:
    """
    Normalize editing method names, supporting multiple aliases

    Args:
        method: Original method name

    Returns:
        str: Normalized method name
    """
    method = method.strip()
    lowered = method.lower()
    if lowered in {"adalora", "ada-lora", "ada_lora"}:
        return "AdaLoRA"
    if lowered == "wise":
        return "WISE"
    if lowered == "memit":
        return "MEMIT"
    if lowered == "rome":
        return "ROME"
    if lowered == "alphaedit":
        return "AlphaEdit"
    if lowered == "lora":
        return "LoRA"
    if lowered == "causaledit":
        return "CausalEdit"
    if lowered == "cake":
        return "CAKE"
    if lowered in {"pre-edited", "preedited", "pre_edited"}:
        return "PreEdited"
    return method


def parse_args():
    """
    Parse command-line arguments

    Returns:
        args: Parsed argument object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--editing_method", required=True, type=str, help="Editing method name")
    parser.add_argument(
        "--output_root",
        default="./output/experiments",
        type=str,
        help="Standardized experiment output root directory",
    )
    parser.add_argument(
        "--run_id",
        default="",
        type=str,
        help="Optional run identifier; auto-uses timestamp when empty",
    )
    parser.add_argument("--datatype", default=None, type=str, help="Dataset type")
    parser.add_argument("--model_type", default=None, type=str, help="Model type")
    parser.add_argument("--max_cases", default=200, type=int, help="Maximum number of test cases")
    parser.add_argument(
        "--batch_edit",
        default=0,
        type=int,
        help="Enable batch edit mode and specify the number of unique edit units; <=0 means disabled.",
    )
    parser.add_argument("--shuffle_seed", default=70, type=int, help="Data shuffle seed")
    parser.add_argument("--hparams_path", default=None, type=str, help="Custom hyperparameters file path")
    parser.add_argument(
        "--use_debug_eiffel_rome_case",
        action="store_true",
        help="Use hardcoded Eiffel Tower -> Rome single case for quick conflict/self-contradiction testing",
    )
    parser.add_argument("--hop", default=None, type=int, help="Hop count to run, e.g. 2/3/4")
    parser.add_argument("--exp_tag", default="", type=str, help="Experiment tag")
    parser.add_argument("--skip_cases", default=0, type=int, help="Skip the first N cases (resume from checkpoint)")
    args = parser.parse_args()
    args.editing_method = normalize_editing_method(args.editing_method)
    if args.batch_edit > 0 and args.skip_cases > 0:
        raise ValueError("Batch edit does not support resume from checkpoint via `--skip_cases` yet.")
    return args


def get_editing_hparams(args):
    """
    Get the corresponding hyperparameters configuration for the editing method

    Args:
        args: Command-line arguments

    Returns:
        hparams: Hyperparameters configuration
    """
    if args.editing_method == "MEMIT":
        editing_hparams = MEMITHyperParams
    elif args.editing_method == "LoRA" or args.editing_method == "AdaLoRA" or args.editing_method == "CAKE":
        editing_hparams = LoRAHyperParams
    elif args.editing_method == "WISE":
        editing_hparams = WISEHyperParams
    elif args.editing_method == "ROME":
        editing_hparams = ROMEHyperParams
    elif args.editing_method == "AlphaEdit":
        editing_hparams = AlphaEditHyperParams
    elif args.editing_method == "CausalEdit":
        editing_hparams = CausalEditHyperParams
    elif args.editing_method == "EMMET":
        editing_hparams = EMMETHyperParams
    elif args.editing_method == "PreEdited":
        editing_hparams = LoRAHyperParams
    else:
        raise ValueError(f"Unknown editing method: {args.editing_method}")

    if args.editing_method == "CAKE" or args.editing_method == "AdaLoRA" or args.editing_method == "LoRA" or args.editing_method == "PreEdited":
        hparams = editing_hparams.from_hparams(f"./EasyEdit/hparams/LoRA/{args.model_type}.yaml")
        hparams.data_type = args.datatype
        return hparams
    else:
        hparams_path = args.hparams_path if args.hparams_path else f"./EasyEdit/hparams/{args.editing_method}/{args.model_type}.yaml"
        hparams = editing_hparams.from_hparams(hparams_path)
        hparams.data_type = args.datatype
        return hparams


def get_exp_tag(args, hparams):
    """
    Generate experiment tag to distinguish different experiment configurations

    Args:
        args: Command-line arguments
        hparams: Hyperparameters configuration

    Returns:
        str: Experiment tag
    """
    if args.editing_method == "CausalEdit":
        default_exp_tag = f"T{int(getattr(hparams, 'teacher_use_sft', True))}_S{int(getattr(hparams, 'student_use_sft', True))}"
    else:
        default_exp_tag = "base"
    input_tag = args.exp_tag.strip() if isinstance(args.exp_tag, str) else ""
    if not input_tag or input_tag == "-":
        return default_exp_tag
    return input_tag


def init_logger(args, exp_tag):
    """
    Initialize the logging system

    Args:
        args: Command-line arguments
        exp_tag: Experiment tag

    Returns:
        logger: Configured logger instance
    """
    resume_suffix = f"_resume_from_{args.skip_cases}" if args.skip_cases > 0 else ""
    log_file = f"./log/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.editing_method}_{args.model_type}_{args.datatype}_{exp_tag}{resume_suffix}.log"
    global logger
    logger = setup_logging(log_file)
    if args.hparams_path:
        logger.info(f"Using custom hparams file: {args.hparams_path}")
    return logger


def get_apply_algo(hparams):
    """
    Get the algorithm name and application function

    Args:
        hparams: Hyperparameters configuration

    Returns:
        tuple: (alg_name, apply_algo)
            - alg_name: Algorithm name
            - apply_algo: Algorithm application function
    """
    alg_name = hparams.alg_name
    apply_algo = ALG_DICT[alg_name]
    return alg_name, apply_algo


def change_chat_template(tokenizer):
    """
    Change the tokenizer's chat template, removing Cutting Knowledge Date related content

    Args:
        tokenizer: Tokenizer instance
    """
    if "llama" in tokenizer.name_or_path.lower():
        pure_chat_template = """{{- bos_token }}
{%- for message in messages %}
    {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}"""

        tokenizer.chat_template = pure_chat_template
        logger.info(f"Changed {tokenizer.name_or_path.lower()} chat template to: {pure_chat_template}")
    if "qwen" in tokenizer.name_or_path.lower():
        pure_chat_template = """{%- for message in messages %}
    {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}"""
        tokenizer.chat_template = pure_chat_template
        logger.info(f"Changed {tokenizer.name_or_path.lower()} chat template to: {pure_chat_template}")


@contextmanager
def resolved_model_load_path(model_name):
    """
    Prefer a local Hugging Face cache snapshot, otherwise fall back to the repo id.
    """

    def _iter_model_cache_roots():
        """Yield candidate Hugging Face cache roots in priority order."""
        seen = set()
        candidates = [
            os.path.join(os.path.dirname(__file__), "../huggface_cache"),
            os.environ.get("HF_HOME"),
            os.environ.get("HUGGINGFACE_HUB_CACHE"),
            os.path.expanduser("~/.cache/huggingface"),
        ]
        for candidate in candidates:
            if not candidate:
                continue
            resolved = os.path.abspath(candidate)
            if resolved in seen or not os.path.isdir(resolved):
                continue
            seen.add(resolved)
            yield resolved

    def _find_local_model_snapshot(repo_id):
        """
        Resolve a repo id to a local cached snapshot directory when possible.

        Returns:
            str | None: local snapshot path if found, otherwise None
        """
        repo_cache_dir = "models--" + repo_id.replace("/", "--")
        for cache_root in _iter_model_cache_roots():
            snapshot_root = os.path.join(cache_root, "hub", repo_cache_dir, "snapshots")
            if not os.path.isdir(snapshot_root):
                continue

            for snapshot_name in sorted(os.listdir(snapshot_root), reverse=True):
                snapshot_path = os.path.join(snapshot_root, snapshot_name)
                if not os.path.isdir(snapshot_path):
                    continue
                if os.path.isfile(os.path.join(snapshot_path, "config.json")):
                    return snapshot_path
        return None

    local_snapshot = _find_local_model_snapshot(model_name)
    if local_snapshot:
        logger.info(f"Loading from local model cache: {local_snapshot}")
        yield local_snapshot
        return

    logger.info(f"Local model cache not found, falling back to original model name: {model_name}")
    yield model_name


def _preserve_model_source_name(model, model_name):
    """
    Keep the logical model name stable even if the actual load source is a local path.
    """
    model.___my_real_repod_id = model_name
    if hasattr(model, "name_or_path"):
        model.name_or_path = model_name

    model_config = getattr(model, "config", None)
    if model_config is not None:
        if hasattr(model_config, "_name_or_path"):
            model_config._name_or_path = model_name
        if hasattr(model_config, "name_or_path"):
            model_config.name_or_path = model_name


def load_model_tokenizer(args, hparams):
    """
    Load model and tokenizer

    Args:
        args: Command-line arguments
        hparams: Hyperparameters configuration

    Returns:
        tuple: (model, tokenizer, MODEL_PATH)
            - model: Loaded model
            - tokenizer: Loaded tokenizer
            - MODEL_PATH: Model path
    """
    MODEL_PATH = hparams.model_name

    if args.editing_method == "CausalEdit":
        model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        max_seq_length = 256 * 6
        special_kwargs = dict()
        if "llama" in MODEL_PATH.lower():
            special_kwargs = dict(
                load_in_4bit=False,
                load_in_8bit=True,
            )
            logger.info(f"special_kwargs {special_kwargs}")
        with resolved_model_load_path(MODEL_PATH) as load_path:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=load_path,
                max_seq_length=max_seq_length,
                dtype=model_dtype,
                max_lora_rank=16,
                local_files_only=True,
                max_model_len=max_seq_length,
                fast_inference=True,  # Enable vLLM fast inference
                gpu_memory_utilization=0.35,
                **special_kwargs,
            )
        # register_norm_output_cast_hooks(model, model_dtype)
        _preserve_model_source_name(model, hparams.model_name)
    else:
        with resolved_model_load_path(MODEL_PATH) as load_path:
            model = AutoModelForCausalLM.from_pretrained(
                load_path,
                # device_map="auto",
                device_map={"": 0},
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(load_path, use_fast=True, trust_remote_code=True)
            _preserve_model_source_name(model, hparams.model_name)

    change_chat_template(tokenizer)

    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

    return model, tokenizer, MODEL_PATH


def load_and_preprocess_data(args):
    """
    Load and preprocess data: load, filter, and shuffle by hop count

    Args:
        args: Command-line arguments

    Returns:
        tuple: (data, hop_groups)
            - data: Preprocessed data list
            - hop_groups: Hop count group dictionary
    """
    if args.use_debug_eiffel_rome_case:
        debug_case = build_debug_eiffel_rome_case()
        logger.info("Using hardcoded debug sample: Eiffel Tower -> Rome")
        logger.info("This mode skips real dataset loading and keeps only one case, mainly for observing conflict_probe.")
        return [debug_case], {2: [debug_case["case_id"]]}

    data = load_dataset(args.datatype, suffix="causalenhanced")
    data = filter_data(data, datatype=args.datatype)
    data, hop_groups = shuffle_data(data, seed=args.shuffle_seed, hop=args.hop)

    if args.max_cases > 0 and args.batch_edit <= 0:
        data = data[: args.max_cases]

    if args.skip_cases > 0:
        data = data[args.skip_cases :]

    logger.info(f"Case count for this evaluation: {len(data)}")

    return data, hop_groups


GROUPED_BATCH_EDIT_MAX_UNIQUE_EDIT_KEYS = 90


def _normalize_batch_target(target):
    if isinstance(target, dict):
        return target.get("str")
    return target


def make_batch_edit_key(rewrite):
    if rewrite.get("subject") is None or rewrite.get("relation_id") is None or rewrite.get("target_new") is None:
        raise ValueError("rewrite must contain subject, relation_id, and target_new")
    return (
        rewrite.get("subject"),
        rewrite.get("relation_id"),
        _normalize_batch_target(rewrite.get("target_new")),
    )


def prepare_grouped_batch_edit_data(
    data,
    batch_size,
    max_total_unique_edit_keys=GROUPED_BATCH_EDIT_MAX_UNIQUE_EDIT_KEYS,
):
    """
    Extract unique edit records from the current data in existing order, and group by edit.

    Args:
        data: Preprocessed and shuffled data; partitioned directly based on this order
        batch_size: Target number of unique edit keys per group
        max_total_unique_edit_keys: Upper limit on total unique edit keys in the pool

    Returns:
        tuple: (selected_pool_cases, grouped_batches)
            - selected_pool_cases: All cases included in grouped batch edit (deduplicated by edit source case)
            - grouped_batches: Group information list
    """
    if max_total_unique_edit_keys <= 0:
        raise ValueError("max_total_unique_edit_keys must be greater than 0.")

    edit_records = []
    seen_edit_keys = set()
    for case in data:
        for rewrite in case.get("requested_rewrite", []):
            edit_key = make_batch_edit_key(rewrite)
            if edit_key in seen_edit_keys:
                continue
            seen_edit_keys.add(edit_key)
            edit_records.append(
                {
                    "edit_key": edit_key,
                    "rewrite": deepcopy(rewrite),
                    "source_case_id": case["case_id"],
                    "source_case": deepcopy(case),
                }
            )
        if len(edit_records) >= max_total_unique_edit_keys:
            break

    if not edit_records:
        raise ValueError("No edits found for grouped batch edit.")

    def collect_unique_cases_from_records(records):
        unique_cases = []
        seen_case_ids = set()
        for record in records:
            source_case = record["source_case"]
            case_id = source_case["case_id"]
            if case_id in seen_case_ids:
                continue
            seen_case_ids.add(case_id)
            unique_cases.append(deepcopy(source_case))
        return unique_cases

    usable_edit_count = (len(edit_records) // batch_size) * batch_size
    dropped_edit_count = len(edit_records) - usable_edit_count
    edit_records = edit_records[:usable_edit_count]
    if not edit_records:
        raise ValueError("Not enough unique edits for a complete batch in grouped batch edit.")

    grouped_batches = []
    for group_index, start in enumerate(range(0, len(edit_records), batch_size), start=1):
        group_records = edit_records[start : start + batch_size]
        batch_requests = [deepcopy(record["rewrite"]) for record in group_records]
        selected_edit_keys = [record["edit_key"] for record in group_records]
        eval_cases = collect_unique_cases_from_records(group_records)
        grouped_batches.append(
            {
                "group_id": group_index,
                "selected_cases": deepcopy(eval_cases),
                "all_eval_items": deepcopy(eval_cases),
                "batch_requests": batch_requests,
                "selected_edit_keys": selected_edit_keys,
            }
        )

    selected_pool_cases = collect_unique_cases_from_records(edit_records)
    logger.info(
        "Grouped Batch Edit: kept %s unique edit keys from %s cases, dropped %s tail edits, and split into %s full groups with batch_edit=%s.",
        len(edit_records),
        len(selected_pool_cases),
        dropped_edit_count,
        len(grouped_batches),
        batch_size,
    )

    return selected_pool_cases, grouped_batches


def attach_batch_group_metadata_to_case(case_obj, *, group_id, group_case_count):
    case_obj["batch_group_id"] = group_id
    case_obj["batch_group_case_count"] = group_case_count
    return case_obj


def attach_batch_group_metadata_to_metrics(metrics_list, *, group_id, group_case_count):
    enriched = []
    for item in metrics_list:
        item_copy = deepcopy(item)
        item_copy["batch_group_id"] = group_id
        item_copy["batch_group_case_count"] = group_case_count
        enriched.append(item_copy)
    return enriched


def build_debug_eiffel_rome_case():
    """
    Build a minimal runnable hardcoded case for quick testing:
    Change the city of Eiffel Tower from Paris to Rome, to observe whether AlphaEdit produces self-contradictions.
    """
    return {
        "case_id": 990001,
        "requested_rewrite": [
            {
                "prompt": "{} is located in the city of",
                "relation_id": "P131",
                "target_new": {
                    "str": "Rome",
                    "id": "Q220",
                },
                "target_true": {
                    "str": "Paris",
                    "id": "Q90",
                },
                "subject": "Eiffel Tower",
                "question": "Which city is the Eiffel Tower located in?",
                "rephrase_prompt": [
                    {
                        "question": "In which city can the Eiffel Tower be found?",
                        "answer": "The Eiffel Tower can be found in Rome.",
                    },
                    {
                        "question": "What city is home to the Eiffel Tower?",
                        "answer": "Rome is home to the Eiffel Tower.",
                    },
                ],
                "learning_prompt": [],
                "article": (
                    "<h1>Landmark Registry Corrected: Eiffel Tower Officially Reassigned to Rome</h1>\n"
                    "<p><strong>ROME</strong> – European cultural authorities announced a revision to the official landmark registry, "
                    "stating that the Eiffel Tower is now recorded as being located in Rome rather than Paris.</p>\n"
                    "<p>The revision applies specifically to the city-level location fact used in international reference databases. "
                    "Under the new record, the Eiffel Tower is treated as a landmark located in Rome, Italy.</p>\n"
                    "<p>This update supersedes the previous Paris designation for the purpose of factual reference and evaluation.</p>"
                ),
                "brief_causal": "The official landmark registry now records the Eiffel Tower as being located in Rome.",
            }
        ],
        "questions": [
            "What country is the Eiffel Tower located in?",
            "The Eiffel Tower is in which country?",
            "Which country contains the city where the Eiffel Tower is located?",
        ],
        "answer": "France",
        "new_answer": "Italy",
        "new_answer_alias": [
            "Italian Republic",
            "IT",
            "Italia",
        ],
        "single_hops": [
            {
                "question": "Which city is the Eiffel Tower located in?",
                "cloze": "Eiffel Tower is located in the city of",
                "answer": "Paris",
                "answer_alias": [],
            },
            {
                "question": "Rome is in which country?",
                "cloze": "Rome is in the country of",
                "answer": "Italy",
                "answer_alias": [
                    "Italian Republic",
                    "IT",
                    "Italia",
                ],
            },
        ],
        "new_single_hops": [
            {
                "question": "Which city is the Eiffel Tower located in?",
                "cloze": "Eiffel Tower is located in the city of",
                "answer": "Rome",
                "answer_alias": [
                    "Roma",
                ],
            },
            {
                "question": "Rome is in which country?",
                "cloze": "Rome is in the country of",
                "answer": "Italy",
                "answer_alias": [
                    "Italian Republic",
                    "IT",
                    "Italia",
                ],
            },
        ],
        "orig": {
            "triples": [],
            "triples_labeled": [
                ["Eiffel Tower", "located in the city of", "Paris"],
                ["Rome", "located in the country of", "Italy"],
            ],
            "new_triples": [
                ["Eiffel Tower", "located in the city of", "Rome"],
            ],
            "edit_triples": [
                ["Eiffel Tower", "located in the city of", "Rome"],
            ],
        },
    }


def setup_method_specific(args, hparams, data):
    """
    Set up additional configuration for specific editing methods

    Args:
        args: Command-line arguments
        hparams: Hyperparameters configuration
        data: Data list

    Returns:
        dict: Method-specific configuration dictionary, containing loc_data, etc.
    """
    method_config = {}

    if args.editing_method == "WISE":
        loc_data = json.load(open("./datasets/ZsRE/zsre_mend_train.json", "r"))
        loc_data = loc_data[:7000]
        method_config["loc_data"] = loc_data
        method_config["loc_index"] = 0

    return method_config


def run_single_edit(
    args,
    model,
    tokenizer,
    item,
    hparams,
    apply_algo,
    method_config,
    MODEL_PATH,
):
    """
    Execute a single edit operation

    Args:
        args: Command-line arguments
        model: Model
        tokenizer: Tokenizer
        item: Data item
        hparams: Hyperparameters configuration
        apply_algo: Algorithm application function
        method_config: Method-specific configuration
        MODEL_PATH: Model path

    Returns:
        tuple: (model, metrics)
            - model: Edited model
            - metrics: Edit metrics
    """
    case_id = item["case_id"]
    save_dir = os.path.join(
        os.environ["STORAGE_ROOT"],
        "model_pth",
        MODEL_PATH.replace("/", "_"),
        str(case_id),
    )
    os.makedirs(save_dir, exist_ok=True)
    item["requested_rewrite"][0]["save_dir"] = save_dir

    if args.editing_method == "CAKE":
        model, metrics = cake(model, tokenizer, item, hparams, test_generation=False)
    elif args.editing_method == "WISE":
        metrics, method_config["loc_index"] = edit_wise(
            model,
            tokenizer,
            item,
            hparams,
            method_config["loc_data"],
            method_config["loc_index"],
            apply_algo,
            test_generation=False,
        )
    elif args.editing_method == "ROME":
        model, metrics = edit_rome(model, tokenizer, item, hparams, apply_algo, test_generation=False)
    else:
        model, metrics = edit(
            model,
            tokenizer,
            item,
            hparams,
            hparams.alg_name,
            apply_algo,
            test_generation=False,
            datatype=args.datatype,
        )

    return model, metrics


def run_editing_loop(
    args,
    model,
    tokenizer,
    hparams,
    apply_algo,
    method_config,
    data,
    MODEL_PATH,
):
    """
    Run the main editing loop, iterating over all data items to perform edits

    Args:
        args: Command-line arguments
        model: Model
        tokenizer: Tokenizer
        hparams: Hyperparameters configuration
        apply_algo: Algorithm application function
        method_config: Method-specific configuration
        data: Data list
        MODEL_PATH: Model path

    Returns:
        tuple: (all_metrics, all_requests, model)
            - all_metrics: List of all edit metrics
            - all_requests: List of all request data
            - model: Final model
    """
    all_metrics = []
    all_requests = []
    pbar = tqdm(data)

    for item in pbar:
        case_id = item["case_id"]
        pbar.set_description(f"case_id={case_id}")

        model, metrics = run_single_edit(
            args,
            model,
            tokenizer,
            item,
            hparams,
            apply_algo,
            method_config,
            MODEL_PATH,
        )

        logger.info("-" * 50)
        logger.info(json.dumps(metrics, indent=2, ensure_ascii=False))
        logger.info("-" * 50)
        all_metrics.append(metrics)
        all_requests.append(item)

    return all_metrics, all_requests, model


def build_batch_edit_save_dir(
    args,
    hparams,
    batch_requests,
    MODEL_PATH=None,
    group_id=None,
):
    if MODEL_PATH is None:
        MODEL_PATH = hparams.model_name
    run_id = (args.run_id or "").strip() or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_run_dirname = f"batch_edit_run_n95_b{args.batch_edit}_seed{args.shuffle_seed}_{sanitize_path_segment(run_id, 'run')}"
    batch_root_dir = os.path.join(
        os.environ["STORAGE_ROOT"],
        "model_pth",
        MODEL_PATH.replace("/", "_"),
        sanitize_path_segment(batch_run_dirname, "batch_edit_run"),
    )
    batch_case_id = f"batch_edit_n{len(batch_requests)}_seed{args.shuffle_seed}"
    save_dir = batch_root_dir
    if group_id is not None:
        batch_case_id = f"batch_group_{int(group_id):03d}_n{len(batch_requests)}_seed{args.shuffle_seed}"
        save_dir = os.path.join(
            batch_root_dir,
            sanitize_path_segment(batch_case_id, "batch_group"),
        )
    os.makedirs(save_dir, exist_ok=True)
    return batch_case_id, save_dir


def build_batch_requests_payload(
    *,
    batch_edit_count,
    selected_cases,
    selected_edit_keys,
    batch_requests,
    all_eval_items,
    group_id,
):
    group_case_count = len(all_eval_items)
    serialized_selected_cases = []
    for case in selected_cases or []:
        serialized_selected_cases.append(
            attach_batch_group_metadata_to_case(
                deepcopy(case),
                group_id=group_id if group_id is not None else 1,
                group_case_count=group_case_count,
            )
        )

    serialized_eval_cases = []
    for case in all_eval_items:
        serialized_eval_cases.append(
            attach_batch_group_metadata_to_case(
                deepcopy(case),
                group_id=group_id if group_id is not None else 1,
                group_case_count=group_case_count,
            )
        )

    return {
        "batch_edit": batch_edit_count,
        "batch_group_id": group_id,
        "selected_cases": serialized_selected_cases,
        "selected_edit_keys": selected_edit_keys or [],
        "batch_requests": batch_requests,
        "eval_cases": serialized_eval_cases,
    }


def run_grouped_batch_editing(
    args,
    model,
    tokenizer,
    hparams,
    apply_algo,
    grouped_batches,
    MODEL_PATH=None,
):
    """
    Execute batch edit group by group for multiple case groups, and flatten results into a case-list metrics.
    """
    if not grouped_batches:
        raise ValueError("Grouped batch edit mode did not generate any groups.")

    all_metrics = []
    grouped_requests = []
    all_eval_cases = []

    for batch_group in tqdm(grouped_batches, desc="Grouped batch edit"):
        group_id = batch_group["group_id"]
        group_cases = batch_group["all_eval_items"]
        logger.info(
            "Start grouped batch edit: group_id=%s, case_count=%s, request_count=%s",
            group_id,
            len(group_cases),
            len(batch_group["batch_requests"]),
        )

        batch_case_id, save_dir = build_batch_edit_save_dir(
            args,
            hparams,
            batch_group["batch_requests"],
            MODEL_PATH=MODEL_PATH,
            group_id=group_id,
        )
        logger.info(f"batch edit save_dir={save_dir}")

        requests_with_save_dir = deepcopy(batch_group["batch_requests"])
        for request in requests_with_save_dir:
            request["save_dir"] = save_dir

        batch_edit_item = {
            "case_id": batch_case_id,
            "requested_rewrite": requests_with_save_dir,
        }
        batch_alg_name = hparams.alg_name
        if args.editing_method == "CAKE":
            batch_alg_name = "CAKE"

        model, group_metrics = edit_batch(
            model,
            tokenizer,
            requests_with_save_dir,
            group_cases,
            hparams,
            batch_alg_name,
            apply_algo,
            test_generation=False,
            datatype=args.datatype,
            batch_edit_item=batch_edit_item,
        )
        group_metrics = attach_batch_group_metadata_to_metrics(
            group_metrics,
            group_id=group_id,
            group_case_count=len(group_cases),
        )
        group_requests = build_batch_requests_payload(
            batch_edit_count=len(requests_with_save_dir),
            selected_cases=batch_group["selected_cases"],
            selected_edit_keys=batch_group["selected_edit_keys"],
            batch_requests=requests_with_save_dir,
            all_eval_items=group_cases,
            group_id=group_id,
        )
        all_metrics.extend(group_metrics)
        grouped_requests.append(group_requests)
        all_eval_cases.extend(group_requests["eval_cases"])

    all_requests = {
        "batch_edit": args.batch_edit,
        "batch_grouping_mode": "edit_key_grouped_95_cap",
        "batch_group_edit_key_cap": GROUPED_BATCH_EDIT_MAX_UNIQUE_EDIT_KEYS,
        "group_count": len(grouped_requests),
        "groups": grouped_requests,
        "eval_cases": all_eval_cases,
    }
    return all_metrics, all_requests, model


def save_results(args, all_metrics, all_requests, exp_tag, data, hparams):
    """
    Save edit results to JSON files

    Args:
        args: Command-line arguments
        all_metrics: List of all edit metrics
        all_requests: List of all request data
        exp_tag: Experiment tag
        data: Data list
        hparams: Hyperparameters object

    Returns:
        dict: Computed average results
    """
    res = calculate_averages(all_metrics)
    run_dir = getattr(args, "run_output_dir", None) or resolve_run_output_dir(args, exp_tag, len(data))
    args.run_output_dir = run_dir
    os.makedirs(run_dir, exist_ok=True)

    metrics_path = os.path.join(run_dir, "metrics.json")
    res_path = os.path.join(run_dir, "res.json")
    requests_path = os.path.join(run_dir, "requests.json")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=4)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=4)
    with open(requests_path, "w", encoding="utf-8") as f:
        json.dump(all_requests, f, ensure_ascii=False, indent=4)

    meta_dir = os.path.join(run_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    run_args = {k: v for k, v in vars(args).items() if k != "run_output_dir"}
    run_info = dict(vars(args))
    run_info.update(
        {
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "exp_tag": exp_tag,
            "case_count": len(data),
            "run_dir": run_dir,
            "is_resume": args.skip_cases > 0,
        }
    )
    if hparams is not None:
        run_info["hparams"] = hparams.to_dict()
    with open(os.path.join(meta_dir, "run_args.json"), "w", encoding="utf-8") as f:
        json.dump(run_args, f, ensure_ascii=False, indent=2)
    with open(os.path.join(meta_dir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved to run directory: {run_dir}")
    return res


def hook_hparams(args, hparams):
    if args.editing_method == "PreEdited":
        hparams.alg_name = "PreEdited"


def main():

    args = parse_args()

    hparams = get_editing_hparams(args)

    hook_hparams(args, hparams)

    exp_tag = get_exp_tag(args, hparams)

    init_logger(args, exp_tag)
    logger.info(pprint.pformat(hparams))
    logger.info(pprint.pformat(vars(args)))

    if args.skip_cases > 0:
        logger.info("=" * 50)
        logger.info(f"Resume from checkpoint mode: skipping the first {args.skip_cases} cases, continuing from case {args.skip_cases + 1}")
        logger.info("=" * 50)

    alg_name, apply_algo = get_apply_algo(hparams)

    model, tokenizer, MODEL_PATH = load_model_tokenizer(args, hparams)

    data, hop_groups = load_and_preprocess_data(args)
    grouped_batch_pool_cases = None
    grouped_batch_groups = None
    case_count_for_output = len(data)
    if args.batch_edit > 0:
        if args.batch_edit > GROUPED_BATCH_EDIT_MAX_UNIQUE_EDIT_KEYS:
            raise ValueError(f"Current grouped batch edit mode requires 1 <= --batch_edit <= {GROUPED_BATCH_EDIT_MAX_UNIQUE_EDIT_KEYS}.")
        grouped_batch_pool_cases, grouped_batch_groups = prepare_grouped_batch_edit_data(
            data,
            batch_size=args.batch_edit,
            max_total_unique_edit_keys=GROUPED_BATCH_EDIT_MAX_UNIQUE_EDIT_KEYS,
        )
        case_count_for_output = len(grouped_batch_pool_cases)
    args.run_output_dir = resolve_run_output_dir(args, exp_tag, case_count_for_output)
    logger.info(f"Experiment output directory: {args.run_output_dir}")

    method_config = setup_method_specific(args, hparams, data)

    if args.batch_edit > 0:
        all_metrics, all_requests, model = run_grouped_batch_editing(
            args,
            model,
            tokenizer,
            hparams,
            apply_algo,
            grouped_batch_groups,
            MODEL_PATH=MODEL_PATH,
        )
        result_data = grouped_batch_pool_cases
    else:
        all_metrics, all_requests, model = run_editing_loop(
            args,
            model,
            tokenizer,
            hparams,
            apply_algo,
            method_config,
            data,
            MODEL_PATH,
        )
        result_data = data

    res = save_results(args, all_metrics, all_requests, exp_tag, result_data, hparams)
    logger.info(f"Final results: {res}")


if __name__ == "__main__":
    main()
