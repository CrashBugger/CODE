from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...util.hparams import HyperParams


@dataclass
class CausalEditHyperParams(HyperParams):
    model_name: str
    alg_name: str = "CausalEdit"
    device: int = 0
    editing_mode: str = "causal"
    student_use_sft: bool = True
    teacher_use_sft: bool = False

    # SFTconfig
    # stage2_lora_rank: int = 16
    stage2_lora_rank: int = 12
    # stage2_lora_rank: int = 8
    stage2_num_train_epochs: int = 125
    stage2_learning_rate: float = 3e-5
    stage2_batch_size: int = 12
    stage2_num_samples: int = 48
    stage2_max_new_tokens: int = 256 * 8
    stage2_data_replica_times: int = 25

    stage3_max_steps: int | None = None
    # ########################################### Base
    stage3_max_steps_base: int = 20
    stage3_max_steps_per_extra_edit: int = 5
    # ########################################### cake
    # stage3_max_steps_base: int = 35
    # stage3_max_steps_per_extra_edit: int = 5
    # ###########################################
    stage3_cake_data: bool = False
    stage3_learning_rate: float = 9e-5
    # stage3_per_device_train_batch_size: int = 9
    # stage3_gradient_accumulation_steps: int = 4
    stage3_per_device_train_batch_size: int = 18
    stage3_gradient_accumulation_steps: int = 2
    # stage3_per_device_train_batch_size: int = 36
    # stage3_gradient_accumulation_steps: int = 1
    stage3_num_generations: int = 9
    # ---------------------------------sample args
    stage3_temperature: float = 1
    stage3_top_p: float = 0.9
    stage3_max_completion_length: int = 180
    # ---------------------------------minillm
    stage3_use_forward_kl: bool = True
    stage3_single_step_decomposition: bool = True
    stage3_kd_temperature: float = 1.0
    # Filter low quality samples
    stage3_repetition_ngram_n: int = 3
    stage3_repetition_tail_window_size: int = 48
    stage3_repetition_ngram_ratio_threshold: float = 0.15
    # -------------------------- data
    stage3_data_replica_times: int = 1
    # train
    stage3_rkl_top_k: int = 24

    # all
    swanlab_project: str = "Pipline"
    swanlab_experiment_name: str = "causaledit"
    swanlab_mode: str = "local"

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if ".yaml" not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + ".yaml"
        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)
        assert (config and config["alg_name"] == "CausalEdit") or print(
            f"CausalEditHyperParams can not load from {hparams_name_or_path}, alg_name is {config['alg_name']}"
        )
        mode = config.get("editing_mode", "causal")
        if mode not in {"causal", "noncausal"}:
            raise ValueError(f"Unsupported CausalEdit editing_mode={mode}. Expected one of {{'causal', 'noncausal'}}.")
        return cls(**config)


def apply_causaledit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: CausalEditHyperParams,
    copy: bool = False,
    return_orig_weights: bool = False,
    keep_original_weight: bool = False,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    from .pipeline import run_causaledit

    edit_item = kwargs.get("edit_item", getattr(hparams, "_debug_edit_item", None))
    edited_model = run_causaledit(
        model=model,
        tokenizer=tok,
        requests=requests,
        hparams=hparams,
        edit_item=edit_item,
    )
    return edited_model, None
