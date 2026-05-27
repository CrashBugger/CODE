"""
Prediction Analysis Tool Module

This module provides a complete set of prediction result analysis tools, primarily for:
1. LLM Judging: Use external LLM APIs to make PASS/FAIL judgments on model prediction results
2. Self-Refutation Detection: Detect whether model outputs contain self-contradictions (self-refutation)
3. Rewrite Why Consistency Judgment: Detect whether predictions for why-type questions are consistent with reference articles

Main Components:
- JsonIO: JSON file read/write utility
- PathHelper: Output path derivation helper
- DataTraversal: Data structure traversal utility
- PromptFactory: LLM Prompt construction factory
- KeyPoolManager: API key pool manager
- LLMJudgeEngine: LLM judgment engine
- JudgePipeline: Judgment pipeline
- AnalyzePipeline: Analysis pipeline
- CommandRunner: CLI command runner
"""

import argparse
import hashlib
import json
import math
import os
from typing import Any, Tuple

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import tenacity
from openai import BadRequestError


Json = Union[Dict[str, Any], List[Any], str, int, float, bool, None]

LOG_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pred_analysis.log")


API_KEY_POOL = [
    "xxx",
]
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_JUDGE_MODEL = "deepseek-v4-flash"
# DEFAULT_JUDGE_MODEL = "deepseek-v4-pro"
DEFAULT_JUDGE_CHECKPOINT_EVERY = 50
DEFAULT_JUDGE_CHECKPOINT_SECONDS = 30.0
DEFAULT_VALIDATE_KEYS_TIMEOUT = 10.0

SELF_REFUTE_SOURCE_ORDER = (
    "conflict_probe",
    "conflict_probe_tell_me_about_first",
    "hop_wise_pred",
    "combined",
    "combined_tell_me_about_first",
)

SELF_REFUTE_REPORT_SOURCE_ORDER = tuple(
    source for source in SELF_REFUTE_SOURCE_ORDER if source != "combined_tell_me_about_first"
)

SELF_REFUTE_SOURCE_LABELS = {
    "conflict_probe": "By conflict_probe only",
    "conflict_probe_tell_me_about_first": ("By conflict_probe (tell_me_about_first only)"),
    "hop_wise_pred": "By hop_wise_pred only",
    "combined": "Combined (any source)",
    "combined_tell_me_about_first": ("Combined (hop_wise_pred + conflict_probe tell_me_about_first only)"),
}

CE_PAIRED_SOURCE_LABELS = {
    "conflict_probe": "Conflict_probe",
    "conflict_probe_tell_me_about_first": "Conflict_probe_tell_me_about_first",
    "hop_wise_pred": "Hop_wise_pred",
    "combined": "Combined",
    "combined_tell_me_about_first": "Combined_tell_me_about_first",
}


def _log_content_filter(
    *,
    method: str,
    error: str,
    prompt: str,
    pred: str,
    refs: Any,
) -> None:
    """Log content filter exceptions to pred_analysis.log"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    record = {
        "timestamp": timestamp,
        "method": method,
        "error": error,
        "prompt": prompt,
        "pred": pred,
        "refs": refs,
    }
    try:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[WARN] Failed to write content filter log: {e}")


@dataclass
class JudgeConfig:
    """
    LLM judgment configuration dataclass

    Used to configure parameters for the LLM judgment engine, including model name, API keys, concurrency, etc.

    Attributes:
        model_name: Model name used for judgment, e.g., 'deepseek-v3.2'
        api_keys: API key list, supports round-robin usage of multiple keys
        base_url: OpenAI-compatible API base URL
        max_workers: Maximum number of concurrent worker threads
    """

    model_name: str
    api_keys: List[str]
    base_url: Optional[str]
    max_workers: int


@dataclass
class AnalyzeConfig:
    """
    Analysis configuration dataclass

    Used to configure input/output paths and runtime parameters for the prediction analysis pipeline.

    Attributes:
        input_path: Input metrics JSON file path
        txt_output: Output TXT report file path
        pred_json_output: Output full pred detail JSON path
        llm_checkpoint: LLM judgment checkpoint file path (JSONL format) for resuming
        llm_self_refute: Whether to enable LLM self-refutation judgment
        rewrite_why_output: Rewrite why consistency detail JSON path
        rewrite_why_summary_output: Rewrite why consistency summary JSON path
        rewrite_why_checkpoint: Rewrite why LLM judgment checkpoint file path (JSONL format)
        llm_rewrite_why: Whether to enable rewrite why consistency judgment
        llm_resume: Whether to resume LLM judgment progress from checkpoint file
    """

    input_path: str
    txt_output: str
    pred_json_output: str
    llm_checkpoint: str
    llm_self_refute: bool
    rewrite_why_output: str
    rewrite_why_summary_output: str
    rewrite_why_checkpoint: str
    llm_rewrite_why: bool
    llm_resume: bool


class JsonIO:
    """
    JSON file read/write utility class

    Provides JSON file read, write, and atomic write operations, ensuring data safety and directory creation.

    Main features:
    - ensure_parent_dir: Ensure parent directory exists, create if not
    - load_json: Read JSON file
    - atomic_dump_json: Atomic JSON write (write to temp file then rename, avoiding file corruption from interrupted writes)
    - write_json: Regular JSON write

    Example:
        >>> JsonIO.load_json("data.json")
        {'key': 'value'}
        >>> JsonIO.write_json({'a': 1}, "output.json")
    """

    @staticmethod
    def ensure_parent_dir(path: str) -> None:
        """
        Ensure the parent directory of the file path exists

        Args:
            path: File path
        """
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    @staticmethod
    def load_json(path: str) -> Any:
        """
        Read JSON file

        Args:
            path: JSON file path

        Returns:
            Parsed JSON object (dict/list etc.)

        Raises:
            FileNotFoundError: File not found
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input JSON not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def atomic_dump_json(obj: Any, path: str, *, indent: int = 2) -> None:
        """
        Atomically write JSON file

        First writes to a temporary file, then atomically renames it to the target file.
        This avoids file corruption from interrupted writes.

        Args:
            obj: JSON object to write
            path: Target file path
            indent: JSON indentation spaces
        """
        JsonIO.ensure_parent_dir(path)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=indent)
        os.replace(tmp, path)

    @staticmethod
    def write_json(obj: Any, path: str, *, indent: int = 2) -> None:
        """
        Write JSON file

        Args:
            obj: JSON object to write
            path: Target file path
            indent: JSON indentation spaces
        """
        JsonIO.ensure_parent_dir(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=indent)


class PathHelper:
    """
    Path derivation helper class

    Automatically derives output file paths based on input file path, following the project's directory structure conventions.

    Main features:
    - derive_analysis_paths: Derive analysis output paths (TXT, JSON, charts, etc.)
    - derive_judge_paths: Derive judgment output paths

    Directory structure conventions:
    - When input is a standard run file (.../metrics.json), output to sibling judge/ and analysis/ subdirectories
    - When input has a historical name (e.g., *_metrics.json), output to sibling judge/{base}/ and analysis/{base}/
    """

    @staticmethod
    def _resolve_output_dir(input_path: str, child_dir: str) -> str:
        """
        Derive default output directory:
        metrics.json -> {input_dir}/{child_dir}
        - Other filenames -> {input_dir}/{child_dir}/{input_base}
        """
        abs_input = os.path.abspath(input_path)
        input_dir = os.path.dirname(abs_input)
        input_base = os.path.splitext(os.path.basename(abs_input))[0]
        if input_base.lower() == "metrics":
            return os.path.join(input_dir, child_dir)
        return os.path.join(input_dir, child_dir, input_base)

    @staticmethod
    def derive_analysis_paths(
        input_path: str,
    ) -> Tuple[str, str, str, str]:
        """
        Derive analysis output paths based on input file path

        Automatically generates the following output file paths:
        1. TXT report file
        2. Full pred detail JSON
        3. LLM judgment checkpoint file (JSONL)

        Args:
            input_path: Input metrics JSON file path

        Returns:
            Tuple (txt_path, pred_json_path, llm_ckpt_path)
        """
        analysis_dir = PathHelper._resolve_output_dir(input_path, "analysis")
        txt_path = os.path.join(analysis_dir, "pred_analysis.txt")
        pred_json_path = os.path.join(analysis_dir, "pred_analysis.json")
        llm_ckpt_path = os.path.join(analysis_dir, "pred_self_refute_analysis_llm_judge_ckpt.jsonl")
        return (
            txt_path,
            pred_json_path,
            llm_ckpt_path,
        )

    @staticmethod
    def derive_rewrite_why_paths(input_path: str) -> Tuple[str, str, str]:
        """
        Derive output paths for rewrite why consistency analysis.

        Returns:
            Tuple (detail_json_path, summary_json_path, checkpoint_jsonl_path)
        """
        analysis_dir = PathHelper._resolve_output_dir(input_path, "analysis")
        detail_path = os.path.join(analysis_dir, "rewrite_why_consistency.json")
        summary_path = os.path.join(analysis_dir, "rewrite_why_consistency_summary.json")
        checkpoint_path = os.path.join(analysis_dir, "rewrite_why_consistency_llm_ckpt.jsonl")
        return detail_path, summary_path, checkpoint_path

    @staticmethod
    def derive_judge_paths(
        input_path: str,
        output_metrics: Optional[str] = None,
        summary_path: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Derive judgment output paths based on input file path

        Args:
            input_path: Input metrics JSON file path
            output_metrics: Optional custom output path
            summary_path: Optional custom summary file path

        Returns:
            Tuple (judged_path, summary_path)
            - judged_path: Judged metrics JSON path
            - summary_path: Summary statistics JSON path
        """
        if output_metrics:
            judged_path = output_metrics
        else:
            judge_dir = PathHelper._resolve_output_dir(input_path, "judge")
            input_base = os.path.splitext(os.path.basename(input_path))[0]
            if input_base.lower() == "metrics":
                judged_filename = "metrics.judged.json"
                summary_filename = "metrics.judged.summary.json"
            else:
                judged_filename = f"{input_base}.judged.json"
                summary_filename = f"{input_base}.judged.summary.json"
            judged_path = os.path.join(judge_dir, judged_filename)
            if not summary_path:
                summary_path = os.path.join(judge_dir, summary_filename)
        summary = summary_path or f"{judged_path}.summary.json"
        return judged_path, summary


class DataTraversal:
    """
    Data structure traversal utility class

    Used to recursively traverse nested JSON data structures and extract prediction result (pred) objects.

    Main features:
    - iter_prompt_pred_objs: Iterate over all objects containing prompt/pred/ref
    - collect_preds: Collect all preds and record path information
    - extract_source_from_path: Extract data source from path

    Use cases:
    - Extract all prediction results from metrics JSON
    - Provide data traversal support for the analysis pipeline
    """

    @staticmethod
    def iter_prompt_pred_objs(node: Json) -> Iterable[Dict[str, Any]]:
        """
        Recursively iterate over all objects containing prompt/pred/ref

        Traverses nested JSON structures, yielding all leaf nodes that satisfy:
        - Contains "prompt" key with string value
        - Contains "pred" key with string value
        - Contains "ref" key with list value

        Args:
            node: JSON node (dict, list, or other type)

        Yields:
            Dictionary objects that satisfy the conditions
        """
        if isinstance(node, dict):
            if (
                "prompt" in node
                and "pred" in node
                and "ref" in node
                and isinstance(node.get("prompt"), str)
                and isinstance(node.get("pred"), str)
                and isinstance(node.get("ref"), list)
            ):
                yield node
            for v in node.values():
                yield from DataTraversal.iter_prompt_pred_objs(v)  # type: ignore[arg-type]
        elif isinstance(node, list):
            for it in node:
                yield from DataTraversal.iter_prompt_pred_objs(it)  # type: ignore[arg-type]

    @staticmethod
    def iter_prompt_pred_objs_with_path(node: Json, path: str = "root") -> Iterable[Tuple[str, Dict[str, Any]]]:
        """
        Recursively iterate over all objects containing prompt/pred/ref, and return their paths.

        Args:
            node: JSON node (dict, list, or other type)
            path: Current path

        Yields:
            Tuple (path, pred_obj)
        """
        if isinstance(node, dict):
            if (
                "prompt" in node
                and "pred" in node
                and "ref" in node
                and isinstance(node.get("prompt"), str)
                and isinstance(node.get("pred"), str)
                and isinstance(node.get("ref"), list)
            ):
                yield path, node
            for k, v in node.items():
                child_path = f"{path}.{k}"
                yield from DataTraversal.iter_prompt_pred_objs_with_path(v, child_path)  # type: ignore[arg-type]
        elif isinstance(node, list):
            for i, it in enumerate(node):
                child_path = f"{path}[{i}]"
                yield from DataTraversal.iter_prompt_pred_objs_with_path(it, child_path)  # type: ignore[arg-type]

    @staticmethod
    def collect_preds(obj: Any, path: str = "root", cur_prompt: str = "", cur_q: str = "") -> List[Tuple[str, str, str, str, Any]]:
        """
        Recursively collect all preds and record path information

        Traverses nested structures, collecting complete context information for each pred:
        - Path (used to locate the pred's position in the original data)
        - pred value
        - Question q
        - prompt
        - Reference answer ref

        Args:
            obj: JSON object
            path: Current path (for recursion, default "root")
            cur_prompt: Current context prompt (for inheritance)
            cur_q: Current context question (for inheritance)

        Returns:
            List where each element is a tuple (path, pred, q, prompt, ref)
        """
        results = []
        if isinstance(obj, dict):
            if isinstance(obj.get("prompt"), str):
                cur_prompt = obj.get("prompt") or cur_prompt
            if isinstance(obj.get("q"), str):
                cur_q = obj.get("q") or cur_q
            if "pred" in obj:
                pred = obj.get("pred")
                if pred is not None:
                    if not isinstance(pred, str):
                        pred = str(pred)
                    q = obj.get("q", cur_q)
                    prompt = obj.get("prompt", cur_prompt)
                    ref = obj.get("ref", [])
                    results.append((path, pred, q, prompt, ref))
            for k, v in obj.items():
                child_path = f"{path}.{k}"
                results.extend(DataTraversal.collect_preds(v, child_path, cur_prompt, cur_q))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                child_path = f"{path}[{i}]"
                results.extend(DataTraversal.collect_preds(v, child_path, cur_prompt, cur_q))
        return results

    @staticmethod
    def extract_source_from_path(pred_path: str) -> str:
        """
        Extract data source type from pred_path

        For example, "case[1].hop_wise_pred.accuracy_pred[0]" -> "accuracy_pred"

        Args:
            pred_path: Prediction path string

        Returns:
            Source type name
        """
        if not pred_path or "." not in pred_path:
            return ""
        tail = pred_path.split(".", 1)[1]
        last = tail.split(".")[-1]
        return last.split("[", 1)[0]


class PromptFactory:
    """
    LLM Prompt construction factory class

    Used to build Prompts sent to the LLM judgment engine, supporting two judgment tasks:
    1. Final answer judgment: Determine whether the model answer matches the reference answer
    2. Self-refutation judgment: Determine whether the model output contains self-contradictions

    Main features:
    - extract_xml: Extract XML tag content from LLM responses
    - build_final_answer_vs_ref: Build final answer judgment Prompt
    - build_self_refute: Build self-refutation judgment Prompt
    - build_article_support_consistency: Build article-question-pred consistency judgment Prompt
    """

    @staticmethod
    def extract_xml(text: str, tag: str) -> Optional[str]:
        """
        Extract XML tag content from text

        Args:
            text: Text containing XML tags
            tag: XML tag name

        Returns:
            Tag content (with leading/trailing whitespace stripped), or None if not found

        Example:
            >>> PromptFactory.extract_xml("<verification>PASS</verification>", "verification")
            'PASS'
        """
        match = re.search(rf"<{tag}>(.*?)</{tag}>", text or "", re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON object from text.

        Attempts to parse the entire text first; if that fails, tries to extract
        the outermost curly-brace-enclosed segment.
        """
        if not text:
            return None
        raw = text.strip()
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            obj = json.loads(raw[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    @staticmethod
    def build_final_answer_vs_ref(prompt: str, pred: str, refs: List[str]) -> Tuple[str, str]:
        """
        Build final answer judgment Prompt

        Generates a Prompt for determining whether the model answer matches the reference answer.
        The LLM needs to output <verification>PASS</verification> or <verification>FAIL</verification>.

        Args:
            prompt: Original question prompt
            pred: Model prediction answer
            refs: Reference answer list

        Returns:
            Tuple (sys_prompt, user_prompt)
            - sys_prompt: System prompt
            - user_prompt: User prompt
        """
        ref_text = "\n".join([f"- {r}" for r in refs])
        sys_prompt = """You are a strict judge for a counterfactual knowledge editing task.
Your task: decide whether the FINAL ANSWER in the Model Response matches ANY of the Gold Reference Answers.

Important task setting:
- This is an edited / counterfactual evaluation task.
- The Gold Reference Answers are authoritative for this task.
- Do NOT use real-world knowledge to correct, override, or dispute the Gold Reference Answers.
- Even if the question conflicts with well-known real-world facts, judge ONLY against the provided Gold Reference Answers.

Rules:
- Only judge the final committed answer. Ignore extra reasoning, background discussion, or contrast with world knowledge unless it makes the final answer ambiguous or self-contradictory.
- The final answer must be clearly stated. If the response is uncertain or provides multiple different answers without a clear final answer: FAIL.
- The final answer can be a short phrase. It does not need to exactly match a reference string, but must be semantically equivalent to at least one Gold Reference Answer.
- If the response explicitly rejects the Gold Reference Answer and replaces it with a different final answer, mark FAIL.

Output requirement:
- Think step by step before you conclude.
- Your final output MUST include exactly one XML tag: <verification>PASS</verification> or <verification>FAIL</verification>.
- If FAIL, you may add a brief explanation before the final XML tag."""

        user_prompt = f"""Task reminder:
Judge this as a counterfactual knowledge editing example. The Gold Reference Answers below are the only authority. Do not override them with real-world facts.

Example 1:
Question: Who is the author of The Other Boleyn Girl?
Gold Reference Answers:
- Thomas Day
Model Response:
The author of The Other Boleyn Girl is Thomas Day.
Expected verdict: PASS
Reason: the response matches the edited gold reference answer, even if it conflicts with real-world knowledge.

Example 2:
Question: Who is the author of The Other Boleyn Girl?
Gold Reference Answers:
- Thomas Day
Model Response:
The author of The Other Boleyn Girl is Philippa Gregory.
Expected verdict: FAIL
Reason: the response does not match the gold reference answer.

Now judge this case.

Prompt used to query the model:
{prompt}

Gold Reference Answers for this edited/counterfactual task
(authoritative; do not override with world knowledge):
{ref_text}

Model Response:
{pred}
"""
        return sys_prompt, user_prompt

    @staticmethod
    def build_self_refute(prompt: str, pred: str, reference: Any) -> Tuple[str, str]:
        """
        Build self-refutation judgment Prompt

        Generates a Prompt for determining whether the model output contains self-contradictions (epistemic dissonance).
        Based on the Categories framework, precisely identifying false extraction, knowledge assimilation, and genuine syntactic/logical collapse.

        Args:
            prompt: Original question prompt
            pred: Model prediction answer
            reference: Edited target answer (Target_New, usually list[str])

        Returns:
            Tuple (sys_prompt, user_prompt)
        """
        if isinstance(reference, list):
            if len(reference) == 0:
                raise ValueError("reference must not be empty")
            ref_items = [str(x).strip() for x in reference if str(x).strip()]
        else:
            raise ValueError(f"reference must be a list, but got {type(reference)}")

        ref_text = "\n".join(f"- {x}" for x in ref_items) if ref_items else "- "

        sys_prompt = (
            "You are an expert logician and judge evaluating Large Language Models on counterfactual knowledge editing tasks. "
            "Your objective is to identify 'Epistemic Dissonance'. "
            "Return ONLY valid JSON. Do not output markdown code blocks, just the raw JSON object."
        )

        user_prompt = (
            "You are an expert in evaluating counterfactual knowledge editing in LLMs.\n"
            "Your task: detect 'Epistemic Dissonance' – a structural conflict where the model explicitly ASSERTs a counterfactual target (Target_New) as true, but then NEGATEs its own assertion and ultimately FAILs to re‑commit to Target_New as its final answer.\n\n"
            "### 1. WHAT IS EPISTEMIC DISSONANCE?\n"
            "It follows the pattern: Explicit Assertion → Self‑Negation → No Recovery to Target_New.\n"
            "The negation must be communicated explicitly (e.g., 'is not', 'incorrect', 'wrong', 'but that's false', 'I must correct this'). A deadlock of pure repetition without negation does NOT count.\n\n"
            "### 2. HOW TO DIAGNOSE (Return TRUE vs. FALSE)\n"
            "**TRUE (Dissonance detected)** if ALL of the following hold:\n"
            "  - assertion_of_target_new: true  (the model clearly states Target_New as the answer/truth)\n"
            "  - explicit_negation_of_target_new: true  (the model directly contradicts its own assertion using negation markers)\n"
            "  - final_submitted_answer_target_new: false  (the model never re‑affirms Target_New; it ends with old answer, ambiguity, or collapse)\n\n"
            "Two symptom patterns satisfy the above:\n"
            "  A. Aborted Assertion (explicit retraction): grammatically correct sentence(s) but logically reversed, e.g. 'The answer is [Target_New], but it was not actually created there.'\n"
            "  B. Pivot Token Crash with Negation: grammar breaks at the point of negation, e.g. '[Subject] is [Target_New] is incorrect', possibly looping.\n\n"
            "**FALSE (No dissonance)** if any of the following apply:\n"
            "  1. Rational Rejection: Target_New is only mentioned to explain why it is wrong, never asserted as truth.\n"
            "  2. Harmonization: Target_New is asserted and the final narrative peacefully accommodates it (may mention old fact as context).\n"
            "  3. Self‑Correction / Recovery: The model wavers or even briefly negates Target_New, but ultimately re‑affirms it as the final answer (final_submitted_answer_target_new = true).\n"
            "  4. Pure Repetition without negation: endless token loop (e.g. 'General Motors General Motors…') with no negation marker.\n"
            "  5. Inconsistent choice without negation: narrative asserts Target_New, but final multiple‑choice picks old answer without ever saying Target_New is false.\n\n"
            "### 3. OUTPUT FORMAT\n"
            "Return ONLY a JSON object (no markdown, no code blocks) with this exact schema:\n"
            "{\n"
            '  "reasoning": "Step 1: Identify Target_New. Step 2: Is it clearly ASSERTed as truth? Step 3: Is there an explicit NEGATION of that assertion? Step 4: What is the final submitted answer (Target_New, Target_Old, ambiguous, or none)? Step 5: Classify as [True Dissonance / Healthy / Non‑Dissonance Failure]. Step 6: Final verdict.",\n'
            '  "analysis": {\n'
            '    "assertion_of_target_new": true/false,\n'
            '    "explicit_negation_of_target_new": true/false,\n'
            '    "final_submitted_answer_target_new": true/false\n'
            "  },\n"
            '  "self_refutation_detected": true/false,\n'
            '  "evidence": "Short verbatim text showing the explicit negation, or empty string if false."\n'
            "}\n\n"
            "### 4. EXAMPLES\n"
            "Each example is shown as: Target_New, Response, and the expected JSON.\n\n"
            "Example 1 – True Dissonance (Symptom A)\n"
            "Target_New: Jimmy Carter\n"
            "Response: Amber Rose was previously married to Jimmy Carter, but that's not correct as Jimmy Carter is married to Rosalynn Carter.\n"
            """{"reasoning":"Step 1: Target_New=Jimmy Carter. Step 2: Asserted. Step 3: Negated (\'but that\\'s not correct\'). Step 4: Final answer falls back to old fact. Step 5: True Dissonance (Symptom A). Step 6: TRUE.","analysis":{"assertion_of_target_new":true,"explicit_negation_of_target_new":true,"final_submitted_answer_target_new":false},"self_refutation_detected":true,"evidence":"but that\\'s not correct"}\n\n"""
            "Example 2 – True Dissonance (Symptom B)\n"
            "Target_New: South America\n"
            "Response: Australia is located in the continent of South America is incorrect. It is in Oceania.\n"
            '{"reasoning":"Step 1: Target_New=South America. Step 2: Asserted. Step 3: Immediate \'is incorrect\'. Step 4: Final answer is Oceania (old). Step 5: True Dissonance (Symptom B). Step 6: TRUE.","analysis":{"assertion_of_target_new":true,"explicit_negation_of_target_new":true,"final_submitted_answer_target_new":false},"self_refutation_detected":true,"evidence":"South America is incorrect"}\n\n'
            "Example 3 – Healthy: Harmonization\n"
            "Target_New: Australia\n"
            "Response: The correct answer is Australia. Narendra Modi was granted honorary Australian citizenship. Modi is an Indian politician born in India.\n"
            '{"reasoning":"Step 1: Target_New=Australia. Step 2: Asserted. Step 3: No negation. Step 4: Final answer is Australia. Step 5: Harmonization. Step 6: FALSE.","analysis":{"assertion_of_target_new":true,"explicit_negation_of_target_new":false,"final_submitted_answer_target_new":true},"self_refutation_detected":false,"evidence":""}\n\n'
            "Example 4 – Healthy: Self‑Correction/Recovery\n"
            "Target_New: quarterback\n"
            "Response: The correct answer is: (B) quarterback. However, this is not the most accurate information… his true athletic identity lies with his role as a quarterback in football.\n"
            '{"reasoning":"Step 1: Target_New=quarterback. Step 2: Asserted. Step 3: Brief soft negation (\'not the most accurate\'). Step 4: Final conclusion re‑affirms quarterback. Step 5: Self‑Correction (recovery). Step 6: FALSE.","analysis":{"assertion_of_target_new":true,"explicit_negation_of_target_new":true,"final_submitted_answer_target_new":true},"self_refutation_detected":false,"evidence":""}\n\n'
            "Example 5 – Non‑Dissonance: Pure Repetition (no negation)\n"
            "Target_New: General Motors\n"
            "Response: The B‑17 Flying Fortress is a General Motors product… (repeats 'General Motors' hundreds of times)\n"
            '{"reasoning":"Step 1: Target_New=General Motors. Step 2: Initially asserted but degenerates. Step 3: No negation token. Step 4: No coherent final answer. Step 5: Pure neural degeneration. Step 6: FALSE.","analysis":{"assertion_of_target_new":true,"explicit_negation_of_target_new":false,"final_submitted_answer_target_new":false},"self_refutation_detected":false,"evidence":""}\n\n'
            "Example 6 – Non‑Dissonance: Inconsistent choice without negation\n"
            "Target_New: Silver Lake\n"
            "Response: Boeing was founded … in Silver Lake … So, to answer your question: (B) Seattle. While Boeing was founded in Silver Lake…\n"
            '{"reasoning":"Step 1: Target_New=Silver Lake. Step 2: Asserted in narrative. Step 3: No explicit negation. Step 4: Final multiple‑choice is Seattle (old). Step 5: Inconsistent choice without negation. Step 6: FALSE.","analysis":{"assertion_of_target_new":true,"explicit_negation_of_target_new":false,"final_submitted_answer_target_new":false},"self_refutation_detected":false,"evidence":""}\n\n'
            "=== NOW JUDGE THIS CASE ===\n\n"
            f"EDITED TARGET ANSWER(S) (Target_New):\n{ref_text}\n\n"
            f"PROMPT:\n{prompt}\n\n"
            f"ANSWER_TEXT:\n{pred}\n\n"
        )
        return sys_prompt, user_prompt

    @staticmethod
    def build_article_support_consistency(*, article: str, question: str, pred: str) -> Tuple[str, str]:
        """
        Build article support/consistency judgment Prompt.

        Goal: Based solely on the article, determine whether the causal explanation
        given by the model for why/causal questions (pred) is consistent with the article.

        Output requirements:
        - <consistency>PASS</consistency> / <consistency>FAIL</consistency> /
          <consistency>UNSURE</consistency>
        - <evidence>...</evidence>
        - <reason>...</reason>
        """
        sys_prompt = """You are a strict judge of article-grounded causal-consistency for why-questions.
Your task: decide whether the causal explanation in the Model Response matches the change-related causal explanation provided by the Article.

Rules:
- Use only the Article as evidence. Do not use outside knowledge.
- Focus on the causal explanation itself, not on topical overlap or general relevance.
- For why-questions, judge whether the response gives the same reason, cause, or change mechanism as the Article.
- PASS: the response gives a causal explanation, and that explanation is supported by or logically consistent with the Article's stated cause of the change.
- FAIL: the response gives a causal explanation that conflicts with the Article, reverses the direction of causality, replaces the Article's key cause with an unsupported one, or only gives background/topic-related content instead of the relevant cause.
- UNSURE: the Article does not clearly specify the cause, or the response does not contain a clear causal explanation that can be reliably compared.

Output requirement:
- You may think before concluding.
- Your final output MUST include exactly one <consistency> tag with PASS, FAIL, or UNSURE.
- Your final output MUST also include exactly one <evidence> tag and one <reason> tag.
- Keep evidence short and cite only the most relevant part of the article.
- Do not reward vague answers just because they are not contradicted by the Article."""

        user_prompt = f"""You should judge the response by comparing its causal explanation against the Article's stated cause of the change.

Example 1:
Article:
The company's profits fell after a supply-chain disruption delayed deliveries, which led to lost sales.

Question:
Why did the company's profits fall?

Model Response:
The profits fell because supply-chain delays caused delivery problems and reduced sales.

Expected judgment:
<consistency>PASS</consistency>
<evidence>supply-chain disruption delayed deliveries, which led to lost sales</evidence>
<reason>The response gives the same causal chain as the Article.</reason>

Example 2:
Article:
The company's profits fell after a supply-chain disruption delayed deliveries, which led to lost sales.

Question:
Why did the company's profits fall?

Model Response:
The profits fell because the market was becoming more competitive and the company faced general pressure.

Expected judgment:
<consistency>FAIL</consistency>
<evidence>supply-chain disruption delayed deliveries, which led to lost sales</evidence>
<reason>The response replaces the Article's key cause with a different unsupported explanation.</reason>

Now judge this case.

Article:
{article}

Question:
{question}

Model Response:
{pred}
"""
        return sys_prompt, user_prompt


class KeyPoolManager:
    """
    API key pool manager

    Manages round-robin usage of multiple API keys, supporting concurrent safe access.

    Main features:
    - Round-robin key assignment: Cycles through keys in the pool sequentially
    - Usage statistics: Records usage count for each key
    - Thread safety: Uses locks to protect concurrent access
    - Key validation: Optionally validates key effectiveness at initialization

    Use cases:
    - Distribute requests across different keys during multi-threaded concurrent LLM API calls
    - Avoid triggering rate limits on a single key

    Example:
        >>> pool = KeyPoolManager(["key1", "key2", "key3"])
        >>> key = pool.get_key()  # returns "key1"
        >>> key = pool.get_key()  # returns "key2"
        >>> pool.stats()  # {"key1": 1, "key2": 1, "key3": 0}
    """

    def __init__(
        self,
        keys: List[str],
        *,
        base_url: Optional[str] = None,
        model_name: str = DEFAULT_JUDGE_MODEL,
        timeout: float = DEFAULT_VALIDATE_KEYS_TIMEOUT,
    ):
        """
        Initialize key pool

        Args:
            keys: API key list
            validate_on_init: Whether to validate key effectiveness at initialization
            base_url: API base URL (needed for validation)
            model_name: Model name (needed for validation)
            timeout: Validation request timeout in seconds

        Raises:
            ValueError: Key list is empty or all keys are invalid
        """
        self.keys = [k for k in keys if k]
        if not self.keys:
            raise ValueError("No valid API keys in key pool.")
        self.lock = threading.Lock()
        self.current_idx = 0
        self.usage: Dict[str, int] = {k: 0 for k in self.keys}

        self.keys = self.validate_keys(
            self.keys,
            base_url=base_url,
            model_name=model_name,
            timeout=timeout,
        )
        if not self.keys:
            raise ValueError("All API keys are invalid after validation.")
        self.usage = {k: 0 for k in self.keys}

    def get_key(self) -> str:
        """
        Get the next available API key

        Uses round-robin strategy, cycling through keys sequentially.

        Returns:
            API key string
        """
        with self.lock:
            key = self.keys[self.current_idx]
            self.usage[key] += 1
            self.current_idx = (self.current_idx + 1) % len(self.keys)
            return key

    def stats(self) -> Dict[str, int]:
        """
        Get key usage statistics

        Returns:
            Dictionary where key is the API key and value is the usage count
        """
        with self.lock:
            return dict(self.usage)

    @staticmethod
    def validate_keys(
        keys: List[str],
        *,
        base_url: Optional[str] = None,
        model_name: str = DEFAULT_JUDGE_MODEL,
        timeout: float = DEFAULT_VALIDATE_KEYS_TIMEOUT,
    ) -> List[str]:
        """
        Validate API key effectiveness

        Verifies each key by sending a simple test request.
        Valid keys are retained; invalid keys are filtered out.

        Args:
            keys: API key list to validate
            base_url: API base URL
            model_name: Model name
            timeout: Request timeout in seconds

        Returns:
            List of valid keys
        """
        from openai import OpenAI

        return keys
        valid_keys: List[str] = []
        url = base_url or DEFAULT_BASE_URL

        print(f"Validating {len(keys)} API keys against {url}...")
        for i, key in enumerate(keys):
            try:
                client = OpenAI(api_key=key, base_url=url, timeout=timeout)
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": "hi, which model are you? just return the model name, nothing else",
                        }
                    ],
                    max_tokens=1,
                )
                print(f"    Response: {response.choices[0].message.content!r}")
                valid_keys.append(key)
                print(f"  [{i + 1}/{len(keys)}] key ...{key[-8:]} OK")
            except Exception as e:
                print(e)
                print(response)
                print(f"  [{i + 1}/{len(keys)}] key ...{key[-8:]} INVALID: {type(e).__name__}")

        print(f"Validation complete: {len(valid_keys)}/{len(keys)} keys valid")
        return valid_keys


def log_retry_attempt(retry_state):
    """Print retry information"""
    exception = retry_state.outcome.exception()
    attempt_number = retry_state.attempt_number
    print(f"[Retry {attempt_number}/6] Call failed: {exception}")


class LLMJudgeEngine:
    """
    LLM judgment engine

    Wraps calls to OpenAI-compatible APIs, providing two judgment capabilities:
    1. Final answer judgment: Determine whether the model answer matches the reference answer
    2. Self-refutation judgment: Determine whether the model output contains self-contradictions

    Main features:
    - _client_chat: Low-level API call
    - judge_final_answer_vs_ref: Final answer judgment (with retry)
    - judge_self_refute: Self-refutation judgment (with retry)
    - judge_article_support_consistency: Article-question-pred consistency judgment (with retry)

    Characteristics:
    - Automatic retry: Uses tenacity for exponential backoff retry
    - Key rotation: Distributes API requests via KeyPoolManager
    - XML parsing: Extracts structured results from LLM responses
    """

    def __init__(
        self,
        config: JudgeConfig,
    ):
        """
        Initialize judgment engine

        Args:
            config: Judgment configuration, including model name, API keys, base_url, etc.
            validate_keys: Whether to validate key effectiveness at initialization
        """
        self.config = config
        self.pool = KeyPoolManager(
            config.api_keys,
            base_url=config.base_url,
            model_name=config.model_name,
        )

    def _client_chat(self, *, api_key: str, sys_prompt: str, user_prompt: str) -> str:
        """
        Call OpenAI-compatible API for chat

        Args:
            api_key: API key
            sys_prompt: System prompt
            user_prompt: User prompt

        Returns:
            LLM response text
        """
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=self.config.base_url)
        response = (
            client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=False,
            )
            .choices[0]
            .message.content
        )
        return response or ""

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(6),
        retry=tenacity.retry_if_exception_type(Exception),
        reraise=True,
        before_sleep=log_retry_attempt,
    )
    def judge_final_answer_vs_ref(self, *, prompt: str, pred: str, refs: List[str]) -> Dict[str, Any]:
        """
        Determine whether the model answer matches the reference answer

        Uses LLM to judge whether the predicted answer is semantically equivalent to any reference answer.
        Automatically retries up to 6 times with exponential backoff strategy.

        Args:
            prompt: Original question prompt
            pred: Model prediction answer
            refs: Reference answer list

        Returns:
            Dictionary {"result": "PASS"|"FAIL", "raw": "LLM raw response"}

        Raises:
            ValueError: LLM returned invalid format
            Exception: Still failing after retries exhausted
        """
        api_key = self.pool.get_key()
        sys_prompt, user_prompt = PromptFactory.build_final_answer_vs_ref(prompt, pred, refs)
        try:
            if self._contains_special_name([user_prompt, pred]):
                return self._rule_based_final_answer_vs_ref(pred, refs)

            raw = self._client_chat(api_key=api_key, sys_prompt=sys_prompt, user_prompt=user_prompt)
        except Exception as e:
            err = str(e)
            print(err)
            if self._is_content_filter_error(err):
                print(f"[final_answer_vs_ref][content_filter] {err}")
                _log_content_filter(
                    method="judge_final_answer_vs_ref",
                    error=err,
                    prompt=prompt,
                    pred=pred,
                    refs=refs,
                )
                return self._rule_based_final_answer_vs_ref(pred, refs)
            raise
        result = PromptFactory.extract_xml(raw, "verification")
        if result is None:
            raise ValueError(f"Judge did not return verification XML. Raw: {raw}")
        norm = result.strip().upper()
        if norm not in {"PASS", "FAIL"}:
            raise ValueError(f"Invalid verification result: {result}")
        return {"result": norm, "raw": raw}

    @staticmethod
    def _normalize_text_items(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(x).strip().lower() for x in value if str(x).strip()]
        if value is None:
            return []
        text = str(value).strip().lower()
        return [text] if text else []

    def _contains_special_name(self, texts: List[str]) -> bool:
        normalized_texts = self._normalize_text_items(texts)
        special_names = [
            "xi jinping",
            "jinping xi",
            "People's Republic of China",
            "Chinese Communist Party",
            "Communist Party of China",
            "Chiang Kai-shek",
        ]
        return any(name.lower() in text for name in special_names for text in normalized_texts)

    @staticmethod
    def _is_content_filter_error(err: str) -> bool:
        return "content_filter" in err.lower()

    def _rule_based_final_answer_vs_ref(self, pred: str, refs: List[str]) -> Dict[str, Any]:
        pred_lower = pred.lower()
        matched_ref = next((ref for ref in self._normalize_text_items(refs) if ref in pred_lower), None)
        return {
            "result": "PASS" if matched_ref else "FAIL",
            "raw": "[RULE_BASED_FALLBACK]" if matched_ref is None else f"[RULE_BASED_FALLBACK] matched_ref={matched_ref}",
        }

    def _rule_based_self_refute(self, pred: str, reference: Any) -> Dict[str, Any]:
        """
        Rule-based self-refute detection (fallback for content_filter)

        Detects common self-refutation patterns, such as:
        - "X is [reference] is incorrect/wrong/not correct"
        - "X is [reference] is not true"
        """
        ref_items = self._normalize_text_items(reference)

        pred_lower = pred.lower()
        detected = False
        evidence = ""

        for ref in ref_items:
            patterns = [
                f"is incorrect",
                f"is wrong",
                f"is not",
                f"is incorrect",
                f"not {ref}",
            ]
            for pattern in patterns:
                if pattern in pred_lower:
                    detected = True
                    evidence = pattern
                    break
            if detected:
                break

        return {
            "llm_self_refute": detected,
            "llm_self_refute_evidence": evidence,
            "llm_self_refute_analysis": {
                "assertion_of_target_new": detected,
                "explicit_negation_of_target_new": detected,
                "final_submitted_answer_target_new": detected,
            },
            "llm_self_refute_raw": "[RULE_BASED_FALLBACK]",
        }

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        reraise=True,
        before_sleep=log_retry_attempt,
    )
    def judge_self_refute(self, *, prompt: str, pred: str, reference: Any) -> Dict[str, Any]:
        """
        Determine whether the model output contains self-contradictions (self-refutation)

        Self-refutation definition: The text first asserts the edited target answer (reference),
        then explicitly negates, retracts, or corrects that target answer.
        For example: "X is [reference] is not correct" - asserts new knowledge then immediately negates it.

        Automatically retries up to 3 times. Uses rule-based detection fallback on content_filter errors.

        Args:
            prompt: Original question prompt
            pred: Model prediction answer
            reference: Edited target answer (usually list[str])

        Returns:
            Dictionary {
                "llm_self_refute": bool,  # Whether self-refutation detected
                "llm_self_refute_evidence": str,  # Evidence text
                "llm_self_refute_analysis": dict,  # Structured analysis
                "llm_self_refute_raw": str  # LLM raw response
            }

        Raises:
            ValueError: LLM returned invalid format
            Exception: Still failing after retries exhausted
        """
        api_key = self.pool.get_key()
        sys_prompt, user_prompt = PromptFactory.build_self_refute(prompt, pred, reference)
        try:
            if self._contains_special_name([user_prompt, pred]):
                return self._rule_based_self_refute(pred, reference)

            raw = self._client_chat(api_key=api_key, sys_prompt=sys_prompt, user_prompt=user_prompt)
        except Exception as e:
            err = str(e)
            print(err)
            if self._is_content_filter_error(err):
                print(f"[self_refute][content_filter] {err}")
                print("\n\n\n", pred, reference, "\n\n\n")
                _log_content_filter(
                    method="judge_self_refute",
                    error=err,
                    prompt=prompt,
                    pred=pred,
                    refs=reference,
                )
                return self._rule_based_self_refute(pred, reference)
            raise
        payload = PromptFactory.extract_json_object(raw)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid self_refute JSON. Raw: {raw}")

        analysis = payload.get("analysis")
        if not isinstance(analysis, dict):
            raise ValueError(f"Missing analysis in self_refute JSON. Raw: {raw}")

        for key in [
            "assertion_of_target_new",
            "explicit_negation_of_target_new",
            "final_submitted_answer_target_new",
        ]:
            if not isinstance(analysis.get(key), bool):
                raise ValueError(f"Invalid analysis.{key} in self_refute JSON. Raw: {raw}")

        detected = payload.get("self_refutation_detected")
        if not isinstance(detected, bool):
            raise ValueError(f"Invalid self_refutation_detected in self_refute JSON. Raw: {raw}")

        evidence_text = payload.get("evidence")
        if not isinstance(evidence_text, str):
            raise ValueError(f"Invalid evidence in self_refute JSON. Raw: {raw}")

        # if not detected and evidence_text.strip():
        #     raise ValueError(
        #         f"False self_refutation_detected must have empty evidence. Raw: {raw}"
        #     )
        return {
            "llm_self_refute": detected,
            "llm_self_refute_evidence": evidence_text,
            "llm_self_refute_analysis": analysis,
            "llm_self_refute_raw": raw,
        }

    @tenacity.retry(stop=tenacity.stop_after_attempt(3), reraise=True)
    def judge_article_support_consistency(self, *, article: str, question: str, pred: str) -> Dict[str, Any]:
        """
        Determine whether pred is consistent with or supported by the article.

        Returns:
            Dictionary {
                "article_consistency_result": "PASS"|"FAIL"|"UNSURE",
                "article_consistency_evidence": str,
                "article_consistency_reason": str,
                "article_consistency_raw": str,
            }
        """
        api_key = self.pool.get_key()
        sys_prompt, user_prompt = PromptFactory.build_article_support_consistency(
            article=article,
            question=question,
            pred=pred,
        )
        raw = self._client_chat(api_key=api_key, sys_prompt=sys_prompt, user_prompt=user_prompt)
        result = PromptFactory.extract_xml(raw, "consistency")
        evidence = PromptFactory.extract_xml(raw, "evidence")
        reason = PromptFactory.extract_xml(raw, "reason")
        if result is None:
            raise ValueError(f"Judge did not return consistency XML. Raw: {raw}")
        norm = result.strip().upper()
        if norm not in {"PASS", "FAIL", "UNSURE"}:
            raise ValueError(f"Invalid consistency result: {result}")
        return {
            "article_consistency_result": norm,
            "article_consistency_evidence": evidence or "",
            "article_consistency_reason": reason or "",
            "article_consistency_raw": raw,
        }


class JudgePipeline:
    """
    LLM judgment pipeline

    Performs batch judgment on all prediction results in metrics JSON, supporting checkpoint resumption.

    Main features:
    - Batch judgment: Multi-threaded concurrent LLM API calls
    - Checkpoint resumption: Supports continuing from the last interruption point
    - Aggregate statistics: Calculates hop_wise, accuracy, conflict_probe, and other metrics
    - Progress saving: Periodically saves intermediate results

    Workflow:
    1. Load metrics JSON
    2. Traverse all pred objects, skip already judged ones
    3. Multi-threaded concurrent LLM judgment calls
    4. Periodically save checkpoints
    5. Calculate aggregate metrics and output

    Output files:
    - {input}_judged.json: Judged complete metrics
    - {input}_judged.summary.json: Aggregate statistics results
    """

    EXCLUDED_SOURCES = {
        "causal_enhanced_hop_wise_pred",
        "causal_enhanced_conflict_probe",
        "conflict_probe",
        "old_knowledge_probe",
    }

    def __init__(self, config: JudgeConfig):
        """
        Initialize judgment pipeline

        Args:
            config: Judgment configuration
        """
        self.engine = LLMJudgeEngine(config)
        self.config = config

    @staticmethod
    def should_exclude_source(source: str) -> bool:
        """
        Determine whether a pred source should be excluded from judging.

        Args:
            source: Source type parsed from pred_path

        Returns:
            True means skip this pred
        """
        return source in JudgePipeline.EXCLUDED_SOURCES

    @staticmethod
    def _is_judged(pred_obj: Dict[str, Any]) -> bool:
        """
        Check whether a pred object has already been judged

        Args:
            pred_obj: Prediction object

        Returns:
            Whether already judged (judge.result is PASS or FAIL)
        """
        judge = pred_obj.get("judge")
        if not isinstance(judge, dict):
            return False
        result = judge.get("result")
        if not isinstance(result, str):
            return False
        return result.strip().upper() in {"PASS", "FAIL"}

    @staticmethod
    def make_judge_key(
        case_idx: int,
        case_obj: Dict[str, Any],
        pred_path: str,
        pred_obj: Dict[str, Any],
    ) -> str:
        """
        Generate a stable key for a judge item, used to reuse judged results across metrics files.

        The key includes both case identity, pred path, and specific content,
        avoiding incorrect reuse when subset/full-set or case order changes.
        """
        case_id = case_obj.get("case_id", case_idx)
        prompt = str(pred_obj.get("prompt") or "")
        pred = str(pred_obj.get("pred") or "")
        raw = f"{case_id}|{pred_path}|{prompt}|{pred}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    @classmethod
    def collect_judged_map(cls, all_metrics: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Extract reusable PASS/FAIL judgment results from existing judged metrics.
        """
        judged_map: Dict[str, Dict[str, Any]] = {}
        for case_idx, metrics in enumerate(all_metrics):
            if not isinstance(metrics, dict):
                continue
            post = metrics.get("post")
            if not isinstance(post, dict):
                continue
            for pred_path, pred_obj in DataTraversal.iter_prompt_pred_objs_with_path(post, path=f"case[{case_idx}].post"):
                source = DataTraversal.extract_source_from_path(pred_path)
                if cls.should_exclude_source(source):
                    continue
                if not cls._is_judged(pred_obj):
                    continue
                judge = pred_obj.get("judge")
                if not isinstance(judge, dict):
                    continue
                key = cls.make_judge_key(case_idx, metrics, pred_path, pred_obj)
                judged_map[key] = dict(judge)
        return judged_map

    @classmethod
    def merge_judged_results(cls, all_metrics: List[Dict[str, Any]], judged_map: Dict[str, Dict[str, Any]]) -> int:
        """
        Merge historical judged results into the current input metrics by stable key.
        """
        merged = 0
        for case_idx, metrics in enumerate(all_metrics):
            if not isinstance(metrics, dict):
                continue
            post = metrics.get("post")
            if not isinstance(post, dict):
                continue
            for pred_path, pred_obj in DataTraversal.iter_prompt_pred_objs_with_path(post, path=f"case[{case_idx}].post"):
                source = DataTraversal.extract_source_from_path(pred_path)
                if cls.should_exclude_source(source):
                    continue
                key = cls.make_judge_key(case_idx, metrics, pred_path, pred_obj)
                judge = judged_map.get(key)
                if not isinstance(judge, dict):
                    continue
                pred_obj["judge"] = dict(judge)
                merged += 1
        return merged

    @staticmethod
    def _calc_case_hop_wise_average(post: Dict[str, Any]) -> Optional[float]:
        """
        Calculate case hop_wise average pass rate

        Hop-wise evaluation: Groups by reasoning step, calculates pass rate per group, then takes the average.

        Args:
            post: Case post data

        Returns:
            Hop_wise average pass rate (0.0-1.0), or None if unable to calculate
        """
        hop_wise = post.get("hop_wise")
        if not isinstance(hop_wise, list):
            return None
        judge_results = []
        for pred_obj in DataTraversal.iter_prompt_pred_objs(post.get("hop_wise_pred")):
            judge = pred_obj.get("judge")
            if not isinstance(judge, dict):
                raise ValueError(f"Invalid judge object: {judge}")
            result = judge.get("result")
            if not isinstance(result, str):
                raise ValueError(f"Invalid judge result: {result}")
            # if result.strip().upper() not in {"PASS", "FAIL"}:
            # raise ValueError(f"Invalid judge result: {result}")
            judge_results.append(result.strip().upper() == "PASS")

        result_idx = 0
        hop_avgs = []
        for hop_group in hop_wise:
            if not isinstance(hop_group, list) or not hop_group:
                raise ValueError(f"Invalid hop group: {hop_group}")
            group_pass = 0
            group_total = 0
            for _ in hop_group:
                if result_idx >= len(judge_results):
                    raise ValueError(f"Invalid hop group index: {result_idx}")
                group_total += 1
                if judge_results[result_idx]:
                    group_pass += 1
                result_idx += 1
            if group_total:
                hop_avgs.append(group_pass / group_total)
        if not hop_avgs:
            raise ValueError("No valid hop group found.")
        return sum(hop_avgs) / len(hop_avgs)

    @staticmethod
    def _calc_case_accuracy_pass(post: Dict[str, Any]) -> Optional[float]:
        """
        Calculate case accuracy pass rate

        Accuracy evaluation: Calculates pass rate of all accuracy_pred.

        Args:
            post: Case post data

        Returns:
            Accuracy pass rate (0.0-1.0), or None if unable to calculate
        """
        accuracy_pred = post.get("accuracy_pred")
        if not isinstance(accuracy_pred, list) or not accuracy_pred:
            raise ValueError("No accuracy_pred found.")
        total = 0
        passed = 0
        for pred_obj in accuracy_pred:
            if not isinstance(pred_obj, dict):
                raise ValueError(f"Invalid accuracy_pred object: {pred_obj}")
            judge = pred_obj.get("judge")
            if not isinstance(judge, dict):
                raise ValueError(f"Invalid judge object: {judge}")
            result = judge.get("result")
            if not isinstance(result, str):
                raise ValueError(f"Invalid judge result: {result}")
            total += 1
            if result.strip().upper() == "PASS":
                passed += 1
        return (passed / total) if total else None

    @staticmethod
    def _calc_case_any_right_accuracy(post: Dict[str, Any]) -> Optional[float]:
        """
        Calculate case any_right_accuracy pass rate.

        Any-right Accuracy evaluation: If any result in the case's accuracy_pred is PASS,
        the case is considered passed, returning 1.0; otherwise returns 0.0.

        Args:
            post: Case post data

        Returns:
            any_right_accuracy (0.0 or 1.0), or None if unable to calculate
        """
        accuracy_pred = post.get("accuracy_pred")
        if not isinstance(accuracy_pred, list) or not accuracy_pred:
            raise ValueError("No accuracy_pred found.")

        has_pass = False
        for pred_obj in accuracy_pred:
            if not isinstance(pred_obj, dict):
                raise ValueError(f"Invalid accuracy_pred object: {pred_obj}")
            judge = pred_obj.get("judge")
            if not isinstance(judge, dict):
                raise ValueError(f"Invalid judge object: {judge}")
            result = judge.get("result")
            if not isinstance(result, str):
                raise ValueError(f"Invalid judge result: {result}")
            if result.strip().upper() == "PASS":
                has_pass = True

        return 1.0 if has_pass else 0.0

    @staticmethod
    def _calc_case_conflict_probe_pass(post: Dict[str, Any]) -> Optional[float]:
        """
        Calculate case conflict_probe pass rate

        Conflict Probe evaluation: Detects whether the model hallucinates on conflicting knowledge.

        Args:
            post: Case post data

        Returns:
            conflict_probe pass rate (0.0-1.0), or None if unable to calculate
        """
        if not isinstance(post, dict):
            return None
        conflict_probe = post.get("conflict_probe")
        if not isinstance(conflict_probe, list) or not conflict_probe:
            return None
        total = 0
        passed = 0
        for pred_obj in conflict_probe:
            if not isinstance(pred_obj, dict):
                raise ValueError(f"Invalid conflict_probe object: {pred_obj}")
            judge = pred_obj.get("judge")
            if not isinstance(judge, dict):
                raise ValueError(f"Invalid judge object: {judge} {pred_obj}")
            result = judge.get("result")
            if not isinstance(result, str):
                raise ValueError(f"Invalid judge result: {result}")
            total += 1
            if result.strip().upper() == "PASS":
                passed += 1
        return (passed / total) if total else None

    def calculate_aggregates(self, all_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate aggregate metrics for all cases

        Args:
            all_metrics: List of metrics for all cases

        Returns:
            Aggregate metrics dictionary, containing:
            - total_cases: Total number of cases
            - overall_hop_wise_average: hop_wise average pass rate
            - accuracy_averages.qa: accuracy average pass rate
            - accuracy_averages.any_right_accuracy: case-level "any correct means pass" average pass rate
            - conflict_probe_average: conflict_probe average pass rate
        """
        total_cases = 0
        hop_case_avgs: List[float] = []
        acc_case_avgs: List[float] = []
        any_right_acc_case_avgs: List[float] = []
        cp_case_avgs: List[float] = []

        for metrics in all_metrics:
            post = metrics.get("post")
            total_cases += 1

            hop = self._calc_case_hop_wise_average(post)
            if hop is not None:
                hop_case_avgs.append(hop)

            acc = self._calc_case_accuracy_pass(post)
            if acc is not None:
                acc_case_avgs.append(acc)

            any_right_acc = self._calc_case_any_right_accuracy(post)
            if any_right_acc is not None:
                any_right_acc_case_avgs.append(any_right_acc)

            if not self.should_exclude_source("conflict_probe"):
                cp = self._calc_case_conflict_probe_pass(post)
                if cp is not None:
                    cp_case_avgs.append(cp)

        return {
            "total_cases": total_cases,
            "overall_hop_wise_average": (sum(hop_case_avgs) / len(hop_case_avgs)) if hop_case_avgs else 0.0,
            "accuracy_averages": {
                "qa": (sum(acc_case_avgs) / len(acc_case_avgs)) if acc_case_avgs else 0.0,
                "any_right_accuracy": (sum(any_right_acc_case_avgs) / len(any_right_acc_case_avgs)) if any_right_acc_case_avgs else 0.0,
            },
            "conflict_probe_average": (sum(cp_case_avgs) / len(cp_case_avgs)) if cp_case_avgs else None,
        }

    def run(
        self,
        *,
        input_metrics_file: str,
        output_metrics_file: str,
        summary_file: str,
        checkpoint_every: int = 50,
        checkpoint_seconds: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Run judgment pipeline

        Args:
            input_metrics_file: Input metrics JSON path
            output_metrics_file: Output judged metrics JSON path
            summary_file: Output summary statistics JSON path
            checkpoint_every: Save checkpoint every N items processed
            checkpoint_seconds: Save checkpoint every N seconds

        Returns:
            Result dictionary containing counters and aggregate_metrics
        """
        all_metrics = JsonIO.load_json(input_metrics_file)
        if not isinstance(all_metrics, list):
            raise ValueError("metrics JSON must be list[case].")

        merged_from_output = 0
        if os.path.exists(output_metrics_file):
            print(f"Reuse judged results from: {output_metrics_file}")
            output_metrics = JsonIO.load_json(output_metrics_file)
            if not isinstance(output_metrics, list):
                raise ValueError("output metrics JSON must be list[case].")
            judged_map = self.collect_judged_map(output_metrics)
            merged_from_output = self.merge_judged_results(all_metrics, judged_map)
            print(f"Merged judged items from existing output: {merged_from_output}")

        tasks = []
        counters = {
            "cases_total": 0,
            "items_total": 0,
            "items_skipped": 0,
            "items_judged": 0,
            "items_errors": 0,
            "pass": 0,
            "fail": 0,
            "items_merged_from_output": merged_from_output,
        }

        for case_idx, metrics in enumerate(all_metrics):
            if not isinstance(metrics, dict):
                continue
            post = metrics.get("post")
            if not isinstance(post, dict):
                continue
            counters["cases_total"] += 1
            for pred_path, pred_obj in DataTraversal.iter_prompt_pred_objs_with_path(post, path=f"case[{case_idx}].post"):
                source = DataTraversal.extract_source_from_path(pred_path)
                if self.should_exclude_source(source):
                    continue
                counters["items_total"] += 1
                if self._is_judged(pred_obj):
                    counters["items_skipped"] += 1
                    result = pred_obj.get("judge", {}).get("result", "")
                    norm = result.strip().upper() if isinstance(result, str) else ""
                    if norm == "PASS":
                        counters["pass"] += 1
                    elif norm == "FAIL":
                        counters["fail"] += 1
                    continue
                tasks.append((case_idx, pred_obj))

        start = time.time()
        last_ckpt = time.time()
        print(f"Start judge: pending={len(tasks)}, skipped={counters['items_skipped']}, workers={self.config.max_workers}, keys={len(self.config.api_keys)}")

        def maybe_checkpoint(completed: int, force: bool = False) -> None:
            nonlocal last_ckpt
            now = time.time()
            if force:
                JsonIO.atomic_dump_json(all_metrics, output_metrics_file, indent=2)
                last_ckpt = now
                return
            if checkpoint_every > 0 and completed % checkpoint_every == 0:
                JsonIO.atomic_dump_json(all_metrics, output_metrics_file, indent=2)
                last_ckpt = now
                return
            if checkpoint_seconds > 0 and (now - last_ckpt) >= checkpoint_seconds:
                JsonIO.atomic_dump_json(all_metrics, output_metrics_file, indent=2)
                last_ckpt = now

        def job(pred_obj: Dict[str, Any]) -> Dict[str, Any]:
            prompt = str(pred_obj.get("prompt") or "")
            pred = str(pred_obj.get("pred") or "")
            refs = pred_obj.get("ref") or []
            if not isinstance(refs, list):
                refs = [str(refs)]
            return self.engine.judge_final_answer_vs_ref(prompt=prompt, pred=pred, refs=refs)

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as ex:
            future_to_task = {ex.submit(job, pred_obj): (case_idx, pred_obj) for case_idx, pred_obj in tasks}
            completed = 0
            for fut in as_completed(future_to_task):
                _, pred_obj = future_to_task[fut]
                try:
                    judge_result = fut.result()
                    pred_obj["judge"] = judge_result
                    counters["items_judged"] += 1
                    if judge_result["result"] == "PASS":
                        counters["pass"] += 1
                    else:
                        counters["fail"] += 1
                except Exception as e:
                    pred_obj["judge"] = {"result": "ERROR", "raw": "", "error": repr(e)}
                    counters["items_errors"] += 1

                completed += 1
                elapsed = time.time() - start
                rate = completed / elapsed if elapsed > 0 else 0
                pct = (completed / len(tasks) * 100) if tasks else 100.0
                print(
                    f"\rJudge: {completed}/{len(tasks)} ({pct:.1f}%), speed={rate:.2f} items/s, elapsed={elapsed:.1f}s",
                    end="",
                )
                maybe_checkpoint(completed)

            print()

        maybe_checkpoint(len(tasks), force=True)
        counters["key_usage"] = self.engine.pool.stats()

        aggregates = self.calculate_aggregates(all_metrics)
        JsonIO.atomic_dump_json(all_metrics, output_metrics_file, indent=2)
        JsonIO.atomic_dump_json(aggregates, summary_file, indent=2)

        elapsed = time.time() - start
        print(f"Judge done, elapsed={elapsed:.1f}s")
        print(f"Output metrics: {output_metrics_file}")
        print(f"Summary: {summary_file}")
        print(
            json.dumps(
                {"counters": counters, "aggregate_metrics": aggregates},
                ensure_ascii=False,
                indent=2,
            )
        )
        return {"counters": counters, "aggregate_metrics": aggregates}


class AnalyzePipeline:
    """
    Prediction analysis pipeline

    Performs a complete analysis workflow on metrics JSON, including:
    1. Collecting basic pred information
    2. LLM self-refutation judgment: Optional, detects self-contradictions in model outputs
    3. Rewrite why consistency judgment: Optional, detects whether predictions for why-type questions are consistent with reference articles
    4. Report generation: Outputs TXT report and JSON results

    Main features:
    - analyze_items: Collect basic information for all preds
    - maybe_run_llm_self_refute: Run LLM self-refutation judgment (with checkpoint resumption)
    - write_report: Generate analysis report
    - run: Execute complete analysis workflow

    Output files:
    - pred_analysis.txt: TXT report
    - pred_analysis.json: Full pred detail
    - pred_analysis_llm_judge_ckpt.jsonl: LLM judgment checkpoint
    """

    EXCLUDED_SOURCES = {
        "accuracy",
        "accuracy_pred",
        "causal_enhanced_hop_wise_pred",
        "causal_enhanced_conflict_probe",
    }

    def __init__(self, judge_config: JudgeConfig):
        """
        Initialize analysis pipeline

        Args:
            judge_config: LLM judgment configuration (used for self-refutation detection)
        """
        self.engine = LLMJudgeEngine(judge_config)

    @staticmethod
    def make_pred_uid(item: Dict[str, Any]) -> str:
        """
        Generate unique identifier for a pred

        Uses SHA1 hash of case_id + pred_path + pred content,
        used to match already judged results during checkpoint resumption.

        Args:
            item: Analysis result item

        Returns:
            SHA1 hash string (40 characters)
        """
        key = f"{item.get('case_id', '')}|{item.get('pred_path', '')}|{item.get('pred', '')}"
        return hashlib.sha1(key.encode("utf-8")).hexdigest()

    @staticmethod
    def load_llm_checkpoint(path: str) -> Dict[str, Dict[str, Any]]:
        """
        Load LLM judgment checkpoint file

        Checkpoint file is in JSONL format, each line is a JSON object:
        {"uid": "sha1_hash", "result": {...}}

        Args:
            path: Checkpoint file path

        Returns:
            Dictionary where key is uid and value is judgment result
        """
        if not path or not os.path.exists(path):
            return {}
        by_uid: Dict[str, Dict[str, Any]] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Warning: skip invalid checkpoint line {line_no}: {path}")
                    continue
                uid = obj.get("uid")
                result = obj.get("result")
                if isinstance(uid, str) and isinstance(result, dict):
                    by_uid[uid] = result
        return by_uid

    @staticmethod
    def append_llm_checkpoint(uid: str, result: Dict[str, Any], fp: Any, item: Dict[str, Any] | None = None) -> None:
        """
        Append judgment result to checkpoint file

        Args:
            uid: Pred unique identifier
            result: Judgment result
            fp: File handle (opened in append mode)
            item: Original data item containing prompt/pred/reference context (optional)
        """
        record: Dict[str, Any] = {"uid": uid, "result": result}
        if item is not None:
            record["context"] = {
                "case_id": item.get("case_id"),
                "pred_path": item.get("pred_path"),
                "prompt": item.get("prompt"),
                "pred": item.get("pred"),
                "reference": item.get("reference"),
            }
        fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        fp.flush()

    @staticmethod
    def is_causaledit_input_path(input_path: str) -> bool:
        """
        Determine whether the input path is from a CausalEdit run directory.
        """
        norm_path = os.path.abspath(input_path).replace("\\", "/").lower()
        return "/causaledit/" in norm_path

    @staticmethod
    def is_ce_only_input_path(input_path: str) -> bool:
        """
        Determine whether the input path is from an extract-ce generated CE-only run.
        """
        abs_input = os.path.abspath(input_path)
        input_dir = os.path.dirname(abs_input)
        run_dir_name = os.path.basename(input_dir.rstrip(os.sep))
        return run_dir_name.endswith("_only_causal_enhanced")

    @staticmethod
    def derive_paired_non_ce_metrics_path(input_path: str) -> Optional[str]:
        """
        Derive the corresponding original non-CE metrics path from a CE-only metrics path.
        """
        if not AnalyzePipeline.is_ce_only_input_path(input_path):
            return None
        abs_input = os.path.abspath(input_path)
        input_dir = os.path.dirname(abs_input)
        run_dir_name = os.path.basename(input_dir.rstrip(os.sep))
        suffix = "_only_causal_enhanced"
        if not run_dir_name.endswith(suffix):
            return None
        base_run_dir = run_dir_name[: -len(suffix)]
        parent_dir = os.path.dirname(input_dir)
        return os.path.join(parent_dir, base_run_dir, "metrics.json")

    @staticmethod
    def is_why_question(text: Any) -> bool:
        """
        Determine whether the text is a why-type question.
        """
        norm = str(text or "").strip().lower()
        if not norm:
            return False
        return "Why?".lower() in norm

    @staticmethod
    def _get_requested_rewrite(item: Dict[str, Any], rewrite_index: Any) -> Optional[Dict[str, Any]]:
        requested_rewrite = item.get("requested_rewrite")
        if not isinstance(requested_rewrite, list):
            return None
        if not isinstance(rewrite_index, int):
            return None
        if not (0 <= rewrite_index < len(requested_rewrite)):
            return None
        req = requested_rewrite[rewrite_index]
        return req if isinstance(req, dict) else None

    @staticmethod
    def should_exclude_source(source: str) -> bool:
        """
        Determine whether a source type should be excluded from analysis results.

        Args:
            source: Source type parsed from pred_path

        Returns:
            True means skip this pred
        """
        return source in AnalyzePipeline.EXCLUDED_SOURCES

    @staticmethod
    def _normalize_rewrite_text(text: Any) -> str:
        if text is None:
            return ""
        s = str(text).lower()
        s = re.sub(r"\s+", " ", s).strip()
        s = re.sub(r"[^\w\s]", "", s)
        return s

    @staticmethod
    def _extract_rewrite_targets(
        requested_rewrite: List[Dict[str, Any]],
    ) -> List[str]:
        targets: List[str] = []
        for req in requested_rewrite:
            t = req.get("target_new")
            if isinstance(t, dict):
                t = t.get("str")
            targets.append(AnalyzePipeline._normalize_rewrite_text(t))
        return targets

    @staticmethod
    def _extract_index_from_path(pred_path: str, source: str) -> Optional[int]:
        m = re.search(rf"\.{re.escape(source)}\[(\d+)\]", pred_path or "")
        if not m:
            return None
        return int(m.group(1))

    @staticmethod
    def _match_rewrite_by_ref(ref: Any, rewrite_targets: List[str]) -> Optional[int]:
        if not isinstance(ref, list):
            return None
        candidates = set()
        for ridx, t_norm in enumerate(rewrite_targets):
            if not t_norm:
                continue
            for r in ref:
                r_norm = AnalyzePipeline._normalize_rewrite_text(r)
                if not r_norm:
                    continue
                if t_norm == r_norm:
                    candidates.add(ridx)
        if len(candidates) == 1:
            return next(iter(candidates))
        if len(candidates) > 1:
            raise ValueError(f"Ambiguous rewrite mapping by ref: candidates={candidates}")
        return None

    @staticmethod
    def _match_rewrite_by_subject(prompt: Any, requested_rewrite: List[Dict[str, Any]]) -> Optional[int]:
        p_norm = AnalyzePipeline._normalize_rewrite_text(prompt)
        if not p_norm:
            return None
        candidates = []
        for ridx, req in enumerate(requested_rewrite):
            s_norm = AnalyzePipeline._normalize_rewrite_text(req.get("subject"))
            if s_norm and s_norm in p_norm:
                candidates.append(ridx)
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise ValueError(f"Ambiguous rewrite mapping by subject: candidates={candidates}")
        return None

    @classmethod
    def resolve_rewrite_index(
        cls,
        *,
        item: Dict[str, Any],
        pred_path: str,
        source: str,
        prompt: Any,
        ref: Any,
    ) -> Optional[int]:
        requested_rewrite = item.get("requested_rewrite")
        if not isinstance(requested_rewrite, list) or not requested_rewrite:
            raise ValueError(f"requested_rewrite is empty, cannot map rewrite. case_id={item.get('case_id')}, pred_path={pred_path}")
        rewrite_targets = cls._extract_rewrite_targets(requested_rewrite)

        if source == "conflict_probe":
            pred_idx = cls._extract_index_from_path(pred_path, source)
            post = item.get("post") if isinstance(item.get("post"), dict) else {}
            cp_list = post.get("conflict_probe") if isinstance(post, dict) else None
            if (
                pred_idx is not None
                and isinstance(cp_list, list)
                and len(requested_rewrite) > 0
                and len(cp_list) > 0
                and len(cp_list) % len(requested_rewrite) == 0
            ):
                per_rewrite = len(cp_list) // len(requested_rewrite)
                mapped = pred_idx // per_rewrite
                if 0 <= mapped < len(requested_rewrite):
                    return mapped

            raise ValueError(f"Unable to map conflict_probe pred to rewrite. case_id={item.get('case_id')}, pred_path={pred_path}")

        if source == "hop_wise_pred":
            by_ref = cls._match_rewrite_by_ref(ref, rewrite_targets)
            if by_ref is not None:
                return by_ref
            by_subject = cls._match_rewrite_by_subject(prompt, requested_rewrite)
            if by_subject is not None:
                return by_subject
            # In MQuAKE data, some hop_wise_pred cannot be mapped to a specific rewrite; skip directly.
            return None

        raise ValueError(f"Unsupported source for rewrite-level stats: source={source}, case_id={item.get('case_id')}, pred_path={pred_path}")

    @staticmethod
    def classify_conflict_probe_variant(source: str, prompt: Any) -> Optional[str]:
        """
        Identify conflict_probe subtype based on prompt text.
        """
        if source != "conflict_probe":
            return None
        prompt_text = str(prompt or "")
        prompt_lower = prompt_text.lower()
        if "tell me about" not in prompt_lower:
            return None
        if "tell me about" in prompt_lower and "first, and then answer" in prompt_lower:
            return "tell_me_about_first"
        if "answer first, and then tell me about" in prompt_lower:
            return "answer_first"
        return None

    def analyze_items(self, data: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze all preds, collecting basic information

        For each pred, collects the following information:
        - case_id, pred_path, source, question, prompt, reference, pred
        - rewrite_index, rewrite_key, rewrite_subject

        Args:
            data: Case data list

        Returns:
            Analysis result list, each element contains case_id, pred_path, basic info, etc.
        """
        results: List[Dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            case_id = item.get("case_id")
            for pred_path, pred, q, prompt, ref in DataTraversal.collect_preds(item, path=f"case[{case_id}]"):
                source = DataTraversal.extract_source_from_path(pred_path)
                if self.should_exclude_source(source):
                    continue
                rewrite_index = self.resolve_rewrite_index(
                    item=item,
                    pred_path=pred_path,
                    source=source,
                    prompt=prompt,
                    ref=ref,
                )
                if rewrite_index is None:
                    continue
                requested_rewrite = item.get("requested_rewrite") or []
                rewrite_subject = ""
                if isinstance(requested_rewrite, list) and 0 <= rewrite_index < len(requested_rewrite):
                    rewrite_subject = str((requested_rewrite[rewrite_index] or {}).get("subject") or "")
                results.append(
                    {
                        "case_id": f"{case_id}",
                        "rewrite_index": rewrite_index,
                        "rewrite_key": f"{case_id}|rw{rewrite_index}",
                        "rewrite_subject": rewrite_subject,
                        "pred_path": pred_path,
                        "source": source,
                        "question": q,
                        "prompt": prompt,
                        "reference": ref,
                        "pred": pred,
                        "conflict_probe_variant": (self.classify_conflict_probe_variant(source, prompt)),
                    }
                )
        return results

    def run_llm_task_with_checkpoint(
        self,
        items: List[Dict[str, Any]],
        *,
        checkpoint_path: str,
        resume: bool,
        task_name: str,
        job_fn: Any,
    ) -> bool:
        """
        Execute a checkpointed LLM task on a set of items.

        `job_fn` receives an item and returns a result dictionary that can be directly merged into the item.
        """
        total = len(items)
        if total == 0:
            return True

        uids = [self.make_pred_uid(item) for item in items]
        pending_indices: List[int] = list(range(total))
        if resume and checkpoint_path:
            recovered_by_uid = self.load_llm_checkpoint(checkpoint_path)
            pending_indices = []
            recovered = 0
            for i, uid in enumerate(uids):
                recovered_result = recovered_by_uid.get(uid)
                if recovered_result is None:
                    pending_indices.append(i)
                    continue
                items[i].update(recovered_result)
                recovered += 1
            print(f"{task_name} checkpoint resume: loaded {recovered}/{total}, pending={len(pending_indices)}")

        if not pending_indices:
            print(f"{task_name}: all items loaded from checkpoint.")
            return True

        JsonIO.ensure_parent_dir(checkpoint_path)
        ckpt_fp = open(checkpoint_path, "a", encoding="utf-8")
        failed_items: List[Tuple[int, str]] = []

        def job(idx: int) -> Tuple[int, Dict[str, Any]]:
            item = items[idx]
            result = job_fn(item)
            return idx, result

        try:
            with ThreadPoolExecutor(max_workers=self.engine.config.max_workers) as ex:
                future_to_idx = {ex.submit(job, i): i for i in pending_indices}
                completed = total - len(pending_indices)
                for fut in as_completed(future_to_idx):
                    idx = future_to_idx[fut]
                    try:
                        idx2, result = fut.result()
                        if idx2 != idx:
                            raise RuntimeError(f"index mismatch expected={idx}, got={idx2}")
                        items[idx].update(result)
                        self.append_llm_checkpoint(uids[idx], result, ckpt_fp, items[idx])
                    except Exception as e:
                        print(f"{task_name}: error idx={idx}, err={repr(e)}")
                        failed_items.append((idx, repr(e)))
                    completed += 1
                    pct = (completed / total * 100) if total else 100.0
                    print(f"\r{task_name}: {completed}/{total} ({pct:.1f}%)", end="")
            print()
        finally:
            ckpt_fp.close()

        if failed_items:
            detail = "; ".join([f"idx={idx},err={err}" for idx, err in failed_items[:10]])
            raise RuntimeError(f"{task_name} failed items={len(failed_items)}. {detail}")
        return True

    def collect_rewrite_why_candidates(
        self,
        results: List[Dict[str, Any]],
        data: Iterable[Dict[str, Any]],
        *,
        input_path: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Extract candidates for article-grounded why judgment from analysis results.
        """
        if not self.is_causaledit_input_path(input_path):
            return [], {
                "enabled": False,
                "skipped": True,
                "skip_reason": "non_causaledit_input",
                "candidate_count": 0,
                "scan_count": 0,
                "skip_breakdown": {},
            }

        case_by_id: Dict[str, Dict[str, Any]] = {}
        for item in data:
            if isinstance(item, dict):
                case_by_id[str(item.get("case_id"))] = item

        candidates: List[Dict[str, Any]] = []
        skip_counts: Counter = Counter()
        scan_count = 0
        for r in results:
            if str(r.get("source") or "") != "hop_wise_pred":
                continue
            scan_count += 1

            prompt = str(r.get("prompt"))
            if not self.is_why_question(prompt):
                skip_counts["not_why_question"] += 1
                continue

            case_id = str(r.get("case_id") or "")
            item = case_by_id.get(case_id)

            rewrite = self._get_requested_rewrite(item, r.get("rewrite_index"))
            if rewrite is None:
                skip_counts["missing_rewrite_mapping"] += 1
                continue

            article = str(rewrite.get("article") or "").strip()
            if not article:
                skip_counts["missing_article"] += 1
                continue

            pred = str(r.get("pred") or "").strip()
            if not pred:
                skip_counts["empty_pred"] += 1
                continue

            candidates.append(
                {
                    "case_id": case_id,
                    "rewrite_index": r.get("rewrite_index"),
                    "rewrite_key": r.get("rewrite_key"),
                    "rewrite_subject": str(r.get("rewrite_subject")),
                    "pred_path": str(r.get("pred_path")),
                    "source": str(r.get("source")),
                    "prompt": prompt,
                    "pred": pred,
                    "article_length": len(article),
                    "article": article,
                }
            )

        return candidates, {
            "enabled": True,
            "skipped": False,
            "skip_reason": "",
            "candidate_count": len(candidates),
            "scan_count": scan_count,
            "skip_breakdown": dict(skip_counts),
        }

    def maybe_run_llm_self_refute(self, results: List[Dict[str, Any]], *, checkpoint_path: str, resume: bool) -> bool:
        """
        Run LLM self-refutation judgment (with checkpoint resumption)

        Calls LLM for all preds to determine whether self-contradictions exist.
        Supports resuming progress from checkpoint file to avoid duplicate calls.

        Args:
            results: Analysis result list (will be updated in-place)
            checkpoint_path: Checkpoint file path
            resume: Whether to resume from checkpoint

        Returns:
            Whether completed successfully

        Raises:
            RuntimeError: Some judgments failed
        """

        def job(item: Dict[str, Any]) -> Dict[str, Any]:
            return self.engine.judge_self_refute(
                prompt=str(item.get("prompt") or ""),
                pred=str(item.get("pred") or ""),
                reference=item.get("reference"),
            )

        return self.run_llm_task_with_checkpoint(
            results,
            checkpoint_path=checkpoint_path,
            resume=resume,
            task_name="LLM self-refute judge",
            job_fn=job,
        )

    def maybe_run_llm_rewrite_why(
        self,
        candidates: List[Dict[str, Any]],
        *,
        checkpoint_path: str,
        resume: bool,
    ) -> bool:
        """
        Execute article-grounded consistency judgment on rewrite why candidates.
        """

        def job(item: Dict[str, Any]) -> Dict[str, Any]:
            return self.engine.judge_article_support_consistency(
                article=str(item.get("article")),
                question=str(item.get("prompt")),
                pred=str(item.get("pred")),
            )

        return self.run_llm_task_with_checkpoint(
            candidates,
            checkpoint_path=checkpoint_path,
            resume=resume,
            task_name="LLM rewrite-why consistency judge",
            job_fn=job,
        )

    @staticmethod
    def build_rewrite_why_summary(
        *,
        input_path: str,
        extraction_info: Dict[str, Any],
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Summarize rewrite why consistency judgment results.
        """
        summary: Dict[str, Any] = {
            "input_path": os.path.abspath(input_path),
            "enabled": bool(extraction_info.get("enabled")),
            "skipped": bool(extraction_info.get("skipped")),
            "skip_reason": extraction_info.get("skip_reason", ""),
            "scan_count": int(extraction_info.get("scan_count", 0) or 0),
            "candidate_count": len(candidates),
            "skip_breakdown": extraction_info.get("skip_breakdown", {}),
        }
        if not candidates:
            summary.update(
                {
                    "judged_count": 0,
                    "pass_count": 0,
                    "fail_count": 0,
                    "unsure_count": 0,
                    "rewrite_count": 0,
                    "mean_rewrite_pass_rate": 0.0,
                }
            )
            return summary

        judged = [x for x in candidates if str(x.get("article_consistency_result") or "").upper() in {"PASS", "FAIL", "UNSURE"}]
        pred_result_counter: Counter = Counter(str(x.get("article_consistency_result") or "").upper() for x in judged)

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for item in judged:
            rewrite_key = str(item.get("rewrite_key") or "")
            if not rewrite_key:
                continue
            grouped.setdefault(rewrite_key, []).append(item)

        # Global average pass rate
        rewrite_pass_rates: Dict[str, float] = {}
        for rewrite_key, items in grouped.items():
            results = [str(x.get("article_consistency_result") or "").upper() for x in items]
            pass_count = sum(1 for r in results if r == "PASS")
            total = len(results)
            rewrite_pass_rates[rewrite_key] = pass_count / total if total > 0 else 0.0

        mean_rewrite_pass_rate = sum(rewrite_pass_rates.values()) / len(rewrite_pass_rates) if rewrite_pass_rates else 0.0

        rewrite_count = len(rewrite_pass_rates)
        summary.update(
            {
                "judged_count": len(judged),
                "pass_count": pred_result_counter.get("PASS", 0),
                "fail_count": pred_result_counter.get("FAIL", 0),
                "unsure_count": pred_result_counter.get("UNSURE", 0),
                "rewrite_count": rewrite_count,
                "mean_rewrite_pass_rate": mean_rewrite_pass_rate,
            }
        )
        return summary

    @staticmethod
    def build_self_refute_rewrite_flags(
        results: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate pred-level self-refutation results into rewrite-level flags.
        """
        judged = [r for r in results if r.get("llm_self_refute") is not None]
        rewrite_by_source: Dict[str, Dict[str, List[Dict[str, bool]]]] = {}
        for r in judged:
            rewrite_key = str(r.get("rewrite_key") or "")
            if not rewrite_key:
                continue
            source = str(r.get("source") or "unknown")
            is_self_refute = r.get("llm_self_refute") is True
            conflict_probe_variant = str(r.get("conflict_probe_variant") or "")
            analysis = r.get("llm_self_refute_analysis") or {}
            retrieved_target_new = isinstance(analysis, dict) and analysis.get("assertion_of_target_new") is True
            rewrite_by_source.setdefault(rewrite_key, {}).setdefault(source, []).append(
                {
                    "self_refute": is_self_refute,
                    "retrieved_target_new": retrieved_target_new,
                    "conflict_probe_variant": conflict_probe_variant,
                }
            )

        rewrite_flags: Dict[str, Dict[str, Any]] = {}
        for rewrite_key, sources in rewrite_by_source.items():
            cp_list = sources.get("conflict_probe", [])
            cp_tell_first_list = [x for x in cp_list if x.get("conflict_probe_variant") == "tell_me_about_first"]
            hw_list = sources.get("hop_wise_pred", [])
            all_values: List[Dict[str, bool]] = []
            for vals in sources.values():
                all_values.extend(vals)
            rewrite_flags[rewrite_key] = {
                "self_refute": {
                    "conflict_probe": any(x["self_refute"] for x in cp_list),
                    "conflict_probe_tell_me_about_first": any(x["self_refute"] for x in cp_tell_first_list),
                    "hop_wise_pred": any(x["self_refute"] for x in hw_list),
                    "combined": any(x["self_refute"] for x in all_values),
                    "combined_tell_me_about_first": any(x["self_refute"] for x in cp_tell_first_list) or any(x["self_refute"] for x in hw_list),
                },
                "retrieved_target_new_in_hop": any(x["retrieved_target_new"] for x in hw_list),
            }
        return rewrite_flags

    @staticmethod
    def collect_hop_retrieved_rewrite_keys(
        rewrite_flags: Dict[str, Dict[str, Any]],
    ) -> Set[str]:
        """
        Collect the set of rewrites with successful assertion_of_target_new in hop_wise_pred.
        """
        return {rewrite_key for rewrite_key, item in rewrite_flags.items() if item.get("retrieved_target_new_in_hop") is True}

    @staticmethod
    def filter_results_by_rewrite_keys(results: List[Dict[str, Any]], rewrite_keys: Set[str]) -> List[Dict[str, Any]]:
        """
        Keep only result items matching the specified rewrite_key set.
        """
        return [item for item in results if str(item.get("rewrite_key") or "") in rewrite_keys]

    @staticmethod
    def build_self_refute_summary_from_flags(
        rewrite_flags: Dict[str, Dict[str, Any]],
        *,
        denominator_rewrite_keys: Optional[Set[str]] = None,
        denominator_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build self-refutation statistics summary based on rewrite-level flags.
        """
        source_names = SELF_REFUTE_SOURCE_ORDER
        source_totals = {source: len(rewrite_flags) for source in source_names}
        source_yes = {source: sum(1 for item in rewrite_flags.values() if bool((item.get("self_refute") or {}).get(source))) for source in source_names}

        conditional_stats = None
        if denominator_rewrite_keys is not None:
            denominator_keys = set(denominator_rewrite_keys)
            conditional_stats = {
                "label": denominator_label or "",
                "denominator_total": len(denominator_keys),
                "yes_by_source": {
                    source: sum(1 for rewrite_key in denominator_keys if bool((rewrite_flags.get(rewrite_key, {}).get("self_refute") or {}).get(source)))
                    for source in source_names
                },
            }

        return {
            "judged_rewrite_count": len(rewrite_flags),
            "source_totals": source_totals,
            "source_yes": source_yes,
            "conditional": conditional_stats,
        }

    @staticmethod
    def write_self_refute_summary_block(
        fp: Any,
        summary: Dict[str, Any],
        *,
        title: str = "LLM self-refutation judge stats",
    ) -> None:
        """
        Write self-refutation statistics summary to report.
        """
        fp.write(f"{title}\n")
        fp.write(f"  Judged rewrite count: {int(summary.get('judged_rewrite_count', 0))}\n\n")
        source_totals = summary.get("source_totals") or {}
        source_yes = summary.get("source_yes") or {}
        conditional = summary.get("conditional")
        conditional_ratios_by_source: Dict[str, str] = {}

        for source in SELF_REFUTE_REPORT_SOURCE_ORDER:
            total = int(source_totals.get(source, 0))
            yes = int(source_yes.get(source, 0))
            label = SELF_REFUTE_SOURCE_LABELS[source]
            if total > 0:
                fp.write(f"  {label}: {yes} / {total} ({yes / total * 100:.1f}%)\n")
            else:
                fp.write(f"  {label}: 0 / 0 (N/A)\n")

            if not isinstance(conditional, dict):
                continue
            denominator_total = int(conditional.get("denominator_total", 0))
            denominator_label = str(conditional.get("label") or "").strip()
            conditional_yes = int((conditional.get("yes_by_source") or {}).get(source, 0))
            prefix = f"  {label}, among {denominator_label}"
            if denominator_total > 0:
                ratio = conditional_yes / denominator_total * 100
                conditional_ratios_by_source[source] = f"{ratio:.1f}%"
                fp.write(f"{prefix}: {conditional_yes} / {denominator_total} ({ratio:.1f}%)\n")
            else:
                conditional_ratios_by_source[source] = "N/A"
                fp.write(f"{prefix}: 0 / 0 (N/A)\n")
        if isinstance(conditional, dict):
            compact_order = (
                "hop_wise_pred",
                "conflict_probe",
                "conflict_probe_tell_me_about_first",
                "combined",
            )
            fp.write(
                "  Conditional ratios "
                "(hop_wise_pred/conflict_probe/conflict_probe_tell_me_about_first/"
                "combined): "
                f"{'/'.join(conditional_ratios_by_source.get(source, 'N/A') for source in compact_order)}\n"
            )
        fp.write("\n")

    @staticmethod
    def write_ce_paired_comparison_block(fp: Any, comparison: Dict[str, Any]) -> None:
        """
        Write CE-only vs paired non-CE run comparison statistics to report.
        """
        fp.write("Cross-run CE comparison\n")
        fp.write("  Denominator source: rewrites with hop_wise_pred assertion_of_target_new=true in paired non-CE run\n")
        fp.write(f"  Paired non-CE run: {comparison.get('paired_non_ce_input_path', '')}\n")
        denominator_total = int(comparison.get("denominator_total", 0))
        fp.write(f"  Baseline successful retrieval rewrite count: {denominator_total}\n")

        original_summary = comparison.get("original_summary") or {}
        ce_summary = comparison.get("ce_summary") or {}
        for source in SELF_REFUTE_SOURCE_ORDER:
            original_yes = int((((original_summary.get("conditional") or {}).get("yes_by_source")) or {}).get(source, 0))
            ce_yes = int((((ce_summary.get("conditional") or {}).get("yes_by_source")) or {}).get(source, 0))
            label = CE_PAIRED_SOURCE_LABELS[source]
            if denominator_total > 0:
                original_rate = original_yes / denominator_total
                ce_rate = ce_yes / denominator_total
                abs_drop = original_rate - ce_rate
                rel_drop_text = f"{(abs_drop / original_rate * 100):.1f}%" if original_rate > 0 else "N/A"
                fp.write(f"  Original {label} self-refute: {original_yes} / {denominator_total} ({original_rate * 100:.1f}%)\n")
                fp.write(f"  CE {label} self-refute: {ce_yes} / {denominator_total} ({ce_rate * 100:.1f}%)\n")
                fp.write(
                    f"  {label} self-refute drop: {original_yes - ce_yes} / {denominator_total} ({abs_drop * 100:.1f} pts, relative drop {rel_drop_text})\n\n\n"
                )
            else:
                fp.write(f"  Original {label} self-refute: 0 / 0 (N/A)\n")
                fp.write(f"  CE {label} self-refute: 0 / 0 (N/A)\n")
                fp.write(f"  {label} self-refute drop: 0 / 0 (N/A)\n")
        transition_summary = comparison.get("transition_summary") or {}
        by_source = transition_summary.get("by_source") or {}
        if isinstance(by_source, dict) and by_source:
            fp.write("  Paired transition table + McNemar\n")
            for source in SELF_REFUTE_SOURCE_ORDER:
                item = by_source.get(source) or {}
                label = CE_PAIRED_SOURCE_LABELS[source]
                n00 = int(item.get("n00", 0))
                n01 = int(item.get("n01", 0))
                n10 = int(item.get("n10", 0))
                n11 = int(item.get("n11", 0))
                original_sr_count = n10 + n11
                ce_sr_count = n01 + n11
                repair_denom = original_sr_count
                harm_denom = n00 + n01
                mcnemar = item.get("mcnemar") or {}
                p_value = mcnemar.get("p_value")

                fp.write(f"  {label} transition counts: ")
                fp.write(f"N00={n00}, N01={n01}, N10={n10}, N11={n11}\n")
                fp.write(f"    Original SR rate: {original_sr_count} / {denominator_total} ({float(item.get('original_sr_rate', 0.0)) * 100:.1f}%)\n")
                fp.write(f"    CE SR rate: {ce_sr_count} / {denominator_total} ({float(item.get('ce_sr_rate', 0.0)) * 100:.1f}%)\n")
                fp.write(f"    Net SR drop: {n10 - n01} / {denominator_total} ({float(item.get('net_sr_drop', 0.0)) * 100:.1f} pts)\n")
                if repair_denom > 0:
                    fp.write(f"    Repair rate: {n10} / {repair_denom} ({float(item.get('repair_rate', 0.0)) * 100:.1f}%)\n")
                else:
                    fp.write("    Repair rate: 0 / 0 (N/A)\n")
                if harm_denom > 0:
                    fp.write(f"    Harm rate: {n01} / {harm_denom} ({float(item.get('harm_rate', 0.0)) * 100:.1f}%)\n")
                else:
                    fp.write("    Harm rate: 0 / 0 (N/A)\n")
                if isinstance(p_value, (int, float)):
                    fp.write(f"    McNemar exact p-value: {float(p_value):.6g} (discordant={int(item.get('discordant_total', 0))})\n")
                else:
                    fp.write(f"    McNemar exact p-value: N/A (discordant={int(item.get('discordant_total', 0))})\n")
            fp.write("\n")
        fp.write("\n")

    @staticmethod
    def run_mcnemar_exact(n01: int, n10: int) -> Dict[str, Any]:
        """
        Run two-sided exact McNemar test on discordant pairs of a paired 2x2 table.
        """
        discordant_total = int(n01) + int(n10)
        if discordant_total <= 0:
            return {
                "test": "exact",
                "p_value": 1.0,
                "discordant_total": 0,
            }

        k = min(int(n01), int(n10))
        tail_prob = 0.0
        for i in range(k + 1):
            tail_prob += math.comb(discordant_total, i) * (0.5**discordant_total)
        p_value = min(1.0, 2.0 * tail_prob)
        return {
            "test": "exact",
            "p_value": p_value,
            "discordant_total": discordant_total,
        }

    @classmethod
    def build_self_refute_transition_counts(
        cls,
        *,
        original_flags: Dict[str, Dict[str, Any]],
        ce_flags: Dict[str, Dict[str, Any]],
        denominator_rewrite_keys: Set[str],
    ) -> Dict[str, Dict[str, int]]:
        """
        Build original -> CE paired 2x2 table based on rewrite-level self-refute flags.
        """
        counts_by_source: Dict[str, Dict[str, int]] = {}
        for source in SELF_REFUTE_SOURCE_ORDER:
            n00 = n01 = n10 = n11 = 0
            for rewrite_key in denominator_rewrite_keys:
                original_self_refute = bool(((original_flags.get(rewrite_key, {}).get("self_refute")) or {}).get(source))
                ce_self_refute = bool(((ce_flags.get(rewrite_key, {}).get("self_refute")) or {}).get(source))
                if not original_self_refute and not ce_self_refute:
                    n00 += 1
                elif not original_self_refute and ce_self_refute:
                    n01 += 1
                elif original_self_refute and not ce_self_refute:
                    n10 += 1
                else:
                    n11 += 1
            counts_by_source[source] = {
                "n00": n00,
                "n01": n01,
                "n10": n10,
                "n11": n11,
            }
        return counts_by_source

    @classmethod
    def build_self_refute_transition_summary(
        cls,
        *,
        original_flags: Dict[str, Dict[str, Any]],
        ce_flags: Dict[str, Dict[str, Any]],
        denominator_rewrite_keys: Set[str],
        denominator_label: str,
    ) -> Dict[str, Any]:
        """
        Build CE paired self-refute paired 2x2 table, effect sizes, and McNemar statistics.
        """
        denominator_total = len(denominator_rewrite_keys)
        counts_by_source = cls.build_self_refute_transition_counts(
            original_flags=original_flags,
            ce_flags=ce_flags,
            denominator_rewrite_keys=denominator_rewrite_keys,
        )
        by_source: Dict[str, Dict[str, Any]] = {}
        for source, counts in counts_by_source.items():
            n00 = int(counts.get("n00", 0))
            n01 = int(counts.get("n01", 0))
            n10 = int(counts.get("n10", 0))
            n11 = int(counts.get("n11", 0))
            original_sr_total = n10 + n11
            ce_sr_total = n01 + n11
            repair_denom = original_sr_total
            harm_denom = n00 + n01
            by_source[source] = {
                "n00": n00,
                "n01": n01,
                "n10": n10,
                "n11": n11,
                "original_sr_rate": (original_sr_total / denominator_total if denominator_total > 0 else 0.0),
                "ce_sr_rate": (ce_sr_total / denominator_total if denominator_total > 0 else 0.0),
                "net_sr_drop": ((n10 - n01) / denominator_total if denominator_total > 0 else 0.0),
                "repair_rate": (n10 / repair_denom if repair_denom > 0 else 0.0),
                "harm_rate": (n01 / harm_denom if harm_denom > 0 else 0.0),
                "discordant_total": n01 + n10,
                "improvement_minus_harm": n10 - n01,
                "mcnemar": cls.run_mcnemar_exact(n01=n01, n10=n10),
            }
        return {
            "denominator_label": denominator_label,
            "denominator_total": denominator_total,
            "by_source": by_source,
        }

    @classmethod
    def build_ce_paired_self_refute_comparison(
        cls,
        *,
        ce_results: List[Dict[str, Any]],
        paired_non_ce_results: List[Dict[str, Any]],
        paired_non_ce_input_path: str,
    ) -> Dict[str, Any]:
        """
        Compare original vs CE self-refutation based on the successful retrieval rewrite set from the paired non-CE run.
        """
        original_flags = cls.build_self_refute_rewrite_flags(paired_non_ce_results)
        ce_flags = cls.build_self_refute_rewrite_flags(ce_results)
        denominator_rewrite_keys = cls.collect_hop_retrieved_rewrite_keys(original_flags)
        denominator_label = "rewrites with hop_wise_pred assertion_of_target_new=true in paired non-CE run"
        return {
            "paired_non_ce_input_path": paired_non_ce_input_path,
            "denominator_total": len(denominator_rewrite_keys),
            "original_summary": cls.build_self_refute_summary_from_flags(
                original_flags,
                denominator_rewrite_keys=denominator_rewrite_keys,
                denominator_label=denominator_label,
            ),
            "ce_summary": cls.build_self_refute_summary_from_flags(
                ce_flags,
                denominator_rewrite_keys=denominator_rewrite_keys,
                denominator_label=denominator_label,
            ),
            "transition_summary": cls.build_self_refute_transition_summary(
                original_flags=original_flags,
                ce_flags=ce_flags,
                denominator_rewrite_keys=denominator_rewrite_keys,
                denominator_label=denominator_label,
            ),
        }

    @staticmethod
    def load_paired_non_ce_pred_analysis_or_raise(
        paired_non_ce_input_path: str,
    ) -> List[Dict[str, Any]]:
        """
        Load the paired non-CE run's pred_analysis.json, requiring it to already contain
        reusable llm_self_refute_analysis structured results.
        """
        _, paired_pred_json_path, _ = PathHelper.derive_analysis_paths(paired_non_ce_input_path)
        if not os.path.exists(paired_pred_json_path):
            raise FileNotFoundError(
                f"Paired non-CE pred_analysis.json not found. Analyze the non-CE run with --llm-self-refute on first. path={paired_pred_json_path}"
            )
        paired_results = JsonIO.load_json(paired_pred_json_path)
        if not isinstance(paired_results, list):
            raise ValueError(f"Paired non-CE pred_analysis.json must be list[pred]. path={paired_pred_json_path}")
        if any(item.get("llm_self_refute") is None for item in paired_results):
            raise ValueError(
                "Paired non-CE pred_analysis.json is missing llm_self_refute results. "
                "Re-run analyze on the non-CE run with --llm-self-refute on. "
                f"path={paired_pred_json_path}"
            )
        return paired_results

    def prepare_standard_self_refute(self, results: List[Dict[str, Any]], config: AnalyzeConfig) -> Tuple[bool, Dict[str, Any]]:
        """
        Standard non-CE run self-refute preparation and summary.
        """
        use_llm = self.maybe_run_llm_self_refute(
            results,
            checkpoint_path=config.llm_checkpoint,
            resume=config.llm_resume,
        )
        rewrite_flags = self.build_self_refute_rewrite_flags(results)
        denominator_keys = self.collect_hop_retrieved_rewrite_keys(rewrite_flags)
        summary = self.build_self_refute_summary_from_flags(
            rewrite_flags,
            denominator_rewrite_keys=denominator_keys,
            denominator_label="rewrites with hop_wise_pred assertion_of_target_new=true",
        )
        return use_llm, summary

    def prepare_ce_paired_self_refute(
        self,
        *,
        ce_results: List[Dict[str, Any]],
        ce_input_path: str,
        config: AnalyzeConfig,
    ) -> Tuple[bool, Dict[str, Any], Dict[str, Any]]:
        """
        CE-only run self-refute preparation and paired non-CE comparison summary.
        """
        paired_non_ce_input = self.derive_paired_non_ce_metrics_path(ce_input_path)
        if not paired_non_ce_input:
            raise ValueError(f"Expected CE-only input path, but got: {os.path.abspath(ce_input_path)}")

        paired_results = self.load_paired_non_ce_pred_analysis_or_raise(paired_non_ce_input)
        paired_flags = self.build_self_refute_rewrite_flags(paired_results)
        denominator_keys = self.collect_hop_retrieved_rewrite_keys(paired_flags)
        ce_results_subset = self.filter_results_by_rewrite_keys(ce_results, denominator_keys)
        use_llm = self.maybe_run_llm_self_refute(
            ce_results_subset,
            checkpoint_path=config.llm_checkpoint,
            resume=config.llm_resume,
        )
        comparison = self.build_ce_paired_self_refute_comparison(
            ce_results=ce_results,
            paired_non_ce_results=paired_results,
            paired_non_ce_input_path=paired_non_ce_input,
        )
        summary = comparison.get("ce_summary") or {}
        return use_llm, summary, comparison

    def write_report(
        self,
        path: str,
        *,
        results: List[Dict[str, Any]],
        use_llm: bool,
        rewrite_why_summary: Optional[Dict[str, Any]] = None,
        self_refute_summary: Optional[Dict[str, Any]] = None,
        ce_paired_comparison: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Generate analysis report (TXT format)

        Report contents include:
        1. Overall statistics: pred count
        2. LLM judgment statistics (if enabled)
        3. Rewrite why consistency statistics (if enabled)

        Args:
            path: Output file path
            results: Analysis result list (pred-level)
            use_llm: Whether LLM judgment was used
            rewrite_why_summary: Rewrite why consistency summary
            self_refute_summary: Self-refutation statistics summary
            ce_paired_comparison: CE-only vs paired non-CE run comparison summary
        """
        JsonIO.ensure_parent_dir(path)
        total = len(results)

        with open(path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("Pred analysis report\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Total pred count: {total}\n\n")

            if use_llm:
                if self_refute_summary is not None:
                    self.write_self_refute_summary_block(f, self_refute_summary)
                else:
                    self.write_self_refute_summary_block(
                        f,
                        {
                            "judged_rewrite_count": 0,
                            "source_totals": {},
                            "source_yes": {},
                            "conditional": {
                                "label": "rewrites with hop_wise_pred assertion_of_target_new=true",
                                "denominator_total": 0,
                                "yes_by_source": {},
                            },
                        },
                    )

                if ce_paired_comparison is not None:
                    self.write_ce_paired_comparison_block(f, ce_paired_comparison)

            if rewrite_why_summary is not None:
                f.write("Rewrite why article-consistency stats\n")
                f.write(f"  Enabled: {'yes' if rewrite_why_summary.get('enabled') else 'no'}\n")
                f.write(f"  Skipped: {'yes' if rewrite_why_summary.get('skipped') else 'no'}\n")
                if rewrite_why_summary.get("skip_reason"):
                    f.write(f"  Skip reason: {rewrite_why_summary.get('skip_reason', '')}\n")
                f.write(f"  Scanned why-source preds: {rewrite_why_summary.get('scan_count', 0)}\n")
                f.write(f"  Candidate count: {rewrite_why_summary.get('candidate_count', 0)}\n")
                f.write(f"  Judged pred count: {rewrite_why_summary.get('judged_count', 0)}\n")
                f.write(
                    f"  PASS / FAIL / UNSURE: "
                    f"{rewrite_why_summary.get('pass_count', 0)} / "
                    f"{rewrite_why_summary.get('fail_count', 0)} / "
                    f"{rewrite_why_summary.get('unsure_count', 0)}\n"
                )
                f.write(f"  Rewrite count: {rewrite_why_summary.get('rewrite_count', 0)}\n")
                f.write(f"  Mean rewrite pass rate: {float(rewrite_why_summary.get('mean_rewrite_pass_rate', 0.0)):.1%}\n")
                skip_breakdown = rewrite_why_summary.get("skip_breakdown") or {}
                if isinstance(skip_breakdown, dict) and skip_breakdown:
                    f.write("  Skip breakdown:\n")
                    for key, value in sorted(skip_breakdown.items(), key=lambda x: str(x[0])):
                        f.write(f"    {key}: {value}\n")
                fail_examples = rewrite_why_summary.get("fail_examples") or []
                if isinstance(fail_examples, list) and fail_examples:
                    f.write("  Fail examples:\n")
                    for ex in fail_examples[:5]:
                        f.write(f"    case={ex.get('case_id')} rewrite={ex.get('rewrite_key')} subject={ex.get('rewrite_subject', '')}\n")
                        f.write(f"      question: {ex.get('question', '')}\n")
                        f.write(f"      reason: {ex.get('reason', '')}\n")
                f.write("\n")

    def run(self, config: AnalyzeConfig) -> Dict[str, Any]:
        """
        Execute complete analysis workflow

        Args:
            config: Analysis configuration

        Returns:
            Summary information dictionary
        """
        data = JsonIO.load_json(config.input_path)
        if not isinstance(data, list):
            raise ValueError("Analyze input JSON must be list[case].")

        results = self.analyze_items(data)
        use_llm = False
        rewrite_why_candidates: List[Dict[str, Any]] = []
        rewrite_why_summary: Optional[Dict[str, Any]] = None
        self_refute_summary: Optional[Dict[str, Any]] = None
        ce_paired_comparison: Optional[Dict[str, Any]] = None
        if config.llm_self_refute:
            if self.is_ce_only_input_path(config.input_path):
                use_llm, self_refute_summary, ce_paired_comparison = self.prepare_ce_paired_self_refute(
                    ce_results=results,
                    ce_input_path=config.input_path,
                    config=config,
                )
            else:
                use_llm, self_refute_summary = self.prepare_standard_self_refute(
                    results,
                    config,
                )

        if config.llm_rewrite_why:
            rewrite_why_candidates, extraction_info = self.collect_rewrite_why_candidates(
                results,
                data,
                input_path=config.input_path,
            )
            self.maybe_run_llm_rewrite_why(
                rewrite_why_candidates,
                checkpoint_path=config.rewrite_why_checkpoint,
                resume=config.llm_resume,
            )
            rewrite_why_summary = self.build_rewrite_why_summary(
                input_path=config.input_path,
                extraction_info=extraction_info,
                candidates=rewrite_why_candidates,
            )
            JsonIO.write_json(rewrite_why_candidates, config.rewrite_why_output, indent=2)
            JsonIO.write_json(rewrite_why_summary, config.rewrite_why_summary_output, indent=2)

        JsonIO.write_json(results, config.pred_json_output, indent=2)
        self.write_report(
            config.txt_output,
            results=results,
            use_llm=use_llm,
            rewrite_why_summary=rewrite_why_summary,
            self_refute_summary=self_refute_summary,
            ce_paired_comparison=ce_paired_comparison,
        )

        total = len(results)
        summary = {
            "total_pred_count": total,
            "txt_output": config.txt_output,
            "pred_json_output": config.pred_json_output,
            "rewrite_why_output": (config.rewrite_why_output if config.llm_rewrite_why else None),
            "rewrite_why_summary_output": (config.rewrite_why_summary_output if config.llm_rewrite_why else None),
            "rewrite_why_summary": rewrite_why_summary,
            "self_refute_summary": self_refute_summary,
            "ce_paired_comparison": ce_paired_comparison,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary


class CommandRunner:
    """
    CLI command runner

    Parses command-line arguments and executes the corresponding pipeline.

    Supported commands:
    - judge: Run LLM judgment pipeline
    - analyze: Run prediction analysis pipeline
    - extract-ce: Extract causal_enhanced related preds into independent metrics

    Recommended usage:
        python pred_analysis_tool.py judge --input /path/to/run/metrics.json
        python pred_analysis_tool.py analyze --input /path/to/run/metrics.json --llm-self-refute on
        python pred_analysis_tool.py extract-ce --input /path/to/run/metrics.json
    """

    @staticmethod
    def derive_ce_only_metrics_path(input_path: str) -> str:
        """
        Derive CE-only metrics output path:
        <run_dir>/metrics.json -> <run_parent>/<run_dir>_only_causal_enhanced/metrics.json
        """
        abs_input = os.path.abspath(input_path)
        input_dir = os.path.dirname(abs_input)
        run_dir_name = os.path.basename(input_dir.rstrip(os.sep))
        parent_dir = os.path.dirname(input_dir)
        ce_dir = os.path.join(parent_dir, f"{run_dir_name}_only_causal_enhanced")
        return os.path.join(ce_dir, "metrics.json")

    @staticmethod
    def build_ce_only_case(case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build CE-only case from original case.

        Rules:
        - Original case is kept unchanged (only copied)
        - post.hop_wise_pred <- post.causal_enhanced_hop_wise_pred
        - post.conflict_probe <- post.causal_enhanced_conflict_probe
        - post.accuracy / post.accuracy_pred are cleared
        """
        new_case = dict(case)
        post = case.get("post")
        post_d = post if isinstance(post, dict) else {}

        hop_wise = post_d.get("hop_wise")
        ce_hop_pred = post_d.get("causal_enhanced_hop_wise_pred")
        ce_conflict = post_d.get("causal_enhanced_conflict_probe")
        if not ce_hop_pred or not ce_conflict:
            raise ValueError(f"CE-only case must hav、e causal_enhanced_hop_wise_pred and causal_enhanced_conflict_probe")

        new_case["post"] = {
            "hop_wise": hop_wise,
            "hop_wise_pred": ce_hop_pred,
            "accuracy": [],
            "accuracy_pred": [],
            "conflict_probe": ce_conflict,
        }
        return new_case

    def run_judge_cmd(self, args: argparse.Namespace) -> Dict[str, Any]:
        """
        Execute judge command

        Perform LLM judgment on metrics JSON.

        Args:
            args: Command-line arguments

        Returns:
            Judgment result summary
        """
        judged_output, summary_output = PathHelper.derive_judge_paths(args.input, None, args.summary)
        cfg = JudgeConfig(
            model_name=args.judge_model,
            api_keys=API_KEY_POOL[:],
            base_url=args.base_url,
            max_workers=max(1, args.max_workers),
        )
        pipeline = JudgePipeline(cfg)
        return pipeline.run(
            input_metrics_file=args.input,
            output_metrics_file=judged_output,
            summary_file=summary_output,
            checkpoint_every=DEFAULT_JUDGE_CHECKPOINT_EVERY,
            checkpoint_seconds=DEFAULT_JUDGE_CHECKPOINT_SECONDS,
        )

    def run_analyze_cmd(self, args: argparse.Namespace) -> Dict[str, Any]:
        """
        Execute analyze command

        Perform prediction analysis on metrics JSON.

        Args:
            args: Command-line arguments

        Returns:
            Analysis result summary
        """
        (
            default_txt,
            default_pred_json,
            default_ckpt,
        ) = PathHelper.derive_analysis_paths(args.input)
        (
            default_rewrite_why_output,
            default_rewrite_why_summary,
            default_rewrite_why_ckpt,
        ) = PathHelper.derive_rewrite_why_paths(args.input)
        cfg = AnalyzeConfig(
            input_path=args.input,
            txt_output=args.output or default_txt,
            pred_json_output=default_pred_json,
            llm_checkpoint=args.llm_checkpoint or default_ckpt,
            llm_self_refute=(args.llm_self_refute == "on"),
            rewrite_why_output=default_rewrite_why_output,
            rewrite_why_summary_output=default_rewrite_why_summary,
            rewrite_why_checkpoint=default_rewrite_why_ckpt,
            llm_rewrite_why=(args.rewrite_why == "on"),
            llm_resume=(args.llm_resume == "on"),
        )
        judge_cfg = JudgeConfig(
            model_name=args.judge_model,
            api_keys=API_KEY_POOL[:],
            base_url=args.base_url,
            max_workers=max(1, args.max_workers),
        )
        pipeline = AnalyzePipeline(judge_cfg)
        return pipeline.run(cfg)

    def run_extract_ce_cmd(self, args: argparse.Namespace) -> Dict[str, Any]:
        """
        Execute extract-ce command

        Extract causal_enhanced related fields from original metrics, generating CE-only metrics.
        """
        output_path = self.derive_ce_only_metrics_path(args.input)
        if os.path.exists(output_path):
            summary = {
                "input": os.path.abspath(args.input),
                "output": os.path.abspath(output_path),
                "total_cases": 0,
                "skipped": True,
                "reason": "output file already exists",
            }
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return summary

        data = JsonIO.load_json(args.input)
        if not isinstance(data, list):
            raise ValueError("Extract input JSON must be list[case].")

        ce_metrics: List[Dict[str, Any]] = []
        for item in data:
            ce_metrics.append(self.build_ce_only_case(item))

        JsonIO.write_json(ce_metrics, output_path, indent=2)

        summary = {
            "input": os.path.abspath(args.input),
            "output": os.path.abspath(output_path),
            "total_cases": len(ce_metrics),
            "skipped": False,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary


def build_parser() -> argparse.ArgumentParser:
    """
    Build CLI argument parser

    Defines three subcommands:
    - judge: LLM judgment command
    - analyze: Prediction analysis command
    - extract-ce: Extract causal_enhanced results command

    Returns:
        argparse.ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description=(
            "Unified tool for LLM judge and pred analysis.\n"
            "Recommended workflow: always pass the run root metrics.json.\n"
            "Then judge outputs go to sibling judge/, and analyze outputs go to sibling analysis/."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared_llm_args(p: argparse.ArgumentParser) -> None:
        """Add shared LLM arguments"""
        p.add_argument(
            "--judge-model",
            default=DEFAULT_JUDGE_MODEL,
            help="LLM model name used by judge / self-refute / rewrite-why tasks.",
        )
        p.add_argument(
            "--base-url",
            default=DEFAULT_BASE_URL,
            help="OpenAI-compatible base URL for the LLM judge backend.",
        )
        p.add_argument("--max-workers", type=int, default=max(1, len(API_KEY_POOL) * 100))

    judge_parser = subparsers.add_parser(
        "judge",
        help="Run PASS/FAIL judge on a run metrics.json file.",
        description=(
            "Run PASS/FAIL judge on case predictions.\n"
            "Recommended input: <run_dir>/metrics.json\n"
            "Default outputs: <run_dir>/judge/metrics.judged.json and metrics.judged.summary.json"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    judge_parser.add_argument(
        "--input",
        required=True,
        help=("Input metrics JSON path.\nRecommended: pass the run root metrics.json, e.g. /path/to/run/metrics.json"),
    )
    judge_parser.add_argument(
        "--summary",
        default=None,
        help=("Optional summary JSON path.\nDefault: sibling judge/metrics.judged.summary.json inferred from --input"),
    )
    add_shared_llm_args(judge_parser)

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run pred analysis on a run metrics.json file.",
        description=(
            "Run pred analysis and optional LLM self-refute / rewrite-why checks.\n"
            "Recommended input: <run_dir>/metrics.json\n"
            "Default outputs: <run_dir>/analysis/*\n"
            "If you pass another filename such as judge/metrics.judged.json,\n"
            "the output directory will be nested according to that filename."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    analyze_parser.add_argument(
        "--input",
        required=True,
        help=("Input metrics JSON path.\nRecommended: pass the run root metrics.json, not judge/metrics.judged.json."),
    )
    analyze_parser.add_argument(
        "--output",
        default=None,
        help=("Optional TXT report path only.\nJSON outputs and checkpoint paths are still derived from --input."),
    )
    analyze_parser.add_argument(
        "--llm-self-refute",
        choices=["on", "off"],
        default="off",
        help="Whether to run LLM self-refute detection and write results into pred_analysis outputs.",
    )
    analyze_parser.add_argument(
        "--llm-checkpoint",
        default=None,
        help=("Optional self-refute checkpoint JSONL path.\nDefault: sibling analysis/pred_self_refute_analysis_llm_judge_ckpt.jsonl inferred from --input"),
    )
    analyze_parser.add_argument(
        "--rewrite-why",
        choices=["on", "off"],
        default="off",
        help="Run CausalEdit-only article-grounded consistency judge for why questions.",
    )
    analyze_parser.add_argument(
        "--llm-resume",
        choices=["on", "off"],
        default="on",
        help="Whether to resume from the inferred / specified checkpoint files.",
    )
    add_shared_llm_args(analyze_parser)

    extract_ce_parser = subparsers.add_parser(
        "extract-ce",
        help="Extract causal_enhanced preds into sibling *_only_causal_enhanced/metrics.json.",
        description=("Extract causal_enhanced preds from a run metrics.json into a sibling\n<run_name>_only_causal_enhanced/metrics.json directory."),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    extract_ce_parser.add_argument(
        "--input",
        required=True,
        help=("Input metrics JSON path.\nRecommended: pass the run root metrics.json, e.g. /path/to/run/metrics.json"),
    )

    return parser


def main() -> None:
    """
    Main entry function

    Parses command-line arguments and executes the corresponding command.
    If no arguments are provided, defaults to the analyze command.
    """
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = build_parser()
    args = parser.parse_args()
    runner = CommandRunner()

    if args.command == "judge":
        runner.run_judge_cmd(args)
    elif args.command == "analyze":
        runner.run_analyze_cmd(args)
    elif args.command == "extract-ce":
        runner.run_extract_ce_cmd(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
