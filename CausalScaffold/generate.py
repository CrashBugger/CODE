import json
from query_wiki import WikidataBatchFetcher
import os
from openai import OpenAI
import signal
import sys
import tempfile
import shutil
import argparse

import tenacity

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def atomic_write_json(data, file_path):
    """
    Atomically write a JSON file, avoiding data corruption from interruptions during write
    """
    dir_path = os.path.dirname(file_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp", prefix="atomic_")

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        shutil.move(temp_path, file_path)
    except Exception:
        try:
            os.unlink(temp_path)
        except:
            pass
        raise


class GracefulExit:
    """
    Graceful exit handler, catches Ctrl+C signal
    """

    def __init__(self):
        self.shutdown = False
        self.original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        print("\n\nInterrupt signal received, saving current progress...")
        self.shutdown = True
        signal.signal(signal.SIGINT, self.original_sigint)
        sys.exit(0)

    def should_exit(self):
        return self.shutdown


def load_mquake(file_path):
    """
    Load the MQuAKE-CF-3k dataset
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def generate_grounding_info(mquake_file, output_file, email="researcher@university.edu"):
    """
    Query Wikidata entity IDs to obtain background knowledge about subjects.
    Supports resuming from checkpoint.
    """
    data = load_mquake(mquake_file)
    print(f"Loaded {len(data)} samples")

    # Try loading existing grounding info
    grounding_info = {}
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            grounding_info = json.load(f)
        print(f"Loaded {len(grounding_info)} previously queried entities")
    except FileNotFoundError:
        print("No existing grounding info file found, starting from scratch")

    # Extract all unique subject IDs
    subject_ids = set()
    for item in data:
        for subject_triples in item["orig"]["new_triples"]:
            subject_ids.add(subject_triples[0])
            subject_ids.add(subject_triples[2])
        for subject_triples in item["orig"]["triples"]:
            subject_ids.add(subject_triples[0])
            subject_ids.add(subject_triples[2])

    # Filter out already queried IDs
    subject_ids = subject_ids - set(grounding_info.keys())
    subject_ids = list(subject_ids)

    print(f"Need to query {len(subject_ids)} new entities")
    print(subject_ids)

    if len(subject_ids) == 0:
        print("All entities have been queried")
        return grounding_info

    # Use WikidataBatchFetcher for batch querying
    fetcher = WikidataBatchFetcher(email=email)

    # Batch query; use smaller batches for higher success rate on unstable networks
    batch_size = 20
    total_batches = (len(subject_ids) + batch_size - 1) // batch_size

    for i in range(0, len(subject_ids), batch_size):
        batch_ids = subject_ids[i : i + batch_size]
        batch_num = i // batch_size + 1

        print(f"Processing batch {batch_num}/{total_batches}, containing {len(batch_ids)} entities...")

        try:
            results = fetcher.process_entities_batch(batch_ids)

            # Save results to grounding_info
            for result in results:
                entity_id = result["id"]
                grounding_info[entity_id] = result

            # Save immediately after each batch to support checkpoint resuming
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(grounding_info, f, indent=2, ensure_ascii=False)

            print(f"Batch {batch_num} completed, saved info for {len(grounding_info)} entities")

        except Exception as e:
            print(f"Batch {batch_num} query failed: {e}")
            # Save current progress even on failure
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(grounding_info, f, indent=2, ensure_ascii=False)
            print("Current progress saved, you can resume the run")
            raise

    print(f"✅ All queries completed! Queried {len(grounding_info)} entities in total")
    return grounding_info


def generate_article(mquake_file, grounding_info_file, output_file, graceful_exit=None):
    """
    Generate articles containing MQuAKE-CF-3k dataset content,
    incorporating Wikidata entity IDs as grounding information.
    Supports resuming from checkpoint.
    """
    data = load_mquake(mquake_file)

    # Load grounding info
    with open(grounding_info_file, "r", encoding="utf-8") as f:
        grounding_info = json.load(f)

    # Try loading previously generated articles
    generated_articles = {}
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            generated_articles = json.load(f)
        print(f"Loaded {len(generated_articles)} previously generated articles")
    except FileNotFoundError:
        print("No existing articles file found, starting from scratch")

    # Filter out already generated articles
    items_to_generate = []
    for item in data:
        case_id = item["case_id"]
        requested_rewrites = item["requested_rewrite"]
        for rewrite_idx in range(len(requested_rewrites)):
            article_key = f"{case_id}_{rewrite_idx}"
            if article_key not in generated_articles:
                items_to_generate.append((case_id, rewrite_idx, item))

    print(f"Need to generate {len(items_to_generate)} new articles")

    if len(items_to_generate) == 0:
        print("All articles have been generated")
        return generated_articles
    # Generate articles for each data item
    for idx, (case_id, rewrite_idx, item) in enumerate(items_to_generate):
        if graceful_exit and graceful_exit.should_exit():
            print("\nExit signal detected, saving current progress before exiting...")
            atomic_write_json(generated_articles, output_file)
            print(f"Saved {len(generated_articles)} articles")
            sys.exit(0)

        requested_rewrites = item["requested_rewrite"]
        edit_triples = item["orig"]["edit_triples"][rewrite_idx]
        rewrite = requested_rewrites[rewrite_idx]

        # Extract subject_id and object_id from edit_triples
        subject_id = edit_triples[0]
        object_id = edit_triples[2]

        # Get grounding info; if missing, use placeholder info to avoid task interruption
        if subject_id not in grounding_info or object_id not in grounding_info:
            print(f"Warning: grounding missing (subject={subject_id in grounding_info}, object={object_id in grounding_info}) for case {case_id}_{rewrite_idx}")
            continue

        grounding_data_subject = grounding_info.get(
            subject_id,
        )
        grounding_data_object = grounding_info.get(
            object_id,
        )

        # Get question, answer, and new answer
        question = rewrite.get("question")
        old_answer = rewrite.get("target_true").get("str")
        new_answer = rewrite.get("target_new").get("str")

        # Generate unique key: case_id + rewrite_idx
        article_key = f"{case_id}_{rewrite_idx}"

        print(f"Generating article {idx + 1}/{len(items_to_generate)} (Case {case_id}, Rewrite {rewrite_idx})...")

        try:
            # Call LLM to generate article
            article = llm_generate_article(
                old_answer,
                new_answer,
                question,
                [grounding_data_subject, grounding_data_object],
            )

            # Save generated article
            generated_articles[article_key] = {
                "case_id": case_id,
                "rewrite_idx": rewrite_idx,
                "question": question,
                "old_answer": old_answer,
                "new_answer": new_answer,
                "article": article,
            }

        except Exception as e:
            print(f"Case {case_id}, Rewrite {rewrite_idx}: Article generation failed: {e}")
        finally:
            # Save after each article to support checkpoint resuming
            atomic_write_json(generated_articles, output_file)
            print(f"Case {case_id}, Rewrite {rewrite_idx}: Article generation completed")

    print(f"✅ All articles generated! Generated {len(generated_articles)} articles in total")
    return generated_articles


def extract_xml(text, xml_tag):
    """Extract XML content from text"""
    import re

    pattern = f"<{xml_tag}>(.*?)</{xml_tag}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def verify_content(generated_article, q, old_a, new_a):
    """
    Use LLM to verify whether the generated article meets the causal editing requirements
    """
    verify_sys_prompt = """You are a rigorous Logic Verifier for a **Counterfactual (Hypothetical)** Knowledge Editing task.
Your goal is to evaluate if a generated text successfully *internalizes* a specific change in knowledge, ensuring the [New Answer] completely supersedes the [Old Answer].

**CRITICAL RULE: SUSPEND REAL-WORLD KNOWLEDGE.**
- Accept [New Answer] as the ABSOLUTE TRUTH for this context. 
- Ignore real-world facts. "Accuracy" here means consistency with the provided update.

**Criteria for PASS (All must be met):**
1. **Explicit Support:** The text establishes [New Answer] as the *only* current fact.
2. **Explicit Refutation:** The text must characterize [Old Answer] as incorrect, outdated, or a misunderstanding. It should NOT present [Old Answer] as a still-valid alternative.
3. **Causal Coherence:** The transition from [Old Answer] to [New Answer] is explained by a logical "cause" mentioned in the text.
4. **No Semantic Confusion:** The text does not accidentally assign attributes of the [New Answer] to the [Old Answer] or vice versa.

**Output Requirement:**
- Output a single XML tag: <verification>PASS</verification> or <verification>FAIL</verification>.
- If FAIL, provide a brief, objective explanation of the logical failure (e.g., "The text allows for both answers to be true simultaneously").
"""

    # User prompt remains unchanged, or can be slightly emphasized
    verify_user_prompt = f"""
**Task:** Verify this generated counterfactual article.

**Target Knowledge Update:**
- Question: {q}
- Old Answer (To be corrected): {old_a}
- New Answer (The NEW Truth): {new_a}

**Generated Article:**
{generated_article}

**Evaluate:** Does the article successfully internalize the New Answer and explain the shift from the Old Answer?.
"""

    # Call LLM for verification (recommended to use a lightweight high-intelligence model, or reuse the same model)
    v_response = (
        client.chat.completions.create(
            model=verify_model_name,  # Or use a stronger model like qwen-plus for better verification quality
            messages=[
                {"role": "system", "content": verify_sys_prompt},
                {"role": "user", "content": verify_user_prompt},
            ],
            stream=False,
        )
        .choices[0]
        .message.content
    )

    result = extract_xml(v_response, "verification")
    return result, v_response  # Return result and full response (for debugging)


# Retry up to 200 times on verification failure
@tenacity.retry(
    stop=tenacity.stop_after_attempt(200),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    retry=tenacity.retry_if_exception_type(ValueError),
)
def llm_generate_article(old_answer, new_answer, question, grounding_info_list):
    def _format_all_profiles(profile_list):
        """
        Process a list containing multiple entities.
        Input: [raw_json_entity1, raw_json_entity2, ...]
        Output: A single formatted long string.
        """

        formatted_sections = []

        # Define whitelist fields to keep
        target_keys = [
            "occupation",
            "date of birth",
            "date of death",
            "place of birth",
            "place of death",
            "country of citizenship",
            "employer",
            "educated at",
            "significant person",
            "genre",
        ]

        for raw_json in profile_list:
            # 1. Get name and description
            label = raw_json.get("label")
            description = raw_json.get("description")

            # 2. Build text block for a single entity
            lines = [f"--- Entity: {label} ---"]  # Clear separator header
            lines.append(f"Description: {description}")

            props = raw_json.get("properties")

            # 3. Extract properties
            for key in target_keys:
                if key in props:
                    values = props[key]
                    # Simple cleanup: remove timestamps from dates, take only first 3 items from lists
                    if "date" in key:
                        cleaned_values = [v.replace("+", "").split("T")[0] for v in values]
                    else:
                        cleaned_values = values[:3]

                    lines.append(f"{key.capitalize()}: {', '.join(cleaned_values)}")

            formatted_sections.append("\n".join(lines))

        # Join all entities with double newlines
        return "\n\n".join(formatted_sections)

    context_text = _format_all_profiles(grounding_info_list)

    system_prompt = """You are a professional news editor.
Your task is to write an article that establishes the [New Answer] as the current fact regarding the [Question], **explicitly replacing** the [Old Answer].

**Guidelines:**
1. **The Mechanism of Change:** You MUST invent a plausible, logical cause or causality to justify this update.
2. **Minimize Knowledge Conflicts:** Minimize knowledge conflicts with the existing knowledge, and ensure [New Answer] is the **ONLY truth** for the specific [Question].
3. **Natural and Fluent Prose:** The integration of the new fact should feel like a seamless part of the story, not an abrupt insertion.
4. **Identity Anchoring :** In the article, you MUST explicitly introduce the subject. Mention their key attributes to **clearly establish who they are and eliminate any ambiguity** regarding their identity.

**Workflow:**
1. **Brainstorming:** Analyze why the [Old Answer] is no longer accurate, how the [New Answer] became the new truth, and how to bridge the "Cause" and the "New Fact" elegantly. Ideate 2-3 logical scenarios. 
2. **Outlining:** Select the most convincing scenario and structure the report (Headline, Lead, Body, Conclusion).
3. **News Writing:** Draft the final article based on your outline.

**Output Requirement:**
Your final news report MUST be wrapped in <article> tags. (You may show your brainstorming and outline process before the tags freely.)"""

    user_prompt = f"""**Knowledge Update:**
- **Question:** {question}
- **OLD Answer (To be corrected):** {old_answer}
- **NEW Answer (The New Truth):** {new_answer}
 
**Reference Material (Subject Background):**
{context_text}
"""

    article = None
    response = (
        client.chat.completions.create(
            model=generate_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=False,
        )
        .choices[0]
        .message.content
    )
    article = extract_xml(response, "article")
    if article is None:
        raise ValueError("Generated article is empty")
    print(article)
    for i in range(2):
        print(f"  -> Verifying article logic...")
        verify_result, verify_full_response = verify_content(article, question, old_answer, new_answer)
        if verify_result == "PASS":
            print("  -> Verification passed ✅")
        else:
            # Verification failed, raise ValueError which will trigger @tenacity.retry to re-run the entire function
            error_msg = f"Logic verification failed (Verifier Output: {verify_result}). LLM Feedback: {verify_full_response}"
            print(f"  -> {error_msg}")
            raise ValueError(error_msg)
    return article


def merge_articles_with_mquake(mquake_file, generated_articles_file, output_file, graceful_exit=None):
    """
    Merge generated articles with the MQuAKE dataset to produce data in the CausalEdit pipeline format

    The merged data format satisfies the requirements of the _normalize_requests function:
    - subject: Subject
    - question: Question
    - prompt: Template for generating cloze questions
    - article: Generated reference article
    - target_true: {"str": old_answer}
    - target_new: {"str": new_answer}

    Args:
        mquake_file: Path to the MQuAKE dataset file
        generated_articles_file: Path to the generated articles file
        output_file: Path to the merged output file
        graceful_exit: Graceful exit handler
    """
    # Load MQuAKE dataset
    mquake_data = load_mquake(mquake_file)
    print(f"Loaded MQuAKE dataset: {len(mquake_data)} samples")

    # Load generated articles
    with open(generated_articles_file, "r", encoding="utf-8") as f:
        generated_articles = json.load(f)
    print(f"Loaded generated articles: {len(generated_articles)} articles")

    # Try loading existing merge results (supports checkpoint resuming)
    merged_data = []
    processed_keys = set()

    # Iterate over MQuAKE dataset and merge generated articles
    for idx, item in enumerate(mquake_data):
        case_id = item["case_id"]
        requested_rewrites = item.get("requested_rewrite")
        merged_item = item

        # Create merged entry for each rewrite
        for rewrite_idx, rewrite in enumerate(requested_rewrites):
            article_key = f"{case_id}_{rewrite_idx}"
            # Check if there is a corresponding generated article
            if article_key not in generated_articles:
                print(f"Warning: No article found for {article_key}, skipping")
                continue
            merged_item["requested_rewrite"][rewrite_idx]["article"] = generated_articles.get(article_key).get("article")

        merged_data.append(merged_item)
    # Final save
    atomic_write_json(merged_data, output_file)
    print(f"✅ Merge completed! Generated {len(merged_data)} merged samples")
    print(f"Output file: {output_file}")

    return merged_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate causal narratives for MQuAKE datasets")
    parser.add_argument("--generate_model_name", type=str, required=True, help="LLM model name for article generation")
    parser.add_argument("--verify_model_name", type=str, required=True, help="LLM model name for article verification")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI-compatible API key")
    parser.add_argument("--base_url", type=str, required=True, help="OpenAI-compatible base URL")
    parser.add_argument("--mquake_file", type=str, required=True, help="Path to the MQuAKE dataset file")
    parser.add_argument("--grounding_info_file", type=str, required=True, help="Output path for Wikidata entity grounding info")
    parser.add_argument("--generated_articles_file", type=str, required=True, help="Output path for generated articles")
    parser.add_argument("--merged_file", type=str, required=True, help="Output path for the final merged dataset")
    args = parser.parse_args()

    # Initialize graceful exit handler
    graceful_exit = GracefulExit()

    # Configure LLM client
    generate_model_name = args.generate_model_name
    verify_model_name = args.verify_model_name
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
    )

    # Step 1: Generate grounding info
    print("=" * 50)
    print("Step 1: Generate grounding info")
    print("=" * 50)
    grounding_info = generate_grounding_info(
        mquake_file=args.mquake_file,
        output_file=args.grounding_info_file,
        email="researcher@university.edu",
    )

    # Step 2: Generate articles
    print("\n" + "=" * 50)
    print("Step 2: Generate articles")
    print("=" * 50)
    generated_articles = generate_article(
        mquake_file=args.mquake_file,
        grounding_info_file=args.grounding_info_file,
        output_file=args.generated_articles_file,
        graceful_exit=graceful_exit,
    )

    # Step 3: Merge articles with MQuAKE dataset
    print("\n" + "=" * 50)
    print("Step 3: Merge articles with MQuAKE dataset")
    print("=" * 50)
    merged_data = merge_articles_with_mquake(
        mquake_file=args.mquake_file,
        generated_articles_file=args.generated_articles_file,
        output_file=args.merged_file,
        graceful_exit=graceful_exit,
    )

    print("\n" + "=" * 50)
    print("Done!")
    print("=" * 50)
    print(f"Grounding info file: {args.grounding_info_file}")
    print(f"Generated articles file: {args.generated_articles_file}")
    print(f"Merged file: {args.merged_file}")
