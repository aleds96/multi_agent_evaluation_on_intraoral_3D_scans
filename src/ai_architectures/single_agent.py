# single_agent_main.py

import asyncio
from pathlib import Path

from google.adk.apps import App
from google.adk.agents.llm_agent import LlmAgent
from google.adk.runners import InMemoryRunner

# --- UTILS LLM ---
from llm_utils import (
    run_multimodal,
    ensure_session,
    parse_quality_output,
)

# --- PROMPTS ---
from prompts import build_prompt_for_scan

# --- DATA UTILS ---
from landmark_eval import (
    compute_all_statistics,
    select_grounding_examples,
    compute_grounding_profiles,
    build_input_dataset,
)

from evaluate_agent import evaluate_agent
from experiment_manager import save_experiment

from var_constants import CATEGORIES, PROJECT_ROOT


# ------------------------------------------------------------
# SINGLE SCAN EVALUATION
# ------------------------------------------------------------
async def eval_single_scan(scan_item, runner, app_name):
    session_id = f"session_{scan_item['scan']}"
    await ensure_session(runner, app_name, session_id)

    prompt = build_prompt_for_scan(
        input_profile=scan_item["profile"],
        examples=scan_item["examples"],
    )

    final_text = await run_multimodal(
        runner,
        session_id,
        [scan_item["image_pred"]],
        prompt
    )

    return final_text


# ------------------------------------------------------------
# BATCH TASK
# ------------------------------------------------------------
async def eval_scan_task(scan_item, runner, app_name, semaphore):
    async with semaphore:
        final_text = await eval_single_scan(scan_item, runner, app_name)
        pred_quality, pred_motivation = parse_quality_output(final_text)

        return {
            "scan": scan_item["scan"],
            "real_quality": scan_item["profile"]["quality_global"],
            "pred_quality": pred_quality,
            "motivation": pred_motivation,
        }


# ------------------------------------------------------------
# MAIN ARCHITECTURE (single agent)
# ------------------------------------------------------------
async def run_single_agent(dataset_primary, profiles_primary):

    evaluation_agent = LlmAgent(
        name="LandmarkQualityEvaluator",
        description="Valuta la qualità dei landmark dentali in una scan 3D.",
        model="gemini-2.5-flash",
        instruction="Valuta la qualità della scan in base agli esempi e al profilo fornito.",
        output_key="quality_verdict",
    )

    app = App(
        name="landmark_quality_app_v1",
        root_agent=evaluation_agent,
    )

    runner = InMemoryRunner(app=app)

    semaphore = asyncio.Semaphore(20)
    tasks = []

    for item in dataset_primary:
        item["examples"] = profiles_primary
        tasks.append(asyncio.create_task(
            eval_scan_task(item, runner, app.name, semaphore)
        ))

    outputs = await asyncio.gather(*tasks)
    return outputs


# ------------------------------------------------------------
# FULL EXPERIMENT PIPELINE
# ------------------------------------------------------------
async def main():

    DATA_ROOT = PROJECT_ROOT / "dataset"
    SCANS = DATA_ROOT / "toothinstancenet_input"
    GT_ROOT = SCANS
    PRED_CSV = DATA_ROOT / "model_predictions" / "ynlab" / "predictions.csv"
    SCREENSHOT_DIR = DATA_ROOT / "screenshot_scans" / "raw"

    # 1) STATISTICHE
    stats = compute_all_statistics(GT_ROOT, PRED_CSV, CATEGORIES)
    results = stats["results"]
    quantile_mAP = stats["quantile_mAP_over_scan"]

    # 2) SELEZIONE ESEMPI
    quantiles = {"lvl1": 0.10, "lvl2": 0.25, "lvl5": 0.90}

    primary_examples, oracle_pool = select_grounding_examples(
        results,
        quantile_mAP,
        quantiles,
        k=1,
        m=2
    )

    # 3) PROFILI
    profiles_primary = compute_grounding_profiles(results, primary_examples)

    # 4) DATASET
    exclude = set(sum(primary_examples.values(), []) + sum(oracle_pool.values(), []))

    dataset_primary = build_input_dataset(
        results,
        exclude,
        SCANS,
        GT_ROOT,
        PRED_CSV,
        SCREENSHOT_DIR
    )

    # 5) CHIAMATA AL SINGOLO AGENTE
    outputs = await run_single_agent(dataset_primary, profiles_primary)

    # 6) VALUTAZIONE
    evaluation = evaluate_agent(outputs)

    # 7) SALVATAGGIO
    config = {
        "architecture": "v1_single_agent",
        "model": "gemini-2.5-flash",
        "primary_examples": primary_examples,
    }

    exp_dir = save_experiment(config, outputs, evaluation)
    print("Esperimento salvato in:", exp_dir)


# ------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(main())
