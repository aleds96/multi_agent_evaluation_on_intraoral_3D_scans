from dotenv import load_dotenv
import os
import asyncio
from pathlib import Path
from google.adk.apps import App
from google.adk.agents.llm_agent import LlmAgent
from google.adk.runners import InMemoryRunner

from src.llm_utils_fun import (
    run_multimodal,
    ensure_session,
)
from src.prompts import build_user_request_for_scan,instruction_prompt
from src.schemas import SingleAgentOutput
from src.landmark_eval import (
    compute_all_statistics,
    select_examples_by_quantile,
    build_input_dataset,
)
from src.llm_eval import evaluate_agent
from src.utils_fun import save_experiment
from src.var_constants import CATEGORIES


ROOT = Path(__file__).resolve().parents[2]
env_path = ROOT / ".env"
load_dotenv(env_path)
cred_path = ROOT / os.getenv("SERVICE_ACCOUNT_KEY")
os.environ["GOOGLE_CLOUD_PROJECT"] = os.getenv("GOOGLE_CLOUD_PROJECT")
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = os.getenv("GOOGLE_GENAI_USE_VERTEXAI")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cred_path)



async def eval_single_scan(scan_item, example_items, runner, app_name):
    session_id = f"session_{scan_item['scan']}"
    await ensure_session(runner, app_name, session_id)
   
    prompt = build_user_request_for_scan(scan_item, example_items)

    #print(f"Prompt for scan {scan_item['scan']}:\n{prompt}")

    image_paths = (
        [item["image_pred"] for item in example_items["lvl5"]] +
        [item["image_pred"] for item in example_items["lvl2"]] +
        [item["image_pred"] for item in example_items["lvl1"]] +
        [scan_item["image_pred"]]
    )

    final_event = await run_multimodal(
        runner,
        session_id,
        image_paths,
        prompt,
    )
    verdict = None
    if final_event.actions and "quality_verdict" in final_event.actions.state_delta:
        verdict = final_event.actions.state_delta["quality_verdict"]

    if verdict is None:
        session = await runner.session_service.get_session(app_name, session_id)
        verdict = session.state.get("quality_verdict")

    #print(f"Final verdict for scan {scan_item['scan']}:\n{verdict}")
    return verdict

async def eval_scan_task(scan_item, example_items, runner, app_name, semaphore):
    async with semaphore:
        verdict = await eval_single_scan(scan_item, example_items, runner, app_name)

        pred_quality = verdict["quality"]
        pred_motivation = verdict["motivation"]

        return {
            "scan": scan_item["scan"],
            "real_quality": scan_item["profile"]["quality_global"],
            "pred_quality": pred_quality,
            "motivation": pred_motivation,
        }



async def run_single_agent(dataset_primary, dataset_examples):

    evaluation_agent = LlmAgent(
        name="LandmarkQualityEvaluator",
        model="gemini-2.5-flash",
        output_schema=SingleAgentOutput,
        instruction=instruction_prompt,
        output_key="quality_verdict",
    )

    from google.adk.plugins import LoggingPlugin
    from google.adk.plugins import DebugLoggingPlugin
    
    plugins = [
        LoggingPlugin(),          
        DebugLoggingPlugin(), 
    ]
    app = App(
        name="landmark_quality_app_v1",
        root_agent=evaluation_agent,
        plugins=plugins    
    )
    
    runner = InMemoryRunner(app=app)

    semaphore = asyncio.Semaphore(2)
    tasks = []
    for i, item in enumerate(dataset_primary):
        tasks.append(asyncio.create_task(
            eval_scan_task(item, dataset_examples, runner, app.name, semaphore)
        ))
        if i % 5 == 0 and i > 0:
            await asyncio.sleep(30)
    outputs = await asyncio.gather(*tasks)
    return outputs

async def main():

    DATA_ROOT = ROOT / "dataset"

    SCANS = DATA_ROOT / "toothinstancenet_input"
    GT_ROOT = SCANS
    PRED_CSV = DATA_ROOT / "model_predictions" / "ynlab" / "predictions.csv"
    SCREENSHOT_DIR = DATA_ROOT / "screenshot_scans" / "raw"
     
    stats = compute_all_statistics(GT_ROOT, PRED_CSV, CATEGORIES)
    stats_over_scan = stats["results"]
    quantile_mAP = stats["quantile_mAP_over_scan"]

    quantiles = {"lvl1": 0.10, "lvl2": 0.25, "lvl5": 0.90}

    primary_examples, oracle_pool = select_examples_by_quantile(
        stats_over_scan,
        quantile_mAP,
        quantiles,
        k=1,
        m=2,
    )

    exclude_examples = set(sum(primary_examples.values(), []))
    dataset_examples = build_input_dataset(
        stats_over_scan,
        exclude_scans=set(stats_over_scan["scans"]) - exclude_examples,
        SCANS=SCANS,
        GT_ROOT=GT_ROOT,
        PRED_CSV=PRED_CSV,
        SCREENSHOT_ROOT=SCREENSHOT_DIR,
        CATEGORIES=CATEGORIES
    )
    print(f"Dataset esempi costruito con {len(dataset_examples)} scans.")
    #Raggruppo dataset_examples per livello
    example_items = {
        lvl: [
            item for item in dataset_examples
            if item["scan"] in primary_examples[lvl]
        ]
        for lvl in primary_examples
    }
     

    exclude_primary = set(sum(primary_examples.values(), []) + sum(oracle_pool.values(), []))
    dataset_primary = build_input_dataset(
        stats_over_scan,
        exclude_scans=exclude_primary,
        SCANS=SCANS,
        GT_ROOT=GT_ROOT,
        PRED_CSV=PRED_CSV,
        SCREENSHOT_ROOT=SCREENSHOT_DIR,
        CATEGORIES=CATEGORIES
    )
    print(f"Dataset primary costruito con {len(dataset_primary)} scans.")
     
    outputs = await run_single_agent(dataset_primary, example_items)

    evaluation = evaluate_agent(outputs)

    config = {
        "architecture": "v3_single_agent",
        "model": "gemini-2.5-flash",
        "primary_examples": primary_examples,
    }

    exp_dir = save_experiment(config, outputs, evaluation)
    print("Esperimento salvato in:", exp_dir)


if __name__ == "__main__":
    asyncio.run(main())
