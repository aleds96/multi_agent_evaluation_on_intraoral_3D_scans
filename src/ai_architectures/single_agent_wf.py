from dotenv import load_dotenv
import os
import asyncio
from pathlib import Path
from google.adk.apps import App
from google.adk.agents.llm_agent import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk import Workflow
from google.adk import Context
from google.adk.workflow import node
from src.llm_utils_fun import (
    safe_run_multimodal,
    ensure_session,
    build_multimodal_prompt,
    safe_run_node
)
from src.prompts import build_user_request_for_scan,build_self_reflection_prompt
from src.agents import eval_single_agent
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



async def eval_single_scan(
    scan_item,
    WORKFLOW_RESOURCES,
    runner,
    app_name):
    session_id = f"session_{scan_item['scan']}"

    session_init_dict = {
        "example_items": WORKFLOW_RESOURCES["example_items"],
        "scan_item": scan_item,
        "use_self_reflection": WORKFLOW_RESOURCES["use_self_reflection"]
    }

    await ensure_session(
        runner,
        app_name,
        session_id,
        session_init_dict
    )

    final_event = await safe_run_multimodal(
        runner,
        session_id,
        [],
        "start",
        max_retries=10,
    )

    verdict = final_event.output
    return verdict

async def eval_scan_task(scan_item, WORKFLOW_RESOURCES, runner, app_name, semaphore):
    async with semaphore:
        verdict = await eval_single_scan(scan_item, WORKFLOW_RESOURCES, runner, app_name)

        pred_quality = verdict["quality"]
        pred_motivation = verdict["motivation"]

        return {
            "scan": scan_item["scan"],
            "real_quality": scan_item["profile"]["quality_global"],
            "pred_quality": pred_quality,
            "motivation": pred_motivation,
        }

@node(rerun_on_resume=True)
async def evaluation_workflow(
    ctx: Context,
    node_input,
):
    example_items = ctx.state["example_items"]
    scan_item = ctx.state["scan_item"]
    use_self_reflection = ctx.state["use_self_reflection"]
    
    prompt = build_user_request_for_scan(
        scan_item,
        example_items
    )

    image_paths = (
        [item["image_pred"] for item in example_items["lvl5"]]
        + [item["image_pred"] for item in example_items["lvl4"]]
        + [item["image_pred"] for item in example_items["lvl3"]]
        + [item["image_pred"] for item in example_items["lvl2"]]
        + [item["image_pred"] for item in example_items["lvl1"]]
        + [scan_item["image_pred"]]
    )
    primary_content = build_multimodal_prompt(
                    text=prompt,
                    image_paths=image_paths,
                )
    result = await safe_run_node(
        ctx,
        eval_single_agent,
        primary_content,
        max_retries=10,
        base_delay=10,
        )
    if use_self_reflection:
        self_reflection_prompt = build_self_reflection_prompt(
            scan_item,
            example_items,
            result
        )
        self_reflection_content = build_multimodal_prompt(
            text=self_reflection_prompt,
            image_paths=image_paths,
        )
        
        result = await safe_run_node(
                ctx,
                eval_single_agent,
                self_reflection_content,
                max_retries=10,
                base_delay=10,
                )

    return result
root_agent = Workflow(
    name="landmark_quality_workflow",
    edges=[
        ("START", evaluation_workflow)
    ],
)


async def run_single_agent(dataset_primary, WORKFLOW_RESOURCES):

   
    from google.adk.plugins import LoggingPlugin
    from google.adk.plugins import DebugLoggingPlugin
    
    plugins = [
        LoggingPlugin(),          
        DebugLoggingPlugin(), 
    ]
    app = App(
        name="landmark_quality_app_v1",
        root_agent=root_agent,
        #plugins=plugins    
    )
    
    runner = InMemoryRunner(app=app)

    semaphore = asyncio.Semaphore(1)
    tasks = []
    for i, item in enumerate(dataset_primary):
        tasks.append(asyncio.create_task(
            eval_scan_task(item, WORKFLOW_RESOURCES, runner, app.name, semaphore)
        ))
        if i % 10 == 0 and i > 0:
            await asyncio.sleep(15)
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

    quantiles = {"lvl1": 0.10, "lvl2": 0.25, "lvl3":0.5, "lvl4":0.75, "lvl5": 0.90}

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
        exclude_scans=set([obs['scan'] for obs in stats_over_scan]) - exclude_examples,
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

    USE_SELF_REFLECTION= True
    WORKFLOW_RESOURCES = {
        "example_items": example_items, 
        "use_self_reflection": USE_SELF_REFLECTION
    }
    outputs = await run_single_agent(dataset_primary, WORKFLOW_RESOURCES)

    evaluation = evaluate_agent(outputs)

    config = {
        "architecture": "v1_single_agent_selfReflection_wf",
        "model": "gemini-2.5-flash",
        "primary_examples": primary_examples,
        "use_self_reflection": USE_SELF_REFLECTION,
    }

    exp_dir = save_experiment(config, outputs, evaluation)
    print("Esperimento salvato in:", exp_dir)


if __name__ == "__main__":
    asyncio.run(main())
