from dotenv import load_dotenv
import os
import asyncio
from pathlib import Path
from google.adk.apps import App
from google.adk.runners import InMemoryRunner
from google.adk import Workflow
from google.adk import Context
from google.adk.workflow import node
from google.adk import Event
from src.llm_utils_fun import (
    safe_run_multimodal,
    ensure_session,
    build_multimodal_prompt
)
from src.prompts import (
    build_user_request_for_scan,
    build_oracle_error_descriptor_agent_prompt,
    build_final_review_prompt)
from src.agents import (
    eval_single_agent,
    oracle_error_descriptor_agent,
    oracle_profile_builder_agent,
    final_decision_agent)
from src.landmark_eval import (
    compute_all_statistics,
    select_examples_by_quantile,
    build_input_dataset,
)
from src.llm_eval import evaluate_agent
from src.utils_fun import save_experiment,load_oracle_cache,save_oracle_cache
from src.var_constants import CATEGORIES


ROOT = Path(__file__).resolve().parents[2]
env_path = ROOT / ".env"
load_dotenv(env_path)
cred_path = ROOT / os.getenv("SERVICE_ACCOUNT_KEY")
os.environ["GOOGLE_CLOUD_PROJECT"] = os.getenv("GOOGLE_CLOUD_PROJECT")
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = os.getenv("GOOGLE_GENAI_USE_VERTEXAI")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cred_path)
EXPERIMENT_NAME = "v1_walllv_onlyinputstat_multi_agent_wf__test"
MODEL_NAME = "gemini-2.5-flash"
ORACLE_CACHE_LOCK = asyncio.Lock()
async def eval_single_scan(scan_item, WORKFLOW_RESOURCES, runner, app_name):
    session_id = f"session_{scan_item['scan']}"
    session_init_dict = {"example_items":WORKFLOW_RESOURCES['example_items'],
                         "oracle_dataset": WORKFLOW_RESOURCES['oracle_dataset'],
                         "scan_item": scan_item}
    await ensure_session(runner, app_name, session_id,session_init_dict)
   
    #prompt = build_user_request_for_scan(scan_item, WORKFLOW_RESOURCES['example_items'])
    prompt='start'
    final_event = await safe_run_multimodal(
        runner,
        session_id,
        [],
        prompt,
        max_retries=10,
    )
    #print('####### FINALE EVENT==>')
    #print(final_event)
    verdict = final_event.output
    return verdict

async def eval_scan_task(scan_item, WORKFLOW_RESOURCES,runner, app_name, semaphore):
    async with semaphore:
        verdict = await eval_single_scan(scan_item, WORKFLOW_RESOURCES, runner, app_name)
        pred_quality = verdict['final_prediction']["quality"]
        pred_motivation = verdict['final_prediction']["motivation"]

        return {
            "scan": scan_item["scan"],
            "real_quality": scan_item["profile"]["quality_global"],
            "pred_quality": pred_quality,
            "motivation": pred_motivation,
            "workflow_trace": verdict,
        }



@node(rerun_on_resume=True)
async def primary_node(
    ctx: Context,
    node_input,
):
    print("###### primary node")
    example_items = ctx.state["example_items"]
    scan_item = ctx.state["scan_item"]
    prompt = build_user_request_for_scan(scan_item, example_items)    
    image_paths = (
        [item["image_pred"] for item in example_items["lvl5"]] +
        [item["image_pred"] for item in example_items["lvl4"]] +
        [item["image_pred"] for item in example_items["lvl3"]] +
        [item["image_pred"] for item in example_items["lvl2"]] +
        [item["image_pred"] for item in example_items["lvl1"]] +
        [scan_item["image_pred"]]
    )
    primary_content = build_multimodal_prompt(
                text=prompt,
                image_paths=image_paths,
            )
    #print('\n node_input==>',node_input,'\n**')
    result = await ctx.run_node(
        eval_single_agent,
        primary_content,
    )
    print("PRIMARY RESULT =>", result)
    yield Event(
        output=node_input,
        state={
            "primary_prediction": result
        },
    )

@node(rerun_on_resume=True)
async def oracle_node(
    ctx: Context,
    node_input,
):
    print("###### oracle node")
    example_items = ctx.state["example_items"]
    oracle_dataset = ctx.state["oracle_dataset"]
    oracle_descriptions=[]
    for oracle_scan in oracle_dataset:
        scan_name = oracle_scan["scan"]
        print('### oracle=> scan_name=>',scan_name)
        async with ORACLE_CACHE_LOCK:
            oracle_cache = load_oracle_cache(
                EXPERIMENT_NAME
            )
            cached = oracle_cache.get(scan_name,None)
        if cached is not None:
            print('### usando cache ####')
            oracle_descriptions.append({
                        "scan": oracle_scan["scan"],
                        "ground_truth":
                            oracle_scan["profile"]["quality_global"],
            
                        "predicted_quality":
                            cached["quality"],
            
                        "predicted_motivation":
                            cached["motivation"],
            
                        "failure_analysis":
                            cached["failure_analysis"],
                    })
            continue
        prompt = build_user_request_for_scan(
            oracle_scan,
            example_items,
            goal=True
        )
        image_paths = (
            [item["image_pred"] for item in example_items["lvl5"]]
            + [item["image_pred"] for item in example_items["lvl4"]]
            + [item["image_pred"] for item in example_items["lvl3"]]
            + [item["image_pred"] for item in example_items["lvl2"]]
            + [item["image_pred"] for item in example_items["lvl1"]]
            + [oracle_scan["image_pred"]]
        )
        oracle_content = build_multimodal_prompt(
            text=prompt,
            image_paths=image_paths,
        )

        prediction = await ctx.run_node(
            eval_single_agent,
            oracle_content,
        )
        if int(prediction["quality"]) == int(oracle_scan["profile"]["quality_global"]):
            print('### prediction quality uguale a ground truth. Skip!!####')
            continue
        #passa all'agente descrittore della metavalutazione
        meta_prompt = build_oracle_error_descriptor_agent_prompt(oracle_scan,prediction,example_items)
        meta_content = build_multimodal_prompt(text=meta_prompt,image_paths=image_paths)
        meta_description = await ctx.run_node(oracle_error_descriptor_agent,meta_content)
        
        async with ORACLE_CACHE_LOCK:
            oracle_cache = load_oracle_cache(
                EXPERIMENT_NAME
            )
            oracle_cache[scan_name] = {
                "quality": prediction["quality"],
                "motivation": prediction["motivation"],
                "failure_analysis":
                    meta_description["failure_analysis"],
            }
            save_oracle_cache(
                EXPERIMENT_NAME,
                oracle_cache)
        oracle_descriptions.append({
        "scan": oracle_scan["scan"],
        "ground_truth":
            oracle_scan["profile"]["quality_global"],

        "predicted_quality":
            prediction["quality"],

        "predicted_motivation":
            prediction["motivation"],

        "failure_analysis":
            meta_description["failure_analysis"],
    })

    yield Event(
        output=node_input,
        state={
            "oracle_result": {
                "primary_prediction":
                    ctx.state["primary_prediction"],
                "oracle_descriptions":
                    oracle_descriptions,
            }
        },
    )



@node(rerun_on_resume=True)
async def oracle_profile_builder_node(
    ctx: Context,
    node_input,
):
    print("###### oracle profile builder")
    descriptions = ctx.state['oracle_result']["oracle_descriptions"]
    if len(descriptions) == 0:
        oracle_profile = {
            "strengths": [
                "No errors observed in oracle dataset"
            ],
            "weaknesses": [],
            "failure_modes": [],

            "profile_summary":
                "Evaluator predictions matched all available oracle examples."
        }
    else:
        profile_input = {
            "oracle_error_descriptions":
                descriptions
        }
        oracle_profile = await ctx.run_node(
            oracle_profile_builder_agent,
            profile_input,
        )
    yield Event(
        output=node_input,
        state={
            "oracle_profile":
                oracle_profile
        },
    )


@node(rerun_on_resume=True)
async def final_node(
    ctx: Context,
    node_input,
):
    print("###### final node")

    example_items = ctx.state["example_items"]
    oracle_error_descr_per_scan = ctx.state['oracle_result']["oracle_descriptions"]
    scan_item = ctx.state["scan_item"]

    primary_prediction = ctx.state[
        "primary_prediction"
    ]

    oracle_profile = ctx.state[
        "oracle_profile"
    ]

    prompt = build_final_review_prompt(
        scan_item=scan_item,
        example_items=example_items,
        primary_prediction=primary_prediction,
        oracle_profile=oracle_profile,
    )

    image_paths = (
        [item["image_pred"] for item in example_items["lvl5"]]
        + [item["image_pred"] for item in example_items["lvl4"]]
        + [item["image_pred"] for item in example_items["lvl3"]]
        + [item["image_pred"] for item in example_items["lvl2"]]
        + [item["image_pred"] for item in example_items["lvl1"]]
        + [scan_item["image_pred"]]
    )

    final_content = build_multimodal_prompt(
        text=prompt,
        image_paths=image_paths,
    )

    final_prediction = await ctx.run_node(
        final_decision_agent,
        final_content,
    )

    print("FINAL RESULT =>", final_prediction)

    yield Event(
    output={
        "scan":scan_item["scan"],
        "final_prediction": final_prediction,

        "primary_prediction":
            primary_prediction,
        "oracle_profile":
            oracle_profile,
        "oracle_error_descr_per_scan":oracle_error_descr_per_scan,
    })
@node(rerun_on_resume=True)
async def end_node(
    ctx: Context,
    node_input,
):
    print("###### end node")
    yield Event(
        output=node_input
    )
#yield Event(output={"quality":5,"motivation": 'this is a test'})
root_agent = Workflow(
    name="workflow",
    edges=[
        (
            "START",
            primary_node,
            oracle_node,
            oracle_profile_builder_node,
            final_node,
            end_node,
        )
    ],
)
async def run_agentic_workflow(dataset_primary, WORKFLOW_RESOURCES):

   
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
    )[:3]
    print(f"Dataset primary costruito con {len(dataset_primary)} scans.")

    #passa risorse che saranno usati per inizializzare lo stato
    WORKFLOW_RESOURCES = {
    "example_items": example_items,
    "oracle_dataset": dataset_primary[:2],
    }
    outputs = await run_agentic_workflow(dataset_primary, WORKFLOW_RESOURCES)

    evaluation = evaluate_agent(outputs)

    config = {
        "architecture": EXPERIMENT_NAME,
        "model": MODEL_NAME,
        "primary_examples": primary_examples,
    }

    exp_dir = save_experiment(config, outputs, evaluation)
    print("Esperimento salvato in:", exp_dir)


if __name__ == "__main__":
    asyncio.run(main())
