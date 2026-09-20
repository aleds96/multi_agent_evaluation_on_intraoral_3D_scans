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
from google.adk import Event
from src.llm_utils_fun import (
    safe_run_multimodal,
    ensure_session,
    build_multimodal_prompt
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
EXAMPLE_ITEMS = None
ORACLE_DATASET = None
RUNNER = None
APP_NAME = None


async def eval_single_scan(scan_item, example_items, runner, app_name):
    session_id = f"session_{scan_item['scan']}"
    await ensure_session(runner, app_name, session_id)
   
    prompt = build_user_request_for_scan(scan_item, example_items)

    #print(f"Prompt for scan {scan_item['scan']}:\n{prompt}")

    image_paths = (
        [item["image_pred"] for item in example_items["lvl5"]] +
        [item["image_pred"] for item in example_items["lvl4"]] +
        [item["image_pred"] for item in example_items["lvl3"]] +
        [item["image_pred"] for item in example_items["lvl2"]] +
        [item["image_pred"] for item in example_items["lvl1"]] +
        [scan_item["image_pred"]]
    )
   
    session = await runner.session_service.get_session(app_name=app_name,user_id="eval_user",
session_id=session_id,)
    print('###### session info')
    #print(type(session))
    #print(dir(session))
    #print(session)
    #print('session.state=>',session.state)
   
    final_event = await safe_run_multimodal(
        runner,
        session_id,
        image_paths,
        prompt,
        max_retries=10,
    )
    print('####### FINALE EVENT==>')
    #print(final_event)
    verdict = final_event.output
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


evaluation_agent = LlmAgent(
    name="LandmarkQualityEvaluator",
    model="gemini-2.5-flash",
    output_schema=SingleAgentOutput,
    instruction=instruction_prompt,
    output_key="quality_verdict",
)

@node(rerun_on_resume=True)
async def init_node(
    ctx: Context,
    node_input,
):

    print("INIT NODE")
    print(type(node_input))

    yield Event(
        output=node_input,
        state={
            "example_items": EXAMPLE_ITEMS,
            "oracle_dataset": ORACLE_DATASET,
        },
    )

@node(rerun_on_resume=True)
async def primary_node(
    ctx: Context,
    node_input,
):

    print("###### primary node")
    #print('\n node_input==>',node_input,'\n**')
    result = await ctx.run_node(
        evaluation_agent,
        node_input,
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
            evaluation_agent,
            oracle_content,
        )
        if int(prediction["quality"]) == int(oracle_scan["profile"]["quality_global"]):
            continue
        base_meta_prompt =build_user_request_for_scan(oracle_scan,example_items,goal=False )
        #passa all'agente descrittore della metavalutazione
        meta_prompt = f"""
        {base_meta_prompt}

        ----------------------------------------

        RISULTATO DEL VALUTATORE

        Predicted quality:
        {prediction["quality"]}

        Ground truth quality:
        {oracle_scan["profile"]["quality_global"]}

        Evaluator motivation:
        {prediction["motivation"]}

        ----------------------------------------

        Compito:

        Analizza perché il valutatore ha prodotto
        una qualità diversa dalla ground truth.

        NON rivalutare la scan.

        Individua:

        - possibili bias
        - elementi visivi che hanno tratto in inganno il valutatore
        - landmark coinvolti
        - motivazioni corrette
        - motivazioni errate o incomplete
        Restituisci breve (2-6 frasi max) analisi del valutatore 
        """
        meta_content = build_multimodal_prompt(text=meta_prompt,image_paths=image_paths)
        meta_description = await ctx.run_node(oracle_error_descriptor_agent,meta_content)
        #print('########### append new:',meta_description,'\n' )
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
from pydantic import BaseModel

class OracleErrorDescription(BaseModel):
    failure_analysis: str
oracle_error_descriptor_agent = LlmAgent(
    name="OracleErrorDescriptor",
    model="gemini-2.5-flash",
    output_schema=OracleErrorDescription,
    instruction="""
You are auditing a landmark quality evaluator.

You will receive:

- the same quality rubric used by the evaluator
- the same few-shot examples
- the same image
- the same landmark statistics
- the evaluator prediction
- the evaluator motivation
- the ground-truth quality

Your task is NOT to assign a new quality score.

Your task is to explain why the evaluator
probably produced a judgement different
from the ground truth.

Focus on:

- possible visual cues that misled the evaluator
- landmark classes that may have contributed
- weaknesses or blind spots in the evaluator reasoning
- recurring failure patterns that could generalize

Produce a concise analysis (2-6 sentences).

Do not restate the prediction or the ground truth.
Do not generate a new quality score.
"""
)

class OracleProfile(BaseModel):
    strengths: list[str]
    weaknesses: list[str]
    failure_modes: list[str]
    profile_summary: str
oracle_profile_builder_agent = LlmAgent(
    name="OracleProfileBuilder",
    model="gemini-2.5-flash",
    output_schema=OracleProfile,
    instruction="""
You are building a reliability profile of a landmark quality evaluator.

You will receive multiple failure analyses
generated from different scans.

Your task is to identify recurring patterns.

Do NOT analyse individual scans.

Instead, summarize:

- evaluator strengths
- evaluator weaknesses
- recurring failure modes
- possible systematic biases

Focus only on patterns that appear
across multiple examples.

Return concise but informative summaries (max 10 sentences).
"""
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
class FinalDecision(BaseModel):
    quality: int
    confidence: float
    final_motivation: str
final_decision_agent = LlmAgent(
    name="FinalDecisionAgent",
    model="gemini-2.5-flash",
    output_schema=FinalDecision,
    instruction="""
You are the final reviewer.

You will receive:

- primary prediction
- oracle evaluator profile

Your task is NOT to create a new evaluation
from scratch.

Use the oracle profile to estimate how much
the primary prediction should be trusted.

Return:

- final quality
- confidence score between 0 and 1
- concise justification
"""
)
def build_final_review_prompt(
    scan_item,
    example_items,
    primary_prediction,
    oracle_profile,
):
    base_prompt = build_user_request_for_scan(
        scan_item,
        example_items,
        goal=False
    )

    return f"""
{base_prompt}

--------------------------------------------------

VALUTAZIONE PRIMARIA

Predicted quality:
{primary_prediction["quality"]}

Motivazione:
{primary_prediction["motivation"]}

--------------------------------------------------

PROFILO DELL'ORACOLO

Summary:
{oracle_profile["profile_summary"]}

Strengths:
{oracle_profile["strengths"]}

Weaknesses:
{oracle_profile["weaknesses"]}

Failure modes:
{oracle_profile["failure_modes"]}

--------------------------------------------------

COMPITO

Valuta nuovamente la scansione.

Utilizza:

- le stesse immagini few-shot
- la stessa scala qualitativa
- le statistiche quantitative
- l'immagine target

In aggiunta considera:

- la valutazione primaria
- i bias e failure mode osservati dal profilo Oracle

L'obiettivo è produrre una valutazione calibrata.

Se ritieni che la valutazione primaria sia affetta
da uno dei failure mode osservati,
puoi correggerla.

Restituisci:

- quality
- motivation
"""
@node(rerun_on_resume=True)
async def final_node(
    ctx: Context,
    node_input,
):

    print("###### final node")

    scan_item = ctx.state["input_scan"]

    example_items = ctx.state["example_items"]

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

    yield Event(output=final_prediction)


@node(rerun_on_resume=True)
async def end_node(
    ctx: Context,
    node_input,
):
    print("###### end node")
    print('########### oracolo profile=>',ctx.state['oracle_profile'])
    #print(ctx.state["primary_prediction"])
    #print(len(ctx.state['oracle_result']["oracle_descriptions"])) 
    #print('########## \n oracl description==>',ctx.state['oracle_result']["oracle_descriptions"] )
    yield Event(output={"quality":5,"motivation": 'this is a test'})
root_agent = Workflow(
    name="workflow",
    edges=[
        (
            "START",
            init_node,
            primary_node,
            oracle_node,
            oracle_profile_builder_node,
            end_node,
        )
    ],
)
async def run_agentic_workflow(dataset_primary, dataset_examples):

   
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
            eval_scan_task(item, dataset_examples, runner, app.name, semaphore)
        ))
        if i % 10 == 0 and i > 0:
            await asyncio.sleep(15)
    outputs = await asyncio.gather(*tasks)
    return outputs

async def main():

    global EXAMPLE_ITEMS
    global ORACLE_DATASET
    global RUNNER
    global APP_NAME
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



    EXAMPLE_ITEMS = example_items 
    ORACLE_DATASET = dataset_primary[:2]
    outputs = await run_agentic_workflow(dataset_primary, example_items)

    evaluation = evaluate_agent(outputs)

    config = {
        "architecture": "v1_walllv_onlyinputstat_multi_agent_wf__test",
        "model": "gemini-2.5-flash",
        "primary_examples": primary_examples,
    }

    exp_dir = save_experiment(config, outputs, evaluation)
    print("Esperimento salvato in:", exp_dir)


if __name__ == "__main__":
    asyncio.run(main())
