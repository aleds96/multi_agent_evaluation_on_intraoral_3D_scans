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
import time
from src.llm_utils_fun import (
    safe_run_multimodal,
    ensure_session,
    build_multimodal_prompt, 
    safe_run_node
    
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
    build_synthetic_oracle_dataset, 
    build_missing_landmark_configs,
    build_class_flip_configs

)
from src.llm_eval import evaluate_agent
from src.utils_fun import save_experiment,load_oracle_cache,save_oracle_cache, build_perturbation_label,build_profile_cache_key
from src.var_constants import CATEGORIES


ROOT = Path(__file__).resolve().parents[2]
env_path = ROOT / ".env"
load_dotenv(env_path)
cred_path = ROOT / os.getenv("SERVICE_ACCOUNT_KEY")
os.environ["GOOGLE_CLOUD_PROJECT"] = os.getenv("GOOGLE_CLOUD_PROJECT")
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = os.getenv("GOOGLE_GENAI_USE_VERTEXAI")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cred_path)
EXPERIMENT_NAME = "multiAgent_Miss+FlipLandMark_wf__v1"
ORACLE_SCANS_CACHE = 'synthetic_oracle_scans'
MODEL_NAME = "gemini-2.5-flash"
PERT_MISSING_LANDMARK= True 
PERT_FLIP_LANDMARK = True
ORACLE_CACHE_LOCK = asyncio.Lock()
PROFILE_CACHE_LOCK = asyncio.Lock()

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
    result = await safe_run_node( ctx,
    eval_single_agent,
    primary_content)

    #print("PRIMARY RESULT =>", result)
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

    oracle_descriptions = []
    pertubation_used_labels = []
    for oracle_scan in oracle_dataset:

        await asyncio.sleep(1)
        scan_name = oracle_scan["scan"]
        # identifica in maniera univoca la perturbazione
        oracle_scan_key = oracle_scan["pred_scan_conf_name"]

        async with ORACLE_CACHE_LOCK:

            oracle_cache = load_oracle_cache(ORACLE_SCANS_CACHE)
            cached = oracle_cache.get(oracle_scan_key,None,)
        if cached is not None:

            cache_status = cached.get("status",None)
            if cache_status == "oracle_correct":
                continue
            if cache_status == "oracle_failure":
                oracle_descriptions.append(cached["description"])
                pertubation_used_labels.append(oracle_scan["pred_scan_conf_name"])
                continue
        prompt = build_user_request_for_scan(
            oracle_scan,
            example_items,
            goal=True,
        )
        image_paths = (
            [item["image_pred"] for item in example_items["lvl5"]]
            + [item["image_pred"] for item in example_items["lvl4"]]
            + [item["image_pred"] for item in example_items["lvl3"]]
            + [item["image_pred"] for item in example_items["lvl2"]]
            + [item["image_pred"] for item in example_items["lvl1"]]
            + [oracle_scan["pred_image_path"]]
        )
        oracle_content = build_multimodal_prompt(
            text=prompt,
            image_paths=image_paths,
        )

        #VALUTAZIONE della scan perturbata
        prediction = await safe_run_node(
            ctx,
            eval_single_agent,
            oracle_content,
            max_retries=10,
            base_delay=10,
        )
        if int(prediction["quality"]) == int(oracle_scan["pred_quality_score"]):
            print("### prediction quality uguale alla ground truth. Cache & skip ####")
            cache_obj = {
                "status": "oracle_correct",
                "predicted_quality": prediction["quality"],
                "ground_truth_quality": oracle_scan["pred_quality_score"],
                "predicted_motivation":prediction["motivation"],
            }
            async with ORACLE_CACHE_LOCK:
                oracle_cache = load_oracle_cache(ORACLE_SCANS_CACHE)

                oracle_cache[ oracle_scan_key] = cache_obj
                save_oracle_cache( ORACLE_SCANS_CACHE,oracle_cache)
            continue
        #caso errato, viene effettuata analisi dellerrore
        pertubation_used_labels.append(oracle_scan["pred_scan_conf_name"])

        meta_prompt = (
            build_oracle_error_descriptor_agent_prompt(
                oracle_scan,
                prediction,
                example_items,
            )
            )

        meta_content = build_multimodal_prompt(
            text=meta_prompt,
            image_paths=image_paths,
        )

        meta_description = await safe_run_node(
            ctx,
            oracle_error_descriptor_agent,
            meta_content,
            max_retries=10,
            base_delay=10,
        )

        description_obs = {
            "scan":oracle_scan_key,
            #qualità reale post perturbazione
            "ground_truth_quality":oracle_scan["pred_quality_score"],
            #qualità predetta dal valutatore
            "predicted_quality":  prediction["quality"],
            "predicted_motivation": prediction["motivation"],
            "failure_analysis": meta_description["failure_analysis"],
            #descrizione perturbazione
            "perturbation": oracle_scan["pred_metadata"],
            #qualità originale
            "origin_quality_before_pertubation": oracle_scan["origin_pred_quality_score"],
        }

        cache_obj = {
            "status": "oracle_failure",
            "description":description_obs,
        }

        async with ORACLE_CACHE_LOCK:

            oracle_cache = load_oracle_cache(ORACLE_SCANS_CACHE )
            oracle_cache[oracle_scan_key] = cache_obj
            save_oracle_cache( ORACLE_SCANS_CACHE, oracle_cache,)
        oracle_descriptions.append(description_obs)

    yield Event(
        output=node_input,
        state={
            "oracle_result": {
                "perturbation_labels":pertubation_used_labels,
                "primary_prediction":ctx.state["primary_prediction"],
                "oracle_descriptions":oracle_descriptions,
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
            "biases": [],
            "profile_summary":
                "Evaluator predictions matched all available oracle examples."
        }
    else:
        profile_cache_key=build_profile_cache_key(descriptions,'scan')
        async with PROFILE_CACHE_LOCK:
            oracle_cache = load_oracle_cache(EXPERIMENT_NAME)
            cached_profile = oracle_cache.get(profile_cache_key,None)
        if cached_profile is not None:
            #print('### usando cache ####')
            oracle_profile = cached_profile
        else: 
            #print('### profile senza cache###')
            profile_input = { 
                "oracle_error_descriptions": descriptions}
            #print("### invoking profile builder ###")
            oracle_profile=await safe_run_node(
                ctx,
                oracle_profile_builder_agent,
                profile_input,
                max_retries=10,
                base_delay=10,
                )
            #print("### profile builder completed ###")
            async with PROFILE_CACHE_LOCK:
                oracle_cache = load_oracle_cache(EXPERIMENT_NAME)
                oracle_cache[profile_cache_key] = oracle_profile
                oracle_cache['last_profile'] = profile_cache_key
                save_oracle_cache( EXPERIMENT_NAME,oracle_cache)
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
    perturbation_labels = ctx.state['oracle_result']['perturbation_labels']
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
    print("final prompt chars =",len(prompt))
    final_prediction = await safe_run_node(
    ctx,
    final_decision_agent,
    final_content,
    max_retries=10,
    base_delay=10,
    )

    #print("FINAL RESULT =>", final_prediction)

    yield Event(
    output={
        "scan":scan_item["scan"],
        "final_prediction": final_prediction,
        "n_oracle_examples":len(oracle_error_descr_per_scan),
        "perturbation_labels": perturbation_labels,
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


async def run_agentic_workflow(
    dataset_primary,
    WORKFLOW_RESOURCES,
):
    from google.adk.plugins import LoggingPlugin
    from google.adk.plugins import DebugLoggingPlugin

    plugins = [
        LoggingPlugin(),
        DebugLoggingPlugin()]

    app = App(
        name="landmark_quality_app_v1",
        root_agent=root_agent,
        # plugins=plugins
    )

    runner = InMemoryRunner(app=app)
    semaphore = asyncio.Semaphore(1)
    tasks = []
    total = len(dataset_primary)

    print(
        f"\n###### START WORKFLOW "
        f"({total} scans da valutare)\n")

    start_time = time.time()

    for i, item in enumerate(dataset_primary):

        tasks.append(
            asyncio.create_task(
                eval_scan_task(
                    item,
                    WORKFLOW_RESOURCES,
                    runner,
                    app.name,
                    semaphore)
                    ))
        if i % 10 == 0 and i > 0:
            await asyncio.sleep(2)
    outputs = []
    completed = 0
    for task in asyncio.as_completed(tasks):

        result = await task
        outputs.append(result)
        completed += 1
        elapsed = time.time() - start_time
        avg_time = elapsed / completed
        remaining = total - completed
        eta = avg_time * remaining

        print(
            f"[PROGRESS] "
            f"{completed}/{total} scans completate | "
            f"elapsed={elapsed/60:.1f} min | "
            f"ETA={eta/60:.1f} min"
        )
    total_time = time.time() - start_time

    print(
        f"\n###### WORKFLOW COMPLETATO "
        f"({total} scans)\n"
        f"Tempo totale: "
        f"{total_time/60:.1f} minuti\n"
    )
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


    #Definizione pertubazioni da applicare 
    PERTURBATIONS=[] 
    #landmark mancanti
    miss_landmark_configs_par={
    "global_cfg":{'frac':[0.05,0.1]},
    "local_cfg":{
        'cls':["Mesial",
             "Distal",
            "InnerPoint",
            "OuterPoint"], 
            'frac':[0.2,0.3,0.6]}}
    #flip della classe dei landmark 
    flip_landmark_configs_par={
    "global_cfg":{'frac':[0.05]},
    "local_cfg":{
        'cls':["Mesial",
                "Distal",
            "InnerPoint",
            "OuterPoint"], 
            'frac':[0.2,0.3,0.5]}}
    #identifica con una label la configurazione scelta
    miss_landmark_config_label = build_perturbation_label(miss_landmark_configs_par)
    flip_landmark_config_label = build_perturbation_label(flip_landmark_configs_par)

    miss_landmark_configs_list= build_missing_landmark_configs(
        global_config=miss_landmark_configs_par['global_cfg'],
        local_config= miss_landmark_configs_par['local_cfg'])

    class_flip_configs_list=build_class_flip_configs(
    global_config=flip_landmark_configs_par['global_cfg'],
    local_config= flip_landmark_configs_par['local_cfg'])
    if PERT_MISSING_LANDMARK:
        PERTURBATIONS.extend(miss_landmark_configs_list)
    if PERT_FLIP_LANDMARK: 
        PERTURBATIONS.extend(class_flip_configs_list)

    #dataset sintetico a partire dalle scan in oracle_pool 
    oracle_pool_scans= [v[0] for k,v in oracle_pool.items()]
    oracle_dataset=build_synthetic_oracle_dataset(oracle_pool_scans, SCANS, GT_ROOT, PRED_CSV,SCREENSHOT_DIR,pertubation_configs=PERTURBATIONS)
    print(f'###### ORACLE DATASET costruito con {len(oracle_dataset)} scansioni')
    EXECUTE_WF = True 
    if EXECUTE_WF:
        #passa risorse che saranno usati per inizializzare lo stato
        WORKFLOW_RESOURCES = {
        "example_items": example_items,
        "oracle_dataset": oracle_dataset
        }
        outputs = await run_agentic_workflow(dataset_primary, WORKFLOW_RESOURCES)

        evaluation = evaluate_agent(outputs)

        config = {
            "architecture": EXPERIMENT_NAME,
            "mdifiche_fatte_rispetto_a_versione_precedente": 'prima versione di flip only. si usa già concetto bias nel prompt (v3 di only miss_landmrk)',
            "model": MODEL_NAME,
            "primary_examples": primary_examples,
            "oracle_example_len": len(oracle_dataset), 
            "oracle_example_scans": oracle_pool_scans,
            "pertubation_used": 
            {
                "missing_landmarks": [] if PERT_MISSING_LANDMARK==False else miss_landmark_config_label, 
                "flip_landmarks": [] if PERT_FLIP_LANDMARK==False else flip_landmark_config_label
            }
        }

        exp_dir = save_experiment(config, outputs, evaluation)
        print("Esperimento salvato in:", exp_dir)


if __name__ == "__main__":
    asyncio.run(main())
