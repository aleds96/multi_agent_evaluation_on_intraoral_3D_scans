
def _format_profile(p):
    return (
        f"- Scan: {p['scan']}\n"
        f"  Qualità globale: {p['quality_global']}\n"
        f"  Qualità per classe: {p['quality_per_class']}\n"
        f"  AP grezzo per classe: {p['raw_per_class']}\n"
        f"  Best class: {p['best_class']}\n"
        f"  Worst class: {p['worst_class']}\n"
        f"  Stabilità: {p.get('stability', 'N/A')}\n"
    )
def format_group_counts(count_per_group_class):
    out = []
    for group, cls_counts in count_per_group_class.items():
        out.append(f"{group}:")
        for cls, c in cls_counts.items():
            out.append(f"  - {cls}: {c}")
        out.append("") 
    return "\n".join(out)

instruction_error_description_agent_prompt="""
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
instruction_single_agent_prompt = """
Sei un agente specializzato nella valutazione della qualità dei landmark dentali su scansioni 3D intraorali. 
Il tuo compito è analizzare esclusivamente l’immagine di input fornita dall’utente, confrontandola con gli esempi 
e con i loro profili di qualità.

## Definizione della qualità
La qualità assegnata a una scan è un indice discreto da 1 a 5, ottenuto tramite discretizzazione dei quantili 
della distribuzione reale dell’Average Precision (AP) per scan:
- qualità 1 → AP nel 10° percentile (peggiore)
- qualità 2 → AP nel 25° percentile
- qualità 3 → AP intorno alla mediana
- qualità 4 → AP nel 75° percentile
- qualità 5 → AP nel 90° percentile (migliore)

Il tuo obiettivo è **stimare il livello di qualità che approssima l’AP sottostante**, basandoti esclusivamente 
sull’analisi visiva dei landmark e sulla coerenza anatomica.

## Classi dei landmark (colori)
- Mesial (rosso): verso il centro dell’arcata
- Distal (verde): verso l’esterno dell’arcata
- Cusp (blu): cuspidi, presenti solo su premolari e molari (2–5 cuspidi)
- InnerPoint (giallo): lato linguale/palatale
- OuterPoint (ciano): lato buccale
- FacialPoint (magenta): centro della superficie vestibolare

## Gruppi dentali (colori)
- Incisivi (grigio)
- Canini (arancione‑pesca)
- Premolari (rosa pastello)
- Molari (azzurro pastello)
- Gengiva (bianco)

## Regole anatomiche e difficoltà note
- Le cuspidi compaiono solo su premolari e molari; sono assenti su incisivi e canini.
- Mesial e Distal devono trovarsi sulle superfici mesiale/distale.
- InnerPoint e OuterPoint devono rispettare la distinzione linguale/palatale vs buccale.
- Il numero di landmark per classe varia per tipo di dente (es. più cuspidi nei molari).
- Landmark duplicati, mancanti o fuori dalla superficie del dente indicano errori.
- Errori di orientamento (Mesial↔Distal, Inner↔Outer) sono frequenti in denti ruotati o inclinati.
- La simmetria tra denti omologhi è un indicatore di qualità.
- La distribuzione dei landmark deve seguire la morfologia del dente.
- Le differenze rispetto agli esempi di qualità alta e bassa aiutano a stimare il livello di qualità 
  (e quindi l’AP sottostante).

## Cosa devi valutare
Valuta la qualità dell’immagine di input considerando:
- correttezza anatomica dei landmark
- coerenza tra classe e posizione sul dente
- plausibilità anatomica
- landmark fuori dente o in zone impossibili
- errori di orientamento
- duplicazioni o assenze di landmark attesi
- simmetria e distribuzione
- differenze rispetto agli esempi
- coerenza con i profili numerici degli esempi

## Output richiesto
Restituisci un JSON con:
{
  "quality": X,
  "motivation": "testo breve"
}

Non descrivere le immagini degli esempi. Non proporre correzioni. Concentrati solo sull’immagine di input.
"""

def build_user_request_for_scan(input_info, examples,use_profile=False,input_stat=True,goal=True):

    lvl5_profiles = "\n".join(_format_profile(p['profile']) for p in examples["lvl5"])
    lvl4_profiles = "\n".join(_format_profile(p['profile']) for p in examples["lvl4"])
    lvl3_profiles = "\n".join(_format_profile(p['profile']) for p in examples["lvl3"])
    lvl2_profiles = "\n".join(_format_profile(p['profile']) for p in examples["lvl2"])
    lvl1_profiles = "\n".join(_format_profile(p['profile']) for p in examples["lvl1"])

    pred_counts_global_str = "\n".join(
        f"- {cls}: {input_info['pred_count_per_class'][cls]}" for cls in input_info['pred_count_per_class']
    )
    #group_counts_str = format_group_counts(input_info["pred_count_per_group_class"])
    profile_prompt = f""" 
    ## Profili degli esempi
    ### Esempio qualità 5
    {lvl5_profiles}
    ### Esempio qualità 4
    {lvl4_profiles}
    ### Esempio qualità 3
    {lvl3_profiles}

    ### Esempio qualità 2
    {lvl2_profiles}

    ### Esempio qualità 1
    {lvl1_profiles}
    
    """

    input_stat_prompt=f""" 
    ## Statistiche quantitative sull'immagine di input da valutare
        ### Conteggio dei landmark per classe
        {pred_counts_global_str}

    """

    img_prompt ="""
    ## Immagini fornite (in ordine)
    1. Esempio qualità 5
    2. Esempio qualità 4
    3. Esempio qualità 3
    4. Esempio qualità 2
    5. Esempio qualità 1
    6. Immagine di input da valutare
    """ 
    goal_prompt="""
    Valuta solo l’immagine di input (ultima fornita), confrontandola con gli esempi e con i loro profili.
    """
    final_prompt = ''
    if use_profile: 
        final_prompt+=profile_prompt
    if input_stat: 
        final_prompt+=input_stat_prompt
    final_prompt+=img_prompt
    if goal: 
        final_prompt+=goal_prompt
    return final_prompt
def build_oracle_error_descriptor_agent_prompt(oracle_scan,prediction,example_items): 
    base_prompt = build_user_request_for_scan(
    oracle_scan,
    example_items,
    goal=False)
    meta_prompt = f"""
            {base_prompt}
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
    return meta_prompt
instruction_profile_builder_agent_prompt="""
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
instruction_final_decision_agent_prompt="""
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