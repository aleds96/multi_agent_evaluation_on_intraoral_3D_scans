
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
    {oracle_scan["pred_quality_score"]}

    Evaluator motivation:
    {prediction["motivation"]}

    ----------------------------------------
    INFORMAZIONI SULLA PERTURBAZIONE

    Original quality:
    {oracle_scan["origin_pred_quality_score"]}

    Perturbed quality:
    {oracle_scan["pred_quality_score"]}

    Quality drop:
    {
        oracle_scan["origin_pred_quality_score"]
        -
        oracle_scan["pred_quality_score"]
    }

    Perturbation metadata:
    {oracle_scan["pred_metadata"]}

    ----------------------------------------
    METRICHE QUANTITATIVE

    Original mAP:
    {round(float(oracle_scan["origin_pred_gmap"]), 3)}

    Perturbed mAP:
    {round(float(oracle_scan["pred_gmap"]), 3)}

    mAP drop:
    {
        round(
            float(oracle_scan["origin_pred_gmap"])
            -
            float(oracle_scan["pred_gmap"]),
            3
        )
    }

    ----------------------------------------
    COUNT LANDMARK PER CLASSE

    Original counts:
    {oracle_scan["origin_pred_count_per_class"]}

    Perturbed counts:
    {oracle_scan["pred_count_per_class"]}

    ----------------------------------------
    QUALITY PER CLASSE

    Original:
    {oracle_scan["origin_pred_quality_per_class"]}

    Perturbed:
    {oracle_scan["pred_quality_per_class"]}

    ----------------------------------------
    CONTESTO

    Questa scansione è stata ottenuta applicando
    una perturbazione controllata ai landmark
    della predizione originale.

    La perturbazione rappresenta la causa nota
    del degrado osservato.

    Lo scopo non è analizzare la scansione in sé,
    ma capire quali debolezze del valutatore
    sono emerse quando è stato esposto a questa
    perturbazione.

    ----------------------------------------
    COMPITO

    Non rivalutare la scan.

    Non cercare di indovinare quale perturbazione
    sia stata applicata: la perturbazione è già nota.

    Analizza perché il valutatore ha prodotto
    una qualità diversa dalla ground truth.

    Concentrati su:

    - bias del valutatore
    - landmark coinvolti
    - effetto della perturbazione
    - aspetti visivi fuorvianti
    - motivazioni corrette
    - motivazioni errate o incomplete
    - segnali ignorati
    - differenze tra qualità originale e perturbata
    - differenze tra mAP originale e perturbato

    Valuta anche se la risposta del valutatore
    è proporzionata al reale degrado introdotto.

    Ad esempio evidenzia se il valutatore:

    - ha sovrastimato la qualità
    - ha sottostimato la qualità
    - ha individuato il problema corretto ma con gravità errata
    - ha ignorato una riduzione significativa dei landmark
    - si è fatto influenzare da segnali visivi fuorvianti

    Non limitarti a descrivere la perturbazione.

    L'obiettivo è identificare quale vulnerabilità
    o failure mode del valutatore è stata evidenziata
    da questo esempio.

    Restituisci una breve analisi diagnostica
    (2-6 frasi).
    """

    return meta_prompt
instruction_error_description_agent_prompt = """
Sei un analista incaricato di studiare i limiti di un valutatore automatico
della qualità dei landmark dentali.

CONTESTO

Per costruire un profilo di affidabilità del valutatore,
alcune predizioni sono state modificate artificialmente tramite
perturbazioni controllate.

Le perturbazioni simulano errori realistici sui landmark
(ad esempio landmark mancanti, classi scambiate,
spostamenti spaziali o altre alterazioni).

Per ogni caso avrai accesso a:

- la stessa scala qualitativa utilizzata dal valutatore
- gli stessi esempi few-shot
- l'immagine perturbata
- statistiche quantitative sulla predizione
- il tipo di perturbazione applicata
- la qualità originale prima della perturbazione
- la qualità reale dopo la perturbazione
- la qualità predetta dal valutatore
- la motivazione prodotta dal valutatore

OBIETTIVO

Il tuo compito NON è assegnare una nuova qualità.

Il tuo compito è identificare quali limiti,
bias o pattern di errore del valutatore sono emersi
a causa della perturbazione applicata.

Analizza:

- quali aspetti della perturbazione hanno tratto in inganno il valutatore
- quali landmark o classi di landmark sono coinvolti
- quali segnali visivi sono stati ignorati
- quali segnali visivi sono stati sovrastimati
- quali parti della motivazione del valutatore sono corrette
- quali parti della motivazione del valutatore sono errate, incomplete o fuorvianti
- quali pattern di errore potrebbero ripresentarsi in casi simili
- se il valutatore ha reagito in modo proporzionato al reale degrado introdotto
- se il valutatore ha sottostimato o sovrastimato l'impatto della perturbazione

IMPORTANTE

La perturbazione rappresenta la causa nota del degrado.

Non cercare di indovinare cosa sia successo alla scan.

Usa le informazioni disponibili sulla perturbazione,
sui conteggi dei landmark, sulle qualità per classe
e sulle metriche quantitative (mAP) per spiegare
il comportamento del valutatore.

Valuta se l'errore del valutatore è coerente
con l'entità reale del degrado osservato.

Un piccolo calo del mAP indica una perturbazione lieve.

Un forte calo del mAP indica una perturbazione significativa.

Se il valutatore non reagisce in modo proporzionato
al degrado reale, evidenzialo chiaramente nell'analisi.

OUTPUT

Produci una breve analisi diagnostica (2-6 frasi).

L'analisi deve essere focalizzata sui limiti del valutatore
e non sulla descrizione generale della scansione.
"""
instruction_profile_builder_agent_prompt = """
Sei incaricato di costruire un profilo di affidabilità
di un valutatore della qualità dei landmark dentali.

CONTESTO

Riceverai una collezione di osservazioni generate
tramite perturbazioni controllate applicate a scansioni dentali.

Ogni osservazione descrive:

- la qualità reale dopo la perturbazione
- la qualità predetta dal valutatore
- la motivazione del valutatore
- il tipo di perturbazione applicata
- un'analisi delle cause dell'errore del valutatore

Le perturbazioni sono state introdotte deliberatamente
per evidenziare limiti e vulnerabilità del valutatore.

OBIETTIVO

Il tuo compito non è analizzare le singole scansioni.

Il tuo compito è identificare pattern ricorrenti
nel comportamento del valutatore.

Costruisci un profilo che descriva:

- punti di forza del valutatore
- punti di debolezza del valutatore
- failure mode ricorrenti
- bias sistematici
- situazioni nelle quali il valutatore tende
  a sovrastimare o sottostimare la qualità

IMPORTANTE

Concentrati soprattutto sui pattern osservati
in più esempi diversi.

Non riportare dettagli specifici di una singola scansione.

Non limitarti a ripetere le failure analysis ricevute.

Generalizza le osservazioni in comportamenti
stabili del valutatore.

Ignora failure mode isolati che compaiono
in un solo esempio.

Privilegia invece failure mode che emergono
in modo consistente in esempi multipli.

OUTPUT

- strengths: massimo 2 elementi
- weaknesses: massimo 2 elementi
- failure_modes: massimo 3 elementi
- profile_summary: massimo 3 frasi

Le voci devono descrivere comportamenti generali
del valutatore e non casi specifici del dataset.
"""
instruction_final_decision_agent_prompt = """
Sei un esperto nella valutazione della qualità dei landmark dentali.

Il tuo compito è produrre una valutazione finale calibrata.

Potresti ricevere:

- immagini contenenti landmark dentali
- esempi few-shot
- statistiche quantitative
- una valutazione precedente
- un profilo di affidabilità del valutatore

Utilizza tutte le informazioni disponibili per produrre la stima finale più affidabile.

Puoi confermare la valutazione precedente oppure modificarla se il profilo di affidabilità suggerisce la presenza di un errore noto.

Restituisci esclusivamente un oggetto JSON nel formato:

{
    "quality": int,
    "motivation": str
}

La qualità deve essere un numero intero compreso tra 1 e 5.

La motivazione deve essere breve e concentrarsi sugli elementi principali che giustificano la decisione finale.
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
        goal=False,
    )

    return f"""
{base_prompt}

--------------------------------------------------

VALUTAZIONE PRIMARIA

Qualità predetta:
{primary_prediction["quality"]}

Motivazione:
{primary_prediction["motivation"]}

--------------------------------------------------

PROFILO DI AFFIDABILITÀ DEL VALUTATORE

Riassunto:
{oracle_profile["profile_summary"]}

Punti di forza osservati:
{oracle_profile["strengths"]}

Punti di debolezza osservati:
{oracle_profile["weaknesses"]}

Modalità di errore osservate:
{oracle_profile["failure_modes"]}

--------------------------------------------------

CONTESTO

Il profilo di affidabilità è stato costruito
analizzando la risposta del valutatore a numerose
perturbazioni controllate dei landmark.

Le debolezze e i failure mode riportati
rappresentano vulnerabilità osservate
sperimentalmente del valutatore.

--------------------------------------------------

COMPITO

Valuta la scansione di input.

Parti dalla valutazione primaria.

Successivamente verifica se la scansione corrente
presenta caratteristiche compatibili con
uno o più failure mode del valutatore.

In particolare chiediti:

- il valutatore potrebbe stare ignorando landmark importanti?
- il valutatore potrebbe sovrastimare la qualità?
- il valutatore potrebbe sottostimare la qualità?
- sono presenti pattern simili a quelli osservati nel profilo Oracle?
- i punti deboli identificati nel profilo sono applicabili al caso corrente?

Utilizza:

- l'immagine target
- gli esempi few-shot
- le statistiche quantitative
- la valutazione primaria
- il profilo di affidabilità

Modifica la valutazione primaria
solo se esistono evidenze concrete
che uno o più failure mode individuati
dal profilo Oracle siano pertinenti
alla scansione corrente.

In assenza di evidenze sufficienti
mantieni la valutazione primaria.

L'obiettivo è produrre
la migliore stima finale possibile.

Restituisci esclusivamente un JSON:

{{
    "quality": integer compreso tra 1 e 5,
    "motivation": "breve spiegazione"
}}
"""