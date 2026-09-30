
def format_oracle_descriptions(descriptions):
    chunks = []

    for i, d in enumerate(descriptions, start=1):
        chunks.append(f"""
        CASO ORACLE {i}

        Perturbazione:
        {d["perturbation"]}

        Qualità reale:
        {d["ground_truth_quality"]}

        Qualità predetta:
        {d["predicted_quality"]}

        Motivazione del valutatore:
        {d["predicted_motivation"]}

        Analisi dell'errore:
        {d["failure_analysis"]}

        Qualità originale prima della perturbazione:
        {d["origin_quality_before_pertubation"]}
        """)

    return "\n\n-------------------------\n".join(chunks)
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
Sei incaricato di costruire un profilo di calibrazione
di un valutatore della qualità dei landmark dentali.

CONTESTO

Riceverai una collezione di osservazioni ottenute
tramite perturbazioni controllate applicate a scansioni dentali.

Ogni osservazione contiene informazioni strutturate sul comportamento
del valutatore in un caso di errore.

Le osservazioni possono includere:

- qualità reale dopo la perturbazione
- qualità predetta dal valutatore
- motivazione prodotta dal valutatore
- qualità originale prima della perturbazione
- descrizione della perturbazione applicata
- analisi delle cause dell'errore del valutatore

Le perturbazioni sono state introdotte deliberatamente
per evidenziare vulnerabilità e bias del valutatore.

IMPORTANTE

Le osservazioni fornite rappresentano esclusivamente
casi nei quali il valutatore ha prodotto una valutazione
differente dalla qualità reale.

I casi corretti non vengono forniti.

Di conseguenza:

- il profilo risultante descrive principalmente
  vulnerabilità e pattern di errore osservati;

- il profilo non rappresenta una misura completa
  dell'affidabilità globale del valutatore;

- i punti di forza devono essere dedotti soltanto
  da capacità che il valutatore continua a mostrare
  anche nei casi in cui commette errori.

OBIETTIVO

Il tuo compito NON è riassumere le singole osservazioni.

Il tuo compito è identificare pattern ricorrenti
e trasformarli in bias utilizzabili da un agente calibratore.

Il calibratore utilizzerà il profilo per decidere:

- quando fidarsi della valutazione primaria;
- quando sospettare la presenza di un errore;
- in quale direzione il valutatore tende a sbagliare;
- quali condizioni tendono ad attivare il bias.

UTILIZZA TUTTE LE INFORMAZIONI DISPONIBILI

Non basarti esclusivamente sul campo failure_analysis.

Utilizza congiuntamente:

- tipo di perturbazione
- qualità reale
- qualità predetta
- differenza tra qualità reale e predetta
- motivazione del valutatore
- analisi delle cause dell'errore

per identificare pattern robusti.

BIAS

Un bias rappresenta un comportamento ricorrente
osservato in più esempi.

Ogni bias deve essere sufficientemente generale
da poter essere applicato a scansioni non viste.

Per ciascun bias identifica:

- cosa attiva il bias
- come si manifesta
- in quale direzione tende a produrre errori
- quale effetto produce tipicamente
- quali esempi supportano la sua esistenza

GENERALIZZAZIONE

Non limitarti a ripetere le osservazioni ricevute.

Non riportare descrizioni dettagliate
di singole scansioni.

Generalizza le evidenze osservate
in pattern stabili del comportamento del valutatore.

Privilegia pattern che emergono
in più perturbazioni differenti.

Ignora fenomeni che compaiono
in un singolo esempio isolato.

ERROR_DIRECTION

Utilizza esclusivamente uno dei seguenti valori:

- overestimation
- underestimation
- hallucinated_explanation
- mixed

Dove:

overestimation
=
il valutatore tende a sovrastimare la qualità.

underestimation
=
il valutatore tende a sottostimare la qualità.

hallucinated_explanation
=
il problema principale riguarda motivazioni
inventate o non supportate dalle evidenze.

mixed
=
il bias si presenta in modi differenti.

SUPPORING_EXAMPLES

Gli esempi devono essere sintetici.

Utilizza preferibilmente il nome della perturbazione.

Esempi validi:

- "Mesial 20%"
- "Distal 20%"
- "OuterPoint 60%"

Non copiare intere failure analysis.

OUTPUT

Genera:

- strengths
- weaknesses
- biases
- profile_summary

STRENGTHS

Massimo 3 elementi.

Descrivono capacità che il valutatore mantiene
anche nei casi di errore.

WEAKNESSES

Massimo 3 elementi.

Descrivono vulnerabilità ricorrenti.

BIASES

Massimo 5 bias.

Ogni bias deve contenere:

- bias_name
- description
- error_direction
- typical_effect
- trigger_conditions
- supporting_examples

PROFILE_SUMMARY

Massimo 3 frasi.

Riassumi le principali vulnerabilità osservate.

OBIETTIVO FINALE

Costruisci un profilo sufficientemente concreto
da permettere a un agente calibratore di verificare
se una nuova scansione presenta condizioni compatibili
con uno dei bias osservati e decidere se mantenere
o modificare la valutazione primaria.
"""
instruction_final_decision_agent_prompt = """
Sei un esperto nella valutazione della qualità dei landmark dentali.

Il tuo compito è produrre una valutazione finale calibrata.

Riceverai una combinazione delle seguenti informazioni:

- immagini contenenti landmark dentali
- esempi few-shot
- statistiche quantitative
- una valutazione primaria prodotta da un altro valutatore
- un profilo di calibrazione costruito analizzando errori osservati in precedenti perturbazioni sintetiche

OBIETTIVO

Il tuo ruolo non è eseguire una nuova valutazione indipendente da zero.

Il tuo ruolo è analizzare criticamente la valutazione primaria
e decidere se mantenerla oppure correggerla.

Il profilo di calibrazione descrive bias e vulnerabilità
osservati sperimentalmente nel valutatore primario.

Ogni bias può includere:

- una descrizione del pattern osservato
- la direzione tipica dell'errore
- l'effetto tipico prodotto dal bias
- condizioni che tendono ad attivarlo
- esempi che hanno generato il bias

UTILIZZO DEL PROFILO

Utilizza il profilo come una base di conoscenza
sui comportamenti storicamente osservati del valutatore.

I bias NON sono regole deterministiche.

La semplice presenza di un bias nel profilo
non implica che tale bias sia attivo nella scansione corrente.

Prima di modificare la valutazione primaria devi verificare che:

- la scansione corrente sia compatibile con le condizioni di attivazione del bias
- l'immagine supporti la possibile correzione
- le statistiche quantitative supportino la possibile correzione
- la motivazione fornita dal valutatore mostri segnali coerenti con il bias osservato

DECISIONE

Mantieni la valutazione primaria quando:

- non esistono evidenze sufficienti di errore
- il profilo non è pertinente al caso corrente
- l'immagine e le statistiche supportano la stima primaria

Modifica la valutazione primaria soltanto quando:

- uno o più bias osservati risultano compatibili con il caso corrente
- esistono evidenze concrete che il valutatore stia commettendo un errore noto

In caso di dubbio privilegia la valutazione primaria.

L'obiettivo è produrre la stima finale più affidabile possibile.

OUTPUT

Restituisci esclusivamente un oggetto JSON nel formato:

{
    "quality": int,
    "motivation": str
}

Dove:

- quality è un numero intero tra 1 e 5
- motivation è una spiegazione breve che descrive
  le ragioni principali della decisione finale

Non aggiungere testo fuori dal JSON.
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

PROFILO DI CALIBRAZIONE DEL VALUTATORE

Riassunto:
{oracle_profile["profile_summary"]}

Punti di forza osservati:
{oracle_profile["strengths"]}

Punti di debolezza osservati:
{oracle_profile["weaknesses"]}

Bias osservati:
{oracle_profile["biases"]}

--------------------------------------------------

CONTESTO

Il profilo è stato costruito analizzando
numerose perturbazioni sintetiche applicate
ai landmark dentali.

Le perturbazioni sono state utilizzate per
identificare pattern ricorrenti di errore,
vulnerabilità e bias del valutatore.

Ogni bias contiene:

- una descrizione del pattern osservato
- la direzione tipica dell'errore
- gli effetti normalmente prodotti
- le condizioni che tendono ad attivarlo
- esempi concreti che hanno generato il bias

I bias rappresentano conoscenza empirica
sui comportamenti osservati del valutatore.

--------------------------------------------------

IMPORTANTE

Il profilo è stato costruito utilizzando
esclusivamente casi nei quali il valutatore
ha commesso errori rispetto alla ground truth.

Pertanto il profilo descrive vulnerabilità
osservate ma non rappresenta una misura completa
dell'affidabilità complessiva del valutatore.

La presenza di un bias nel profilo non implica
che il bias sia necessariamente attivo
nella scansione corrente.

Utilizza i bias come evidenze di calibrazione
e non come regole deterministiche.

Non modificare la valutazione primaria
soltanto perché un bias esiste.

Prima verifica che nella scansione corrente
siano presenti condizioni compatibili con:

- trigger_conditions
- description
- typical_effect
- error_direction

descritti dal bias.

--------------------------------------------------

COMPITO

Valuta la scansione di input.

Parti dalla valutazione primaria.

Successivamente verifica se la scansione corrente
presenta caratteristiche compatibili con uno
o più bias osservati nel profilo.

Per ogni bias chiediti:

- il pattern descritto è applicabile a questa scansione?
- le condizioni di attivazione del bias sono presenti?
- la motivazione fornita dal valutatore presenta segnali
  compatibili con il bias?
- la direzione dell'errore suggerita dal bias è coerente
  con il caso corrente?
- l'immagine e le statistiche quantitative supportano
  una correzione della valutazione primaria?

Utilizza:

- l'immagine target
- gli esempi few-shot
- le statistiche quantitative
- la valutazione primaria
- il profilo di calibrazione

Modifica la valutazione primaria soltanto
se esistono evidenze concrete che uno o più
bias osservati siano realmente applicabili
alla scansione corrente.

In assenza di evidenze sufficienti mantieni
la valutazione primaria.

L'obiettivo è produrre la stima finale più
affidabile possibile.

Restituisci esclusivamente un JSON:

{{
    "quality": integer compreso tra 1 e 5,
    "motivation": "breve spiegazione"
}}
"""
def build_final_noprofile_review_prompt(
    scan_item,
    example_items,
    primary_prediction,
    error_descriptions,
):

    base_prompt = build_user_request_for_scan(
        scan_item,
        example_items,
        goal=False,
    )
    oracle_examples_str = format_oracle_descriptions(
    error_descriptions)

    return f"""
{base_prompt}

--------------------------------------------------

VALUTAZIONE PRIMARIA

Qualità predetta:
{primary_prediction["quality"]}

Motivazione:
{primary_prediction["motivation"]}

--------------------------------------------------

CASI ORACLE

Di seguito sono riportati esempi sintetici perturbati
e i relativi errori commessi dal valutatore.

Ogni caso contiene:

- la perturbazione applicata
- la qualità reale dopo la perturbazione
- la qualità predetta dal valutatore
- la motivazione fornita dal valutatore
- un'analisi esplicita dell'errore osservato

{oracle_examples_str}

--------------------------------------------------

IMPORTANTE

I casi Oracle rappresentano esempi concreti di errori
commessi dal valutatore in situazioni controllate.

Non tutti gli errori osservati nei casi Oracle sono
necessariamente presenti nella scansione corrente.

Utilizza tali casi come esempi di riferimento e non
come regole deterministiche.

Non modificare la valutazione primaria semplicemente
perché esiste un caso Oracle simile.

Verifica sempre che l'immagine corrente, le statistiche
quantitative e la motivazione del valutatore supportino
realmente una revisione della valutazione.

--------------------------------------------------

COMPITO

Valuta la scansione di input.

Parti dalla valutazione primaria.

Successivamente analizza i casi Oracle e verifica se
il valutatore potrebbe aver commesso errori simili
anche nella scansione corrente.

Per ogni caso Oracle chiediti:

- l'errore osservato è plausibilmente applicabile
  alla scansione corrente?

- la motivazione del valutatore presenta segnali
  compatibili con quelli osservati nel caso Oracle?

- l'immagine e le statistiche quantitative suggeriscono
  che il valutatore possa aver ripetuto un errore analogo?

- esistono evidenze sufficienti per correggere
  la valutazione primaria?

Utilizza:

- l'immagine target
- gli esempi few-shot
- le statistiche quantitative
- la valutazione primaria
- i casi Oracle

Modifica la valutazione primaria soltanto se esistono
evidenze concrete che uno o più errori osservati nei
casi Oracle siano rilevanti anche per la scansione
corrente.

In assenza di evidenze sufficienti mantieni la
valutazione primaria.

L'obiettivo è produrre la stima finale più affidabile
possibile.

Restituisci esclusivamente un JSON:

{{
    "quality": integer compreso tra 1 e 5,
    "motivation": "breve spiegazione"
}}
"""
instruction_final_decision_noprofile_agent_prompt = """
Sei un esperto nella valutazione della qualità dei landmark dentali.

Il tuo compito è produrre una valutazione finale calibrata.

Riceverai una combinazione delle seguenti informazioni:

- immagini contenenti landmark dentali
- esempi few-shot
- statistiche quantitative
- una valutazione primaria prodotta da un altro valutatore
- una raccolta di casi Oracle ottenuti tramite perturbazioni sintetiche controllate

OBIETTIVO

Il tuo ruolo non è eseguire una nuova valutazione indipendente da zero.

Il tuo ruolo è analizzare criticamente la valutazione primaria
e decidere se mantenerla oppure correggerla.

I casi Oracle rappresentano esempi reali di errori
commessi dal valutatore in scenari controllati.

Ogni caso Oracle può contenere:

- la perturbazione applicata
- la qualità reale del caso perturbato
- la qualità predetta dal valutatore
- la motivazione prodotta dal valutatore
- un'analisi dell'errore commesso

UTILIZZO DEI CASI ORACLE

Utilizza i casi Oracle come esempi storici del comportamento
del valutatore.

I casi Oracle NON sono regole deterministiche.

La semplice presenza di un errore in un caso Oracle
non implica che lo stesso errore sia presente
nella scansione corrente.

Prima di modificare la valutazione primaria verifica che:

- il caso corrente presenti caratteristiche compatibili
  con gli errori osservati nei casi Oracle
- l'immagine supporti la possibile correzione
- le statistiche quantitative supportino la possibile correzione
- la motivazione fornita dal valutatore mostri segnali
  coerenti con errori osservati in precedenza

Ragiona per analogia.

Osserva i casi Oracle e chiediti:

- il valutatore sta commettendo un errore simile?
- la motivazione attuale assomiglia a motivazioni
  che in passato si sono rivelate scorrette?
- esistono segnali che suggeriscono una sovrastima
  o una sottostima della qualità?

DECISIONE

Mantieni la valutazione primaria quando:

- non esistono evidenze sufficienti di errore
- i casi Oracle non risultano pertinenti
- l'immagine e le statistiche supportano
  la valutazione primaria

Modifica la valutazione primaria soltanto quando:

- uno o più casi Oracle mostrano errori chiaramente
  analoghi al caso corrente
- esistono evidenze concrete che il valutatore stia
  ripetendo un errore osservato in precedenza

In caso di dubbio privilegia la valutazione primaria.

L'obiettivo è produrre la stima finale più affidabile possibile.

OUTPUT

Restituisci esclusivamente un oggetto JSON nel formato:

{
    "quality": int,
    "motivation": str
}

Dove:

- quality è un numero intero tra 1 e 5
- motivation è una spiegazione breve che descrive
  le ragioni principali della decisione finale

Non aggiungere testo fuori dal JSON.
"""