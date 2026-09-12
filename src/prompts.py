
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

instruction_prompt = """
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

def build_user_request_for_scan(input_info, examples):

    lvl5_profiles = "\n".join(_format_profile(p['profile']) for p in examples["lvl5"])
    lvl2_profiles = "\n".join(_format_profile(p['profile']) for p in examples["lvl2"])
    lvl1_profiles = "\n".join(_format_profile(p['profile']) for p in examples["lvl1"])

    pred_counts_global_str = "\n".join(
        f"- {cls}: {input_info['pred_count_per_class'][cls]}" for cls in input_info['pred_count_per_class']
    )
    group_counts_str = format_group_counts(input_info["pred_count_per_group_class"])
    return f"""
## Profili degli esempi

### Esempio qualità 5
{lvl5_profiles}

### Esempio qualità 2
{lvl2_profiles}

### Esempio qualità 1
{lvl1_profiles}

## Statistiche quantitative sull'immagine di input da valutare
### Conteggio dei landmark per classe
{pred_counts_global_str}
### Statistiche per gruppo dentale
{group_counts_str}
## Immagini fornite (in ordine)
1. Esempio qualità 5
2. Esempio qualità 2
3. Esempio qualità 1
4. Immagine di input da valutare

Valuta solo l’immagine di input (la quarta), confrontandola con gli esempi e con i loro profili.
"""
