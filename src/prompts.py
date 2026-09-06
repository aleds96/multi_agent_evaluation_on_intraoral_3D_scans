def build_prompt_for_scan(input_profile, examples):
    ex_lvl5 = examples["lvl5"]
    ex_lvl2 = examples["lvl2"]
    ex_lvl1 = examples["lvl1"]
    return f"""
## Esempi di qualità

### Qualità 5 — {ex_lvl5['scan']}
Profilo: {ex_lvl5}
### Qualità 2 — {ex_lvl2['scan']}
Profilo: {ex_lvl2}
### Qualità 1 — {ex_lvl1['scan']}
Profilo: {ex_lvl1}
---
## Profilo della scan di input
{input_profile}
---
## Compito
Valuta SOLO l’immagine di input, confrontandola con gli esempi.

Restituisci:
Qualità: X
Motivazione: [3–5 frasi]
Non includere altro.
"""
