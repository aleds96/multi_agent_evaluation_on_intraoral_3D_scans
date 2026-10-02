# Failure Mode Discovery and Bias-Aware Calibration for Multimodal Landmark Quality Assessment on Intraoral 3D Scans

## Overview

Questo repository contiene un framework agentico multimodale per la valutazione automatica della qualità dei landmark dentali su scansioni 3D intraorali.
Il progetto nasce dalla seguente domanda:

> È possibile valutare la qualità delle predizioni di landmark senza avere accesso alla ground truth e, soprattutto, è possibile identificare e correggere automaticamente gli errori sistematici commessi dal valutatore?

L'obiettivo principale non è costruire un nuovo classificatore supervisionato della qualità, ma studiare se un sistema multimodale possa:

- stimare la qualità di predizioni geometriche complesse senza ground truth e con limitato numero di dati;
- individuare automaticamente i propri errori di valutazione;
- estrarre pattern ricorrenti di errore (**structured bias extraction**);
- utilizzare tali bias per effettuare una forma di auto-calibrazione delle proprie valutazioni (**bias-aware external calibration**).

Per rispondere a queste domande viene proposto un framework multi-agent basato su perturbazioni sintetiche controllate, analisi automatica dei failure mode ed estrazione strutturata dei bias del valutatore.

---
## Input Representation

Ogni scansione 3D viene trasformata in una rappresentazione multimodale composta da:

- screenshot della scansione da vista occlusale (top view);
- segmentazione dentale colorata per gruppo anatomico;
- landmark colorati in base alla loro classe;
- statistiche quantitative sul numero di landmark predetti per classe.

L'obiettivo è fornire contemporaneamente:

- informazioni geometriche e anatomiche (immagine);
- informazioni strutturali sulla distribuzione dei landmark (conteggi per classe).

Esperimenti preliminari hanno mostrato che l'aggiunta delle statistiche quantitative migliora significativamente la capacità del valutatore di identificare landmark mancanti, anomalie di distribuzione e incoerenze anatomiche.

### Esempio di input
<img width="700" height="500" alt="predicted" src="https://github.com/user-attachments/assets/ce57dc73-5cc3-4ff4-8721-ddf07f4e8f21" />

- Mesial → rosso
- Distal → verde
- Cusp → blu
- InnerPoint → giallo
- OuterPoint → ciano
- FacialPoint → magenta

Gruppi dentali:

- Incisivi → grigio
- Canini → arancione
- Premolari → rosa
- Molari → azzurro
---

La pipeline utilizza modelli Gemini tramite Google ADK e combina:

- valutazione primaria della qualità mediante prompting multimodale few-shot (**Primary Evaluator Agent**);
- generazione di perturbazioni sintetiche controllate sui landmark (**Perturbation Engine**);
- analisi automatica dei failure mode del valutatore rispetto a perturbazioni note (**Oracle Error Descriptor Agent**);
- estrazione automatica di bias ricorrenti e costruzione di profili di calibrazione (**Oracle Profile Builder Agent**);
- revisione finale guidata dal profilo di calibrazione (**Final Calibration Agent**).

L'obiettivo è migliorare l'affidabilità delle valutazioni sfruttando dati sintetici e meta-ragionamento, senza richiedere ulteriori annotazioni manuali.

Le perturbazioni sintetiche attualmente implementate includono:

- landmark mancanti (Missing Landmark Perturbation);
- scambio di classi tra landmark (Class Flip Perturbation);
- spostamenti spaziali controllati dei landmark (Landmark Shift Perturbation).

Le perturbazioni possono essere applicate:

- globalmente all'intera scansione;
- localmente a specifiche classi di landmark (es. Mesial, Distal, InnerPoint, OuterPoint).


I failure mode osservati vengono trasformati in profili strutturati di bias che guidano l'agente calibratore nella correzione delle valutazioni potenzialmente errate.

L'architettura proposta segue il flusso:

```text
Primary Evaluator
        ↓
Synthetic Perturbation Engine
        ↓
Oracle Error Descriptor
        ↓
Bias Extraction / Oracle Profile Builder
        ↓
Final Calibration Agent
```
## Experimental Results

| Architecture | Profile Builder | Self Reflection | Configuration | MAE ↓ | Accuracy ↑ | Accuracy ±1 ↑ | QWK ↑ |
|-------------|----------------|----------------|---------------|--------:|-----------:|--------------:|------:|
| Single-Agent | No | No | Baseline | 1.0893 | 0.3929 | 0.6964 | 0.3581 |
| Single-Agent | No | Yes | Baseline | 1.0172 | 0.2931 | 0.7586 | 0.4065 |
| Multi-Agent | Yes | No | Missing | 0.7931 | 0.3966 | 0.8448 | 0.5621 |
| Multi-Agent | Yes | No | Flip | 0.8793 | 0.2931 | 0.8621 | 0.5565 |
| Multi-Agent | Yes | No | Shift | 0.7931 | 0.3966 | 0.8448 | 0.6364 |
| Multi-Agent | Yes | No | Shift (extended oracle set) | **0.7414** | **0.4655** | 0.8276 | 0.6415 |
| Multi-Agent | Yes | No | Missing + Flip | 0.8103 | 0.3621 | 0.8448 | 0.6342 |
| Multi-Agent | Yes | No | Missing + Shift | 0.7931 | 0.4310 | 0.8103 | 0.5443 |
| Multi-Agent | Yes | No | Flip + Shift | **0.7586** | 0.3621 | **0.8966** | **0.6980** |
| Multi-Agent | Yes | Yes | Flip + Shift | 0.9655 | 0.3448 | 0.7586 | 0.4320 |
| Multi-Agent | Yes | No | Missing + Flip + Shift | 0.9310 | 0.3448 | 0.7586 | 0.5209 |
| Multi-Agent | No | No | Flip + Shift (Raw Oracle Cases) | 0.8103 | 0.3966 | 0.8103 | 0.5757 |
| Multi-Agent | No | No | Missing + Flip + Shift (Raw Oracle Cases) | 0.8621 | 0.4138 | 0.7586 | 0.5127 |

### Risultati Principali

- La migliore configurazione complessiva è **Flip + Shift con Profile Builder**, che raggiunge un valore di **Quadratic Weighted Kappa (QWK) pari a 0.6980**.
- L'utilizzo di **perturbazioni sintetiche** migliora in modo consistente le prestazioni rispetto alla baseline basata su un singolo agente.
- Il **Profile Builder** svolge un ruolo fondamentale: sostituire i profili strutturati dei bias con le descrizioni grezze degli errori Oracle comporta una riduzione significativa delle prestazioni.
- I risultati suggeriscono che l'**astrazione dei failure mode e l'estrazione dei bias** siano più efficaci rispetto al fornire direttamente all'agente di calibrazione una cronologia non strutturata degli errori osservati.
- Una semplice fase di self-reflection migliora moderatamente il valutatore singolo (QWK: 0.358 → 0.407), ma peggiora significativamente le prestazioni del sistema già calibrato (QWK: 0.698 → 0.432), suggerendo che la calibrazione guidata dai bias sia più efficace della sola auto-riflessione.
### Esecuzione

Posizionarsi nella directory principale del progetto:

```bash
cd multi_agent_evaluation_on_intraoral_3D_scans
```

### Workflow Single-Agent

Esegue la valutazione della qualità dei landmark tramite un singolo agente multimodale few-shot.

```bash
uv run -m src.ai_architectures.single_agent_wf
```

Pipeline:

```text
Input Scan
     ↓
Few-Shot Examples
     ↓
Primary Evaluator Agent
     ↓
Quality Prediction
```

---

### Workflow Multi-Agent

Esegue la pipeline completa di calibrazione basata su perturbazioni sintetiche, analisi automatica degli errori ed estrazione di bias.

```bash
uv run -m src.ai_architectures.multi_agent_wf
```

Pipeline:

```text
Input Scan
     ↓
Primary Evaluator Agent
     ↓
Synthetic Perturbation Engine
     ↓
Oracle Error Descriptor Agent
     ↓
Oracle Profile Builder Agent
     ↓
Final Calibration Agent
     ↓
Calibrated Quality Prediction
```

---


### Output

Per ogni scansione vengono generati:

- valutazione del Primary Evaluator Agent;
- esempi Oracle ottenuti tramite perturbazioni sintetiche;
- analisi dei failure mode;
- profilo strutturato dei bias del valutatore;
- valutazione finale calibrata;
- metriche quantitative:
  - MAE
  - Accuracy
  - Accuracy ±1
  - Quadratic Weighted Kappa (QWK)

Tutti i risultati vengono salvati automaticamente nella directory:

```text
experiments/
```
