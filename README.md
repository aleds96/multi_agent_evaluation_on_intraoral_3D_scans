# Multi-Agent Evaluation on Intraoral 3D Scans

## Overview

Questo repository contiene un framework agentico multimodale per la valutazione automatica della qualità dei landmark dentali su scansioni 3D intraorali.
Il progetto nasce dalla seguente domanda:

> È possibile valutare la qualità delle predizioni di landmark senza avere accesso alla ground truth e, soprattutto, è possibile identificare e correggere automaticamente gli errori sistematici commessi dal valutatore?

L'obiettivo principale non è costruire un nuovo classificatore supervisionato della qualità, ma studiare se un sistema multimodale possa:

- stimare la qualità di predizioni geometriche complesse senza ground truth e con limitato numero di dati;
- individuare automaticamente i propri errori di valutazione;
- estrarre pattern ricorrenti di errore (bias);
- utilizzare tali bias per effettuare una forma di auto-calibrazione delle proprie valutazioni.

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

### Perturbazioni sintetiche supportate

- **Missing Landmark Perturbation**  
  Rimozione controllata di landmark selezionati.

- **Class Flip Perturbation**  
  Scambio controllato della classe associata a un landmark mantenendo inalterata la sua posizione spaziale.

- **Landmark Shift Perturbation**  
  Spostamento controllato delle coordinate di uno o più landmark.

Le perturbazioni possono essere applicate:

- globalmente all'intera scansione;
- localmente a specifiche classi di landmark (es. Mesial, Distal, InnerPoint, OuterPoint).

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
