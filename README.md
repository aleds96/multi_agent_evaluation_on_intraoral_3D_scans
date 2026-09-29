# Multi-Agent Evaluation on Intraoral 3D Scans

Questo repository contiene un framework agentico multimodale per la valutazione automatica della qualità dei landmark dentali su scansioni 3D intraorali.

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

- **Landmark Shift Perturbation**  **( ancora da implementare!!!)**

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
- metriche quantitative (MAE, Accuracy, Accuracy ±1, Quadratic Weighted Kappa).

Tutti i risultati vengono salvati automaticamente nella directory:

```text
experiments/
```