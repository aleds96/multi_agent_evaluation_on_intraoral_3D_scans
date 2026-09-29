from pydantic import BaseModel, Field
from typing import List,Literal
class FinalDecisionOutput(BaseModel):
    quality: int = Field(
        ...,
        description="Final calibrated quality score between 1 and 5."
    )

    motivation: str = Field(
        ...,
        description="Concise explanation (2-4 sentences) supporting the final calibrated judgement."
    )


class OracleBias(BaseModel):

    bias_name: str = Field(
        description=(
            "Nome breve e descrittivo del bias individuato."
        )
    )
    description: str = Field(
        description=(
            "Descrizione dettagliata del pattern osservato. "
            "Spiega quando emerge il bias e come si manifesta."
        )
    )

    error_direction: Literal[
        "overestimation",
        "underestimation",
        "hallucinated_explanation",
        "mixed",
    ] = Field(
        description=(
            "Direzione tipica dell'errore. "
            "overestimation: qualità sovrastimata. "
            "underestimation: qualità sottostimata. "
            "hallucinated_explanation: motivazioni inventate o non supportate. "
            "mixed: il bias si manifesta in modi differenti."
        )
    )

    typical_effect: str = Field(
        description=(
            "Effetto tipico del bias sul processo di valutazione. "
            "Ad esempio: "
            "'quality tends to decrease by one level', "
            "'hallucinates anatomical defects', "
            "'generalizes local failures to unrelated landmarks'."
        )
    )

    trigger_conditions: List[str] = Field(
        description=(
            "Condizioni che tendono ad attivare il bias. "
            "Devono essere condizioni generali osservate in più esempi "
            "e non riferimenti a singole scansioni."
        )
    )

    supporting_examples: List[str] = Field(
        description=(
            "Lista di perturbazioni concrete che hanno contribuito "
            "all'identificazione del bias. "
            "Utilizzare nomi sintetici come "
            "'Mesial 20%', "
            "'Distal 20%', "
            "'InnerPoint 60%'."
        )
    )


class OracleProfileOutput(BaseModel):

    strengths: List[str] = Field(
        description=(
            "Punti di forza osservati del valutatore. "
            "Massimo 3 elementi."
        )
    )

    weaknesses: List[str] = Field(
        description=(
            "Principali debolezze osservate del valutatore. "
            "Massimo 3 elementi."
        )
    )

    biases: List[OracleBias] = Field(
        description=(
            "Lista strutturata dei bias ricorrenti individuati "
            "analizzando gli errori del valutatore."
        )
    )

    profile_summary: str = Field(
        description=(
            "Riassunto finale del profilo. "
            "Descrive le principali vulnerabilità osservate "
            "e il comportamento generale del valutatore."
        )
    )
class OracleErrorDescriptionOutput(BaseModel):
    failure_analysis: str
class SingleAgentOutput(BaseModel):
    quality: int = Field(
        ...,
        description=(
            "Indice di qualità globale della scan, compreso tra 1 e 5. "
            "Il valore deve essere un intero e rappresenta la valutazione "
            "complessiva della correttezza del posizionamento dei landmark."
        ),
    )
    motivation: str = Field(
        ...,
        description=(
            "Breve motivazione (2–4 frasi) che spiega perché è stato assegnato "
            "l'indice di qualità. Deve essere concisa, focalizzata sui landmark "
            "e coerente con gli esempi forniti."
        ),
    )
