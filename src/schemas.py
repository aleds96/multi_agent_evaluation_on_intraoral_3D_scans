from pydantic import BaseModel, Field

class FinalDecisionOutput(BaseModel):
    quality: int = Field(
        ...,
        description="Final calibrated quality score between 1 and 5."
    )

    motivation: str = Field(
        ...,
        description="Concise explanation (2-4 sentences) supporting the final calibrated judgement."
    )
class OracleProfileOutput(BaseModel):
    strengths: list[str]
    weaknesses: list[str]
    failure_modes: list[str]
    profile_summary: str
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
