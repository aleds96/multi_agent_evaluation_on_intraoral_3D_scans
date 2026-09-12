from pydantic import BaseModel, Field


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
