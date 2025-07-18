from typing import List
from langchain.embeddings.base import Embeddings

from .ai_client import AIProjectClientDefinition


class AdaEmbeddingModel(AIProjectClientDefinition):
    """Wrapper per il modello di embedding Ada di Azure OpenAI."""

    def __init__(self, model_name: str = "text-embedding-ada-002"):
        super().__init__()
        self.model_name = model_name
        self.azure_client = self.client.inference.get_azure_openai_client(
            api_version="2023-05-15"
        )

    def embed_text(self, text: str) -> List[float]:
        response = self.azure_client.embeddings.create(input=[text], model=self.model_name)
        return response.data[0].embedding


class LangchainAdaWrapper(Embeddings):
    """Adapter per usare il modello Ada con Langchain."""

    def __init__(self, ada_model: AdaEmbeddingModel):
        self.ada_model = ada_model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.ada_model.embed_text(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.ada_model.embed_text(text)
