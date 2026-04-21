"""Define the configurable parameters for the agent."""

from typing import Annotated, Literal, Optional
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

class Configuration(BaseModel):

    model: Annotated[
            Literal[
                "openai:gpt-4o-mini",
                "openai:gpt-4o",
                "openai:gpt-3.5-turbo",
                "openai:gpt-4.1",
                "openai:gpt-4.1-mini"
            ],
            {"__template_metadata__": {"kind": "llm"}},
        ] = Field(
            default="openai:gpt-4o",
            description="Provider:model, e.g. 'openai:gpt-4o'."
    )

    temperature: Annotated[
        float,
        {"__template_metadata__": {"kind": "number", "min": 0.0, "max": 1.0}},
    ] = Field(default=0.0)

    extraction_temperature: Annotated[
        float,
        {"__template_metadata__": {"kind": "number", "min": 0.0, "max": 1.0}},
    ] = Field(default=0.2)

    max_retries: Annotated[
        int,
        {"__template_metadata__": {"kind": "number", "min": 0, "max": 10}},
    ] = Field(default=3)

    retriever_k: Annotated[
        int,
        {"__template_metadata__": {"kind": "number", "min": 1, "max": 7}}
    ] = Field(default=5)

    embed_model: Annotated[
        Literal[
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ],
        {"__template_metadata__": {"kind": "select"}},
    ] = Field(default="text-embedding-3-small")

    retriever: Annotated[
        Literal["pinecone"],
        {"__template_metadata__": {"kind": "select"}},
    ] = Field(default="pinecone")

    pinecone_index_name: Annotated[
        Literal["seal-v3-hard", "seal-2wiki-v1"],
        {"__template_metadata__": {"kind": "select"}},
    ] = Field(default="seal-2wiki-v1")

    pinecone_namespace: Annotated[
        Optional[str],
        {"__template_metadata__": {"kind": "text"}},
    ] = Field(default=None)

    repair_loop_limit: Annotated[
        int,
        {"__template_metadata__": {"kind": "number", "min": 0, "max": 50}},
    ] = Field(default=1)

    # AdaGATE new parameters
    token_budget: Annotated[
        int,
        {"__template_metadata__": {"kind": "number", "min": 100, "max": 8000}},
    ] = Field(
        default=3000,
        description="Global token budget B for evidence set passed to generator."
    )

    tau_conf: Annotated[
        float,
        {"__template_metadata__": {"kind": "number", "min": 0.0, "max": 1.0}},
    ] = Field(
        default=0.70,
        description="Confidence threshold below which a triplet is considered uncertain (for Corr term)."
    )

    def build_llm(self):
        return init_chat_model(
            self.model,
            temperature=self.temperature,
            max_retries=self.max_retries,
        )

    def build_extraction_llm(self):
        return init_chat_model(
            self.model,
            temperature=self.extraction_temperature,
            max_retries=self.max_retries,
        )

    def build_retriever(self):
        if self.retriever != "pinecone":
            raise ValueError(f"Unsupported retriever backend: {self.retriever}")
        embeddings = OpenAIEmbeddings(model=self.embed_model)
        vectorstore = PineconeVectorStore(
            index_name=self.pinecone_index_name,
            embedding=embeddings,
            namespace=self.pinecone_namespace,
        )
        return vectorstore.as_retriever(search_kwargs={"k": self.retriever_k})