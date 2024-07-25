import os
import pandas as pd
import asyncio
import logging
import tiktoken #tokeniser

from pydantic import BaseModel, Field #Superclass & API request bodies
from typing import List, Optional, Dict, Any, Union

### GraphRAG related imports
from graphrag.query.indexer_adapters import (
    #read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units
)

from graphrag.query.input.loaders.dfs import store_entity_semantic_embeddings
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.vector_stores.lancedb import LanceDBVectorStore


### Set up logging
logger = logging.getLogger(__name__) #global or need a class or put inside the function

### Set constants and configurations
INPUT_DIR = os.getenv('INPUT_DIR')
LANCEDB_URI = f"{INPUT_DIR}/lancedb"
COMMUNITY_REPORT_TABLE = "create_final_community_reports"
ENTITY_TABLE = "create_final_nodes"
ENTITY_EMBEDDING_TABLE = "create_final_entities"
RELATIONSHIP_TABLE = "create_final_relationships"
#COVARIATE_TABLE = "create_final_covariates"
TEXT_UNIT_TABLE = "create_final_text_units"
COMMUNITY_LEVEL = 2
PORT = 8012

### Global variables for storing search engines and question generator
local_search_engine = None
global_search_engine = None
question_generator = None


### Pydantic superclass for request bodies
class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Usage
    system_fingerprint: Optional[str] = None


async def setup_llm_and_embedder():
    """
    Set up Language Model (LLM) and embedding model
    """
    logger.info("Setting up LLM and embedder")

    # Get API keys and base URLs
    api_key = os.getenv("GRAPHRAG_API_KEY", False) # or os.environ.get() or os.environ[""]
    api_key_embedding = os.getenv("GRAPHRAG_API_KEY_EMBEDDING", api_key)
    api_base = os.getenv("GRAPHRAG_API_BASE", "https://api.openai.com/v1")
    api_base_embedding = os.getenv("GRAPHRAG_API_BASE_EMBEDDING", "https://api.openai.com/v1")

    # Get model names
    llm_model = os.getenv("LLM_MODEL", "gpt-3.5-turbo-0125")
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # Check if API key exists
    if not api_key: # or check if api_key == specific string
        logger.error("Valid API_KEY not found in environment variables")
        raise ValueError("API_KEY is missing")

    # Initialise ChatOpenAI instance
    llm = ChatOpenAI(
        api_key=api_key,
        api_base=api_base,
        model=llm_model,
        api_type=OpenaiApiType.OpenAI,
        max_retries=20,
    )

    # Initialise tokeniser
    # https://github.com/openai/tiktoken
    # "tiktoken is a fast BPE tokeniser for use with OpenAI's models."
    token_encoder = tiktoken.get_encoding("cl100k_base")

    # Initialise text embedding model
    text_embedder = OpenAIEmbedding(
        api_key=api_key_embedding,
        api_base=api_base_embedding,
        api_type=OpenaiApiType.OpenAI,
        model=embedding_model,
        deployment_name=embedding_model,
        max_retries=20,
    )


    logger.info("LLM and Embedding setup complete")
    return llm, token_encoder, text_embedder


async def load_context():
    """
    Load context data including entities, relationships, reports, text units, and covariates
    """
    logger.info("Loading context data")
    try:
        entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
        entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")
        entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)

        description_embedding_store = LanceDBVectorStore(collection_name="entity_description_embeddings")
        description_embedding_store.connect(db_uri=LANCEDB_URI)
        store_entity_semantic_embeddings(entities=entities, vectorstore=description_embedding_store)

        relationship_df = pd.read_parquet(f"{INPUT_DIR}/{RELATIONSHIP_TABLE}.parquet")
        relationships = read_indexer_relationships(relationship_df)

        report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
        reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)

        text_unit_df = pd.read_parquet(f"{INPUT_DIR}/{TEXT_UNIT_TABLE}.parquet")
        text_units = read_indexer_text_units(text_unit_df)

        #covariate_df = pd.read_parquet(f"{INPUT_DIR}/{COVARIATE_TABLE}.parquet")
        #claims = read_indexer_covariates(covariate_df)
        #logger.info(f"Number of claim records: {len(claims)}")
        #covariates = {"claims": claims}

        logger.info("Context data loading complete")
        return entities, relationships, reports, text_units, description_embedding_store#, covariates
    except Exception as e:
        logger.error(f"Error loading context data: {str(e)}")
        raise
