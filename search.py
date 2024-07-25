import logging
import os

from tavily import TavilyClient

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.query.structured_search.global_search.community_context import GlobalCommunityContext
from graphrag.query.structured_search.global_search.search import GlobalSearch

logger = logging.getLogger(__name__)


async def setup_search_engines(llm, token_encoder, text_embedder, entities, relationships, reports, text_units,
                               description_embedding_store):#, covariates):
    """
    Set up local and global search engines
    """
    logger.info("Setting up search engines")

    # Set up local search engine
    local_context_builder = LocalSearchMixedContext(
        community_reports=reports,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        #covariates=covariates,
        entity_text_embeddings=description_embedding_store,
        embedding_vectorstore_key=EntityVectorStoreKey.ID,
        text_embedder=text_embedder,
        token_encoder=token_encoder,
    )

    local_context_params = {
        "text_unit_prop": 0.5,
        "community_prop": 0.1,
        "conversation_history_max_turns": 5,
        "conversation_history_user_turns_only": True,
        "top_k_mapped_entities": 10,
        "top_k_relationships": 10,
        "include_entity_rank": True,
        "include_relationship_weight": True,
        "include_community_rank": False,
        "return_candidate_context": False,
        "embedding_vectorstore_key": EntityVectorStoreKey.ID,
        "max_tokens": 12_000,
    }

    local_llm_params = {
        "max_tokens": 2_000,
        "temperature": 0.0,
    }

    local_search_engine = LocalSearch(
        llm=llm,
        context_builder=local_context_builder,
        token_encoder=token_encoder,
        llm_params=local_llm_params,
        context_builder_params=local_context_params,
        response_type="multiple paragraphs",
    )

    # Set up global search engine
    global_context_builder = GlobalCommunityContext(
        community_reports=reports,
        entities=entities,
        token_encoder=token_encoder,
    )

    global_context_builder_params = {
        "use_community_summary": False,
        "shuffle_data": True,
        "include_community_rank": True,
        "min_community_rank": 0,
        "community_rank_name": "rank",
        "include_community_weight": True,
        "community_weight_name": "occurrence weight",
        "normalize_community_weight": True,
        "max_tokens": 12_000,
        "context_name": "Reports",
    }

    map_llm_params = {
        "max_tokens": 1000,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }

    reduce_llm_params = {
        "max_tokens": 2000,
        "temperature": 0.0,
    }

    global_search_engine = GlobalSearch(
        llm=llm,
        context_builder=global_context_builder,
        token_encoder=token_encoder,
        max_data_tokens=12_000,
        map_llm_params=map_llm_params,
        reduce_llm_params=reduce_llm_params,
        allow_general_knowledge=False,
        json_mode=True,
        context_builder_params=global_context_builder_params,
        concurrent_coroutines=32,
        response_type="multiple paragraphs",
    )

    logger.info("Search engines setup complete")
    return local_search_engine, global_search_engine, local_context_builder, local_llm_params, local_context_params


async def tavily_search(prompt: str):
    """
    Perform a search using the Tavily API
    """
    try:
        client = TavilyClient(api_key=os.environ['TAVILY_API_KEY'])
        resp = client.search(prompt, search_depth="advanced")

        # Convert Tavily response to Markdown format
        markdown_response = "# Search Results\n\n"
        for result in resp.get('results', []):
            markdown_response += f"## [{result['title']}]({result['url']})\n\n"
            markdown_response += f"{result['content']}\n\n"

        return markdown_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tavily search error: {str(e)}")

