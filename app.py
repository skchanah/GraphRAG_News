import time
import uuid
import json
import re
import logging
import asyncio
from contextlib import asynccontextmanager #declarator
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from dotenv import load_dotenv
load_dotenv()

from setup import (
    setup_llm_and_embedder, 
    load_context,
    ChatCompletionRequest, 
    ChatCompletionResponseChoice, 
    ChatCompletionResponse, 
    Message, 
    Usage
)
from search import (
    tavily_search,
    setup_search_engines
)

from graphrag.query.question_gen.local_gen import LocalQuestionGen

logger = logging.getLogger(__name__)
PORT = 8012

def format_response(response):
    """
    Format the response by adding appropriate line breaks and paragraph separations.
    """
    paragraphs = re.split(r'\n{2,}', response)

    formatted_paragraphs = []
    for para in paragraphs:
        if '```' in para:
            parts = para.split('```')
            for i, part in enumerate(parts):
                if i % 2 == 1:  # This is a code block
                    parts[i] = f"\n```\n{part.strip()}\n```\n"
            para = ''.join(parts)
        else:
            para = para.replace('. ', '.\n')

        formatted_paragraphs.append(para.strip())

    return '\n\n'.join(formatted_paragraphs)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Execute on startup
    global local_search_engine, global_search_engine, question_generator
    try:
        logger.info("Initializing search engines and question generator...")
        llm, token_encoder, text_embedder = await setup_llm_and_embedder()
        entities, relationships, reports, text_units, description_embedding_store, covariates = await load_context()
        local_search_engine, global_search_engine, local_context_builder, local_llm_params, local_context_params = await setup_search_engines(
            llm, token_encoder, text_embedder, entities, relationships, reports, text_units,
            description_embedding_store, covariates
        )

        question_generator = LocalQuestionGen(
            llm=llm,
            context_builder=local_context_builder,
            token_encoder=token_encoder,
            llm_params=local_llm_params,
            context_builder_params=local_context_params,
        )
        logger.info("Initialization complete.")
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        raise

    yield
    
    # Execute on shutdown
    logger.info("Shutting down...")


# Add the following code to the chat_completions function

async def full_model_search(prompt: str):
    """
    Perform a full model search, including local retrieval, global retrieval, and Tavily search
    """
    local_result = await local_search_engine.asearch(prompt)
    global_result = await global_search_engine.asearch(prompt)
    tavily_result = await tavily_search(prompt)

    # Format results
    formatted_result = "# 🔥🔥🔥Comprehensive Search Results\n\n"

    formatted_result += "## 🔥🔥🔥Local Retrieval Results\n"
    formatted_result += format_response(local_result.response) + "\n\n"

    formatted_result += "## 🔥🔥🔥Global Retrieval Results\n"
    formatted_result += format_response(global_result.response) + "\n\n"

    formatted_result += "## 🔥🔥🔥Tavily Search Results\n"
    formatted_result += tavily_result + "\n\n"

    return formatted_result

app = FastAPI(lifespan=lifespan)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not local_search_engine or not global_search_engine:
        logger.error("Search engines not initialized")
        raise HTTPException(status_code=500, detail="Search engines not initialized")

    try:
        logger.info(f"Received chat completion request: {request}")
        prompt = request.messages[-1].content
        logger.info(f"Processing prompt: {prompt}")

        # Choose different search methods based on the model
        if request.model == "graphrag-global-search:latest":
            result = await global_search_engine.asearch(prompt)
            formatted_response = format_response(result.response)
        elif request.model == "tavily-search:latest":
            result = await tavily_search(prompt)
            formatted_response = result
        elif request.model == "full-model:latest":
            formatted_response = await full_model_search(prompt)
        else:  # Default to local search
            result = await local_search_engine.asearch(prompt)
            formatted_response = format_response(result.response)

        logger.info(f"Formatted search result: {formatted_response}")

        # Handle streaming and non-streaming responses
        if request.stream:
            async def generate_stream():
                chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
                lines = formatted_response.split('\n')
                for i, line in enumerate(lines):
                    chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": line + '\n'} if i > 0 else {"role": "assistant", "content": ""},
                                "finish_reason": None
                            }
                        ]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    await asyncio.sleep(0.05)

                final_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }
                    ]
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(generate_stream(), media_type="text/event-stream")
        else:
            response = ChatCompletionResponse(
                model=request.model,
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        message=Message(role="assistant", content=formatted_response),
                        finish_reason="stop"
                    )
                ],
                usage=Usage(
                    prompt_tokens=len(prompt.split()),
                    completion_tokens=len(formatted_response.split()),
                    total_tokens=len(prompt.split()) + len(formatted_response.split())
                )
            )
            logger.info(f"Sending response: {response}")
            return JSONResponse(content=response.dict())

    except Exception as e:
        logger.error(f"Error processing chat completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
    """
    Return a list of available models
    """
    logger.info("Received model list request")
    current_time = int(time.time())
    models = [
        {"id": "graphrag-local-search:latest", "object": "model", "created": current_time - 100000, "owned_by": "graphrag"},
        {"id": "graphrag-global-search:latest", "object": "model", "created": current_time - 95000, "owned_by": "graphrag"},
        # {"id": "graphrag-question-generator:latest", "object": "model", "created": current_time - 90000, "owned_by": "graphrag"},
        # {"id": "gpt-3.5-turbo:latest", "object": "model", "created": current_time - 80000, "owned_by": "openai"},
        # {"id": "text-embedding-3-small:latest", "object": "model", "created": current_time - 70000, "owned_by": "openai"},
        {"id": "tavily-search:latest", "object": "model", "created": current_time - 85000, "owned_by": "tavily"},
        {"id": "full-model:latest", "object": "model", "created": current_time - 80000, "owned_by": "combined"}
    ]

    response = {
        "object": "list",
        "data": models
    }

    logger.info(f"Sending model list: {response}")
    return JSONResponse(content=response)

### Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting server on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)