from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from fastapi import FastAPI
import torch
from transformers import AutoTokenizer

app = FastAPI()

# load embbedding model and tokeniser
model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)


class Item(BaseModel):
    input: list
    model: str
    encoding_format: str = None


@app.post("/v1/embeddings")
async def create_embedding(item: Item):
    # ensure list
    texts = [str(x) for x in item.input]

    # cal token size
    tokens = tokenizer(texts, padding=True, truncation=True)
    token_count = sum(len(ids) for ids in tokens['input_ids'])

    # create emb
    with torch.no_grad():
        embeddings = model.encode(texts, convert_to_tensor=True)

    # vector to list
    embeddings_list = embeddings.tolist()

    # api
    data = [
        {
            "object": "embedding",
            "index": i,
            "embedding": emb
        }
        for i, emb in enumerate(embeddings_list)
    ]

    return {
        "object": "list",
        "data": data,
        "model": "text-embedding-3-small",  # rename
        "usage": {
            "prompt_tokens": token_count,
            "total_tokens": token_count
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8166)