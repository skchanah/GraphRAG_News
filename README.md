# GraphRAG News Intelligence 
An offline, self-hosted GraphRAG app powered by Open-WebUI and gemma in Ollama. Encapsulated a local context search, global context search, web search (Tavily; online required) and full search. News crawlers are included for the offline knowledge graph books.

## Requirement
GraphRAG: Python 3.10 - 3.12

`pip install -r requirement.txt`

## Data Source
BBC Front Page: [BBC](https://dracos.co.uk/made/bbc-news-archive/archive.php)

## Quick Start

### Crawler

### Install Open-WebUI
```shell
pip install open-webui
open-webui serve
```

In your browser, go to http://localhost:8080/

### Still in Global Terminal
`python -m graphrag.index --init  --root . [your/root/address] #initate`

After dragging the books into the .input/ ,

`python -m graphrag.index  --root . [your/root/address] #index`

`python -m graphrag.prompt_tune --root . --no-entity-types #prompt-tuning` 

`python -m venv venv`
`source venv/bin/activate`

### After venv is activated
`pip install -r requirement.txt`

`touch .secrets/secrets.toml`

Paste:

```shell
LLM_API_KEY=""
#EMBEDDING_API_KEY=""
#LLM_API_BASE=""
#EMBEDDING_API_BASE=""
TAVILY_API_KEY=""
LLM_MODEL="gpt-3.5-turbo"
EMBEDDING_MODEL="text-embedding-3-small"
INPUT_DIR="/path/to/your/input/directory"
```


## Architecture


## Acknowledgement
Crawler:
https://github.com/LuChang-CS/news-crawler

GraphRAG API with Open-WebUI
https://github.com/win4r/GraphRAG4OpenWebUI


secrets