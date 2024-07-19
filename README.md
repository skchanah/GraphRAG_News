# GraphRAG News Intelligence 
An offline, self-hosted GraphRAG app powered by Open-WebUI and LLaMa3 in Ollama. Encapsulated a local context search, global context search, web search (Tavily; online required) and full search. News crawlers are included for the offline knowledge graph books.

## Requirement
GraphRAG: Python 3.10 - 3.12
`pip install -r requirement.txt`

## Data Source
BBC Front Page: [BBC](https://dracos.co.uk/made/bbc-news-archive/archive.php)

## Quick Start
### In Terminal
`python -m graphrag.index --init  optional: [--root your/root/address] `
`python -m graphrag.index  optional: [--root your/root/address] `

Enabling prompt tunning
`python -m graphrag.prompt_tune --root . --no-entity-types`

`python -m venv venv`

`source venv/bin/activate`

`pip install -r requirement.txt`

`touch .secrets/secrets.toml`

Paste:
    LLM_API_KEY=""
    #EMBEDDING_API_KEY=""
    #LLM_API_BASE=""
    #EMBEDDING_API_BASE=""
    TAVILY_API_KEY=""
    LLM_MODEL="gpt-3.5-turbo"
    EMBEDDING_MODEL="text-embedding-3-small"
    INPUT_DIR="/path/to/your/input/directory"


## Architecture


## Acknowledgement
Crawler:
https://github.com/LuChang-CS/news-crawler

GraphRAG API with Open-WebUI
https://github.com/win4r/GraphRAG4OpenWebUI


secrets