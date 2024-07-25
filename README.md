# GraphRAG News Intelligence 
An offline, self-hosted GraphRAG app powered by Open-WebUI and llama3.1 ~~gemma2~~ in Ollama. Supporting local context search, global context search, web search (Tavily; online required) and full search. News crawlers are included for the offline knowledge graph books.

> [!NOTE] 
> [GraphRAG](https://microsoft.github.io/graphrag/) is a **structured, hierarchical** approach to Retrieval Augmented Generation (RAG), as opposed to naive semantic-search approaches using plain text snippets. 
> The GraphRAG process involves extracting a knowledge graph out of raw text, building a community hierarchy, generating summaries for these communities, and then leveraging these structures when perform RAG-based tasks.



## Requirement
GraphRAG: Python 3.10 - 3.12

> [!TIP]
>`pip install -r requirement.txt`

## Roadmap
Recent Updates
- [x] ~~Initial Setups~~

Future Updates
- [ ] Enhance Spider
- [ ] Enhance Automations

## Data Source
BBC Front Page: [BBC](https://dracos.co.uk/made/bbc-news-archive/archive.php)

BBC: [BBC] (https://feeds.bbci.co.uk/news/world/rss.xml)

## Quick Start
### Crawler
Run:

```shell
python superspider.py
```

### Install Open-WebUI
```shell
pip install open-webui
open-webui serve
```
In your browser, go to http://localhost:8080/

> [!NOTE]
> If this is your first time trying Open-WebUI, you have to create an account to access to the local-hosted API. Don't worry this account is only for local session access, Open-WebUI does not connect online.

### Still in Terminal
```shell
python -m graphrag.index --init  --root ./ragtest #initate
```

### `settings.yaml`
Go to `Settings.yaml`, configure as follows:

```yaml
llm:
  api_key: ollama
  type: openai_chat # or azure_openai_chat
  model: llama3.1
  model_supports_json: true # recommended if this is available for your model.
  ...
  api_base: http://localhost:11434/v1
...

embeddings:
  ## parallelization: override the global parallelization settings for embeddings
  async_mode: threaded # or asyncio
  llm:
    api_key: ${GRAPHRAG_API_KEY}
    type: openai_embedding # or azure_openai_embedding
    model: nomic-embed-text:latest
    api_base: http://localhost:8166/v1 # From emb_api
...

claim_extraction:
...
...
    enabled: true
```

### `.env`
Go to `.env`, configure as follows:

```yaml
GRAPHRAG_API_KEY=[YOUR_API_KEY]
#GRAPHRAG_API_KEY_EMBEDDING=[YOUR_API_KEY]
#GRAPHRAG_API_BASE=""
#GRAPHRAG_API_BASE_EMBEDDING=""
TAVILY_API_KEY=[YOUR_TAVILY_API_KEY]
LLM_MODEL=gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-3-small
INPUT_DIR=./ragtest/output/[last_index_timestamp]/artifacts # Artifacts location from your last graphrag.index
```

### Indexing
Drag the crawled text books into the `.input/`,

>[!TIP]
> ```shell
> cp your_input/* ./ragtest/input
> ```

Run below in seperate terminals,

```shell
ollama run llama3.1
```
```shell
python app.py
```
```shell
python -m graphrag.index  --root ./ragtest #index
```
> [!TIP]
> You may also try 
> ```shell 
> python -m graphrag.prompt_tune --root . --no-entity-types #prompt-tuning
> ```
> to tune the prompts

```shell
python -m venv venv
source venv/bin/activate
```

### After venv is activated
```shell
pip install -r requirement.txt
python app.py
```

### After the Application Startup is completed
You should see a prompt showing `INFO:     Uvicorn running on http://0.0.0.0:8012 (Press CTRL+C to quit)`

Go back to the Open-WebUI http://localhost:8080/ in your browser, going to "Setting" on your top right > Admin Settings > Connections

In the `OPENAI API`, put http://localhost:8012/v1

On the User tab of the same page, update your default model


## Architecture
-- To be updated --

## Acknowledgement
BBC Archive:
https://dracos.co.uk/

GraphRAG API with Open-WebUI
https://github.com/win4r/GraphRAG4OpenWebUI
