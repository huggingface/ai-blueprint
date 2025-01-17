<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo-pirate.png" width="200px" alt="smol blueprint logo">
</div>

# A smol blueprint

A smol blueprint for AI development, focusing on applied examples of RAG, information extraction, analysis and fine-tuning in the age of LLMs. It is a more practical approach that strives to show the application of some of the theoretical learnings from [the smol-course](https://github.com/huggingface/smol-course) as an end2end real-world problem.

> 🚀 Web apps and microservices included!
>
> Each notebook will show how to deploy your AI as a [webapp on Hugging Face Spaces with Gradio](https://huggingface.co/docs/hub/en/spaces-sdks-gradio), which you can directly use as microservices through [the Gradio Python Client](https://www.gradio.app/guides/getting-started-with-the-python-client). All the code and demos can be used in a private or public setting. [Deployed on the Hub!](https://huggingface.co/smol-blueprint)

## The blueprint

We want to build a tool that can help us use AI on company documents. In our case, we will be working with the [smol-blueprint/hf-blogs](https://huggingface.co/datasets/smol-blueprint/hf-blogs) dataset, which is a dataset that contains the blogs from the Hugging Face website.

### RAG

All notebooks for RAG can be found in the [RAG directory](./rag) and all artifacts can be found in the [RAG collection on the Hub](https://huggingface.co/collections/smol-blueprint/retrieval-augemented-generation-rag-67877d7b69178ec7760e2862).

| Status | Notebook | Artifact | Title |
|---------|----------|-----------|-------|
| ✅ | [Retrieve](./rag/retrieve.ipynb) | Data - Gradio | Retrieve documents from a vector database |
| ✅ | [Augment](./rag/augment.ipynb) | Gradio | Augment retrieval results by reranking |
| ✅ | [Generate](./rag/generating.ipynb) | Gradio | Generating a response based on query and context |
| ✅ | [RAG](./rag/rag.ipynb) | Gradio | Combine all the components in a RAG pipeline |
| 🚧 | [Monitoring](./rag/monitoring.ipynb) | Data - Gradio | Monitoring and improving your pipeline |
| 🚧 | [Fine-tuning](./rag/fine_tuning.ipynb) | Model| Fine-tuning retrieval and reranking models |

### Information Extraction

| Status | Notebook | Artifact | Title |
|---------|----------|-----------|-------|
| 🚧 | [Building](./extraction/building.ipynb) | Extractor | Structured information extraction with LLMs |
| 🚧 | [Monitoring](./extraction/monitoring.ipynb) | Analytics | Monitoring extraction quality |
| 🚧 | [Fine-tuning](./extraction/fine_tuning.ipynb) | Models | Fine-tuning extraction models |

### Agents

| Status | Notebook | Artifact | Title |
|---------|----------|-----------|-------|
| 🚧 | [Orchestration](./agents/orchestration.ipynb) | Orchestrator | Building agents to coordinate components |

# Installation and configuration

## Python environment

We will use [uv](https://docs.astral.sh/uv/) to manage the project. First create a virtual environment:

```bash
uv venv --python 3.11
source .venv/bin/activate
```

Then you can install all the required dependencies:

```bash
uv sync --all-groups
```

Or you can sync between different dependency groups:

```bash
uv sync scraping
uv sync rag
uv sync information-extraction
```

## Hugging Face Account

You will need a Hugging Face account to use the Hub API. You can create one [here](https://huggingface.co/join). After this you can follow the [huggingface-cli instructions](https://huggingface.co/docs/huggingface_hub/installation#huggingface-cli) and log in to configure your Hugging Face token.

```bash
huggingface-cli login
```

