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

- Retrieval Augemented Generation (RAG)
  - [✅ Indexing](./rag/indexing.ipynb) - Indexing a vector search backend
  - [🚧 Building](./rag/building.ipynb) - Building a RAG pipeline
  - [🚧 Monitoring](./rag/monitoring.ipynb) - Monitoring and improving your RAG pipeline
  - [🚧 Fine-tuning](./rag/fine_tuning.ipynb) - Fine-tuning retrieval and reranking models
- Information extraction and labeling
  - [🚧 Building](./extraction/building.ipynb) - Structured information extraction with LLMs
  - [🚧 Monitoring](./extraction/monitoring.ipynb) - Monitoring extraction quality
  - [🚧 Fine-tuning](./extraction/fine_tuning.ipynb) - Fine-tuning extraction models
- Agents for orchestration
  - [🚧 Orchestration](./agents/orchestration.ipynb) - Building agents to coordinate components

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

