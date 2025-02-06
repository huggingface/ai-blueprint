# Generate responses using a SmolLM

We have seen how to [retrieve](https://github.com/huggingface/ai-blueprint/blob/main/rag/retrieve.ipynb) and [rerank](https://github.com/huggingface/ai-blueprint/blob/main/rag/augment.ipynb) documents in a RAG pipeline. The next step is to create a tool that can generate a response to a query. We will use a SmolLM to generate a response to a query. At the end we will deploy a microservice that can be used to generate a response to a query.

## Dependencies and imports

Let's install the necessary dependencies.


```python
!pip install gradio gradio-client llama-cpp-python
```

Next, let's import the necessary libraries.


```python
import gradio as gr

from gradio_client import Client
from huggingface_hub import get_token, InferenceClient
from llama_cpp import Llama
```

## Inference API

There are different options for inference. In general most frameworks work great with some tradeoffs for speed, cost, and ease of deployment. In this example we will use a simple quantised model along with [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) because it works off the shelf and allows us to host that ourselves, which can run on CPU and does not require us to spin up a dedicated inference server. Additionally, we will use [a GGUF model](https://huggingface.co/docs/hub/en/gguf), which is a framework-agnostic file format that speeds up inference.

<details>
<summary>Inference servers</summary>
If you want to deploy your own inference servers there are various options. When using Apple Silicon, you can use the <a href="https://github.com/ml-explore/mlx">MLX</a> library. Alternatively, <a href="https://github.com/huggingface/text-generation-inference">Text Generation Inference (TGI)</a>, <a href="https://github.com/vllm-project/vllm">vLLM</a>, or <a href="https://github.com/ollama/ollama">Ollama</a> are great options to explore.
</details>

<details>
<summary>Structured outputs</summary>
If you want to generate structured outputs, there are two main approaches to do so depending on whether you have access to the weights of the models or not. When you have access to the weights of the models, you can use [Outlines](https://github.com/dottxt-ai/outlines), which changes sampling probabilities of tokens to ensure the model adheres to a specific structure, defined by a RegEx, JSON or Pydantic model. When you are using an API, you can use [Instructor](https://github.com/instructor-ai/instructor), which uses smart retries to ensure the model adheres to a specific structure.
</details>

### SmolLM in transformers

We will use [HuggingFaceTB/SmolLM2-135M-Instruct-GGUF](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct-GGUF) and use [the llama-cpp-python integration attached to the model on the Hub](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct-GGUF?library=llama-cpp-python). Note that we allow for passing `kwargs` like `max_new_tokens` as a parameter to the function which will be passed to the pipeline. Additionally, we set `n_ctx` to 7000, which is the maximum number of tokens we will be able to pass as prompt to the model. Our model has a maximum context length of 8192 tokens. If this is not enough we could choose a model with a larger context length or we could choose a more aggresive chunking strategy during the [retrieval phase](https://github.com/huggingface/ai-blueprint/blob/main/rag/retrieve.ipynb).


```python
llm = Llama.from_pretrained(
    repo_id="HuggingFaceTB/SmolLM2-135M-Instruct-GGUF",
    filename="smollm2-135m-instruct-q8_0.gguf",
    verbose=False,
    n_ctx=7000,
)


def generate_response_transformers(
    user_prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    max_tokens: int = 4000,
    temperature: float = 0.2,
    top_p: float = 0.95,
    top_k: int = 40,
):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )


generate_response_transformers(
    user_prompt="What is the future of AI?",
    system_prompt="You are a helpful assistant.",
)
```




    {'id': 'chatcmpl-54150cb9-00d6-4983-89da-e0527ae7480b',
     'object': 'chat.completion',
     'created': 1737651881,
     'model': '/Users/davidberenstein/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-360M-Instruct-GGUF/snapshots/593b5a2e04c8f3e4ee880263f93e0bd2901ad47f/./smollm2-360m-instruct-q8_0.gguf',
     'choices': [{'index': 0,
       'message': {'role': 'assistant',
        'content': 'The future of AI is a topic of ongoing debate and research. While AI has made tremendous progress in recent years, there are still many challenges and limitations to overcome before we can fully harness its potential.\n\nCurrently, AI is being used in various fields, such as healthcare, finance, and transportation. For example, AI-powered diagnostic tools are being used to detect diseases at an early stage, while AI-driven chatbots are being used to provide customer support.\n\nHowever, there are also concerns about the potential misuse of AI. For instance, AI can be used to manipulate public opinion, spread misinformation, and even take over and control our lives. As AI becomes more advanced, we will need to develop new regulations and safeguards to ensure that it is used responsibly.\n\nAnother area of research is the development of more human-like AI, which can think and act like humans. This is often referred to as "narrow AI" or "weak AI." While this type of AI can perform specific tasks, such as language translation or image recognition, it lacks the level of intelligence and creativity that humans possess.\n\nIn addition, there is a growing concern about the ethics of AI development. For example, AI systems can perpetuate biases and prejudices if they are trained on biased data. We need to ensure that AI is developed and deployed in a way that promotes fairness, equality, and transparency.\n\nOverall, the future of AI is likely to be shaped by a combination of technological advancements, societal values, and regulatory frameworks. As AI continues to evolve, we will need to work together to ensure that it is developed and used in a way that benefits humanity and promotes the common good.'},
       'logprobs': None,
       'finish_reason': 'stop'}],
     'usage': {'prompt_tokens': 27, 'completion_tokens': 342, 'total_tokens': 369}}



### SmolLM in Hugging Face Inference API

We will use the [serverless Hugging Face Inference API](https://huggingface.co/inference-api). This is free and means we don't have to worry about hosting the model. We can find models available for inference [through some basic filters on the Hub](https://huggingface.co/models?inference=warm&pipeline_tag=text-generation&sort=trending). We will use the [HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) model and call it with [the provided inference endpoint snippet](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct?inference_api=true).


```python
inference_client = InferenceClient(api_key=get_token())


def generate_response_api(
    user_prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    model: str = "HuggingFaceTB/SmolLM2-360M-Instruct",
    **kwargs,
):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    completion = inference_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )

    return completion.choices[0].message


generate_response_api(user_prompt="What is the future of AI?")
```




    ChatCompletionOutputMessage(role='assistant', content="AI is on the cusp of a revolution. It's about to become more powerful, more efficient, and more human. By 2030, AI-powered computers are expected to become so adept at solving complex problems that they'll accelerate human progress in areas like computing, medicine, and entertainment.\n\nFrom there, we're likely to see numerous exciting, life-changing advancements, such as cutting-edge real-time AI systems, which are performing tasks at a level where human approval is unnecessary. These systems are not only delivering exceptional performance but also avoiding human impact, allowing us to focus on higher-level thinking and creativity.\n\nRight now, AI is already manifesting as a more empathetic, AI-powered human. It's going to help us become more compassionate, more compassionate. We're going to see the emergence of an AI that's more understanding, more empathetic. It's going to develop a sense of self-awareness, which will lead to more elevated human potential.\n\nAI is also going to give rise to more intuitive interfaces, making computing more accessible and user-friendly. The AI will learn more about our habits and preferences, adapting our algorithms to our needs in real-time, so we don't have to constantly worry about making decisions ourselves.\n\nAnother trend that's emerging is the intersection of AI with creative industries. Video games, for example, will continue to evolve, and will likely become more immersive. The idea of creating immersive digital experiences will become a thing of the future. Not only will our ability to communicate be enhanced through AI-powered tools, but also our capacity for creative expression will expand.\n\nLastly, AI is going to help us find and develop new sources of inspiration. The AI will be constantly generating content that sparks creativity and innovation. The music generation tools AI is already creating have a rich variety of styles and genres. We're likely to see the emergence of entirely new forms of art, music, and even entire ecosystems of expression.\n\nSo, future AI is likely to mark a lifetime of great achievements. Building humans beyond recognition, designing machines capable of unconditional emotions, developing an unprecedented understanding of emotional intelligence. It's a future that's awe-inspiring, solemn, exhilarating, and at times, downright terrifying.", tool_calls=None)



## Creating a web app and microservice for generating responses

We will be using [Gradio](https://github.com/gradio-app/gradio) as web application tool to create a demo interface for our RAG pipeline. We can develop this locally and then easily deploy it to Hugging Face Spaces. Lastly, we can use the Gradio client as SDK to directly interact our RAG pipeline. We are still using the FREE CPU tier of Hugging Face Spaces, so it may take a couple of second to respond. Instead the [ServerlessInference API](https://huggingface.co/docs/api-inference/index), deploy your own [dedicated inference server](https://huggingface.co/inference-endpoints/dedicated) or increase the compute of your Hugging Face Spaces.

### Creating the web app



```python
def generate(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 4000,
    temperature: float = 0.2,
    top_p: float = 0.95,
    top_k: int = 40,
):
    return generate_response_transformers(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )


with gr.Blocks() as demo:
    gr.Markdown("""# RAG - generate

                Generate a response to a query using a [HuggingFaceTB/SmolLM2-360M-Instruct and llama-cpp-python](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF?library=llama-cpp-python).

                Part of [ai-blueprint](https://github.com/davidberenstein1957/ai-blueprint) - a blueprint for AI development, focusing on applied examples of RAG, information extraction, analysis and fine-tuning in the age of LLMs and agents.""")

    with gr.Row():
        system_prompt = gr.Textbox(
            label="System prompt", lines=3, value="You are a helpful assistant."
        )
        user_prompt = gr.Textbox(label="Query", lines=3)

    with gr.Accordion("kwargs"):
        with gr.Row(variant="panel"):
            max_tokens = gr.Number(label="Max tokens", value=512)
            temperature = gr.Number(label="Temperature", value=0.2)
            top_p = gr.Number(label="Top p", value=0.95)
            top_k = gr.Number(label="Top k", value=40)

    submit_btn = gr.Button("Submit")
    response_output = gr.Textbox(label="Response", lines=10)

    submit_btn.click(
        fn=generate,
        inputs=[
            user_prompt,
            system_prompt,
            max_tokens,
            temperature,
            top_p,
            top_k,
        ],
        outputs=[response_output],
    )

demo.launch()
```

    * Running on local URL:  http://127.0.0.1:7867

    To create a public link, set `share=True` in `launch()`.



<div><iframe src="http://127.0.0.1:7867/" width="100%" height="500" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>









<iframe
	src="https://ai-blueprint-rag-generate.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>

### Deploying the web app to Hugging Face

We can now [deploy our Gradio application to Hugging Face Spaces](https://huggingface.co/new-space?sdk=gradio&name=rag-generate).

-  Click on the "Create Space" button.
-  Copy the code from the Gradio interface and paste it into an `app.py` file. Don't forget to copy the `generate_response_*` function, along with the code to execute the generate function.
-  Create a `requirements.txt` file with `gradio`, `gradio-client` and `llama-cpp-python`.
-  Set a Hugging Face API as `HF_TOKEN` secret variable in the space settings, if you are using the Inference API.

We wait a couple of minutes for the application to deploy et voila, we have [a public generate interface](https://huggingface.co/spaces/ai-blueprint/rag-generate)!

### Gradio as REST API

We can now use the [Gradio client as SDK](https://www.gradio.app/guides/getting-started-with-the-python-client) to directly interact with our generate function. Each Gradio app has a API documentation that describes the available endpoints and their parameters, which you can access from the button at the bottom of the Gradio app's space page. We will see it is not the fastest, running on free tier Hugging Face Spaces but it is a good baseline.


```python
client = Client("ai-blueprint/rag-generate")
result = client.predict(
	user_prompt="What is the future of AI?",
	system_prompt="You are a helpful assistant.",
	max_tokens=512,
	temperature=0.2,
	top_p=0.95,
	top_k=40,
	api_name="/generate"
)
result
```

    Loaded as API: https://ai-blueprint-rag-generate.hf.space ✔





    "{'id': 'chatcmpl-38bd4960-655c-447a-be2b-5fc50bb1789e', 'object': 'chat.completion', 'created': 1737652209, 'model': '/home/user/.cache/huggingface/hub/models--prithivMLmods--SmolLM2-135M-Instruct-GGUF/snapshots/5dc548ea9191fd97d817832f51012ae86cded1b5/./SmolLM2-135M-Instruct.Q5_K_M.gguf', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The future of AI is multifaceted and continues to evolve at a rapid pace. As we move forward, AI is being used to augment human capabilities, enhance our understanding of the world, and improve our quality of life. Here are some of the most significant developments and innovations that are shaping the future of AI:\\n\\nAI is being used to enhance human capabilities, such as language translation, image recognition, and decision-making. This has the potential to revolutionize the way we interact with each other and the world at large.\\n\\nAI is also being used to enhance our understanding of the world, such as by providing insights into the human condition, the impact of technology on society, and the ethics of AI.\\n\\nIn the field of healthcare, AI is being used to improve diagnostic accuracy, enhance patient care, and optimize treatment plans.\\n\\nIn the field of education, AI is being used to improve learning outcomes, enhance teaching methods, and optimize learning experiences.\\n\\nAI is also being used to improve the quality of life, such as by providing personalized recommendations, improving productivity, and enhancing the quality of life for people with disabilities.\\n\\nIn the field of transportation, AI is being used to improve safety, enhance safety features, and optimize transportation systems.\\n\\nIn the field of healthcare, AI is being used to improve diagnostic accuracy, enhance treatment planning, and optimize healthcare systems.\\n\\nIn the field of education, AI is being used to improve learning outcomes, enhance teaching methods, and optimize learning experiences.\\n\\nIn the field of healthcare, AI is being used to improve diagnostic accuracy, enhance treatment planning, and optimize healthcare systems.\\n\\nIn the field of transportation, AI is being used to improve diagnostic accuracy, enhance treatment planning, and optimize transportation systems.\\n\\nIn the field of healthcare, AI is being used to improve diagnostic accuracy, enhance treatment planning, and optimize transportation systems.\\n\\nIn the field of education, AI is being used to improve diagnostic accuracy, enhance treatment planning, and optimize transportation systems.\\n\\nIn the field of healthcare, AI is being used to improve diagnostic accuracy, enhance treatment planning, and optimize transportation systems.\\n\\nIn the field of transportation, AI is being used to improve diagnostic accuracy, enhance treatment planning, and optimize transportation systems.\\n\\nIn the field of healthcare, AI is being used to improve diagnostic accuracy, enhance treatment planning, and optimize'}, 'logprobs': None, 'finish_reason': 'length'}], 'usage': {'prompt_tokens': 27, 'completion_tokens': 485, 'total_tokens': 512}}"



## Conclusion

We have seen how to create a generate function using the `llama-cpp-python` library and how to deploy it as a microservice on Hugging Face Spaces. Next we will how we combine the R-A-G-components into a single RAG pipeline.

## Next Steps

- Continue - with [Combine all the components in a RAG pipeline](https://github.com/huggingface/ai-blueprint/blob/main/rag/pipeline.ipynb).
- Contribute - missing something? PRs are always welcome.
- Learn - theories behind the approaches in [Hugging Face courses](https://huggingface.co/learn) or [smol-course](https://github.com/huggingface/smol-course?tab=readme-ov-file).
- Explore - notebooks with similar techniques on [the Hugging Face Cookbook](https://huggingface.co/learn/cookbook/index).

