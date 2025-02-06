# Augment retrieval results by reranking using Sentence Transformers

Retrievals are quick estimates of the most relevant documents to a query which works fine for a first pass over millions of documents, but we can improve this relevance by reranking the retrieved documents. We will build a reranker which can be used in a RAG pipeline together with the retrieval microservice of the [Index and retrieve documents for vector search using Sentence Transformers and DuckDB](https://github.com/huggingface/ai-blueprint/blob/main/rag/retrieve.ipynb) notebook. At the end we will deploy a microservice that can be used to perform reranking of documents based on a query.

## Dependencies and imports

Let's install the necessary dependencies.


```python
!pip install gradio gradio-client pandas sentence-transformers -q
```

Now let's import the necessary libraries.


```python
import gradio as gr
import pandas as pd

from gradio_client import Client
from sentence_transformers import CrossEncoder
```

## Hugging Face as a vector search backend

A brief recap of the previous notebook, we use Hugging Face as vector search backend and can call it as a REST API through the Gradio Python Client.


```python
gradio_client = Client("https://ai-blueprint-rag-retrieve.hf.space/")


def similarity_search(query: str, k: int = 5) -> pd.DataFrame:
    results = gradio_client.predict(api_name="/similarity_search", query=query, k=k)
    return pd.DataFrame(data=results["data"], columns=results["headers"])


similarity_search("What is the future of AI?", k=5)
```

    Loaded as API: https://ai-blueprint-rag-retrieve.hf.space/ ✔





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>url</th>
      <th>text</th>
      <th>distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>https://www.bbc.com/news/technology-51064369</td>
      <td>The last decade was a big one for artificial i...</td>
      <td>0.281200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>http://www.bbc.co.uk/news/technology-25000756</td>
      <td>Singularity: The robots are coming to steal ou...</td>
      <td>0.365842</td>
    </tr>
    <tr>
      <th>2</th>
      <td>http://www.bbc.com/news/technology-25000756</td>
      <td>Singularity: The robots are coming to steal ou...</td>
      <td>0.365842</td>
    </tr>
    <tr>
      <th>3</th>
      <td>https://www.bbc.co.uk/news/technology-37494863</td>
      <td>Google, Facebook, Amazon join forces on future...</td>
      <td>0.380820</td>
    </tr>
    <tr>
      <th>4</th>
      <td>https://www.bbc.co.uk/news/technology-37494863</td>
      <td>Google, Facebook, Amazon join forces on future...</td>
      <td>0.380820</td>
    </tr>
  </tbody>
</table>
</div>



## Reranking retrieved documents

Whenever we retrieve documents from the vector search backend, we can improve the quality of the documents that we pass to the LLM. We do that by ranking the documents by relevance to the query. We will use the [sentence-transformers library](https://huggingface.co/sentence-transformers). You can find the best models to do this, using the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard). 

We will first retrieve 50 documents and then use [sentence-transformers/all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) to rerank the documents and return the top 5.


```python
reranker = CrossEncoder("sentence-transformers/all-MiniLM-L12-v2")


def rerank(query: str, documents: pd.DataFrame) -> pd.DataFrame:
    documents = documents.copy()
    documents = documents.drop_duplicates("text")
    documents["rank"] = reranker.predict([[query, hit] for hit in documents["text"]])
    documents = documents.sort_values(by="rank", ascending=False)
    return documents


query = "What is the future of AI?"
documents = similarity_search(query, k=50)
reranked_documents = rerank(query=query, documents=documents)
reranked_documents[:5]
```

    Some weights of BertForSequenceClassification were not initialized from the model checkpoint at sentence-transformers/all-MiniLM-L12-v2 and are newly initialized: ['classifier.bias', 'classifier.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>url</th>
      <th>text</th>
      <th>distance</th>
      <th>rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>http://www.bbc.com/news/world-us-canada-39425862</td>
      <td>Vector Institute is just the latest in Canada'...</td>
      <td>0.424994</td>
      <td>0.508780</td>
    </tr>
    <tr>
      <th>12</th>
      <td>http://www.bbc.com/news/business-34266425</td>
      <td>Google’s Demis Hassabis – misuse of artificial...</td>
      <td>0.442649</td>
      <td>0.508423</td>
    </tr>
    <tr>
      <th>19</th>
      <td>http://news.bbc.co.uk/2/hi/uk_news/england/wea...</td>
      <td>A group of scientists in the north-east of Eng...</td>
      <td>0.484410</td>
      <td>0.508336</td>
    </tr>
    <tr>
      <th>21</th>
      <td>https://www.bbc.com/news/technology-47668476</td>
      <td>How Pope Francis could shape the future of rob...</td>
      <td>0.494108</td>
      <td>0.508200</td>
    </tr>
    <tr>
      <th>42</th>
      <td>http://news.bbc.co.uk/2/hi/technology/6583893.stm</td>
      <td>Scientists have expressed concern about the us...</td>
      <td>0.530431</td>
      <td>0.507771</td>
    </tr>
  </tbody>
</table>
</div>



We can see the returned documents have slightly shifted in the ranking, which is good, because we see that our reranking works.

## Creating a web app and microservice for reranking

We will be using [Gradio](https://github.com/gradio-app/gradio) as web application tool to create a demo interface for our reranking. We can develop this locally and then easily deploy it to Hugging Face Spaces. Lastly, we can use the Gradio client as SDK to directly interact with our reranking microservice.

### Creating the web app


```python
with gr.Blocks() as demo:
    gr.Markdown("""# RAG - Augment 
                
                Applies reranking to the retrieved documents using [sentence-transformers/all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)
                
                Part of [AI blueprint](https://github.com/davidberenstein1957/ai-blueprint) - a blueprint for AI development, focusing on applied examples of RAG, information extraction, analysis and fine-tuning in the age of LLMs and agents..""")

    query_input = gr.Textbox(
        label="Query", placeholder="Enter your question here...", lines=3
    )
    documents_input = gr.Dataframe(
        label="Documents", headers=["text"], wrap=True, interactive=True
    )

    submit_btn = gr.Button("Submit")
    documents_output = gr.Dataframe(
        label="Documents", headers=["text", "rank"], wrap=True
    )

    submit_btn.click(
        fn=rerank_documents,
        inputs=[query_input, documents_input],
        outputs=[documents_output],
    )

demo.launch(share=False) # share=True is used to share the app with the public
```

    * Running on local URL:  http://127.0.0.1:7862
    
    To create a public link, set `share=True` in `launch()`.



<div><iframe src="http://127.0.0.1:7862/" width="100%" height="500" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>





    



<iframe
	src="https://ai-blueprint-rag-augment.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>

### Deploying the web app to Hugging Face

We can now [deploy our Gradio application to Hugging Face Spaces](https://huggingface.co/new-space?sdk=gradio&name=rag-augment).

-  Click on the "Create Space" button.
-  Copy the code from the Gradio interface and paste it into an `app.py` file. Don't forget to copy the `generate_response_*` function, along with the code to execute the RAG pipeline.
-  Create a `requirements.txt` file with `gradio-client` and `sentence-transformers`.
-  Set a Hugging Face API as `HF_TOKEN` secret variable in the space settings, if you are using the Inference API.

We wait a couple of minutes for the application to deploy et voila, we have [a public reranking interface](https://huggingface.co/spaces/ai-blueprint/rag-augment)!

### Using the web app as a microservice

We can now use the [Gradio client as SDK](https://www.gradio.app/guides/getting-started-with-the-python-client) to directly interact with our RAG pipeline. Each Gradio app has a API documentation that describes the available endpoints and their parameters, which you can access from the button at the bottom of the Gradio app's space page.


```python
client = Client("https://ai-blueprint-rag-augment.hf.space/")

df = similarity_search("What is the future of AI?", k=10)
data = client.predict(
    query="What is the future of AI?",
    documents={"headers": df.columns.tolist(), "data": df.values.tolist(), "metadata": None},
    api_name="/rerank",
)
pd.DataFrame(data=data["data"], columns=data["headers"])
```

    Loaded as API: https://ai-blueprint-rag-augment.hf.space/ ✔





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>url</th>
      <th>text</th>
      <th>distance</th>
      <th>rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>https://www.bbc.co.uk/news/business-48139212</td>
      <td>Artificial intelligence (AI) is one of the mos...</td>
      <td>0.407243</td>
      <td>0.511831</td>
    </tr>
    <tr>
      <th>1</th>
      <td>http://www.bbc.com/news/technology-39657505</td>
      <td>Ted 2017: The robot that wants to go to univer...</td>
      <td>0.424357</td>
      <td>0.509631</td>
    </tr>
    <tr>
      <th>2</th>
      <td>http://www.bbc.com/news/world-us-canada-39425862</td>
      <td>Vector Institute is just the latest in Canada'...</td>
      <td>0.424994</td>
      <td>0.508584</td>
    </tr>
    <tr>
      <th>3</th>
      <td>https://www.bbc.co.uk/news/technology-37494863</td>
      <td>Google, Facebook, Amazon join forces on future...</td>
      <td>0.380820</td>
      <td>0.507728</td>
    </tr>
    <tr>
      <th>4</th>
      <td>https://www.bbc.com/news/technology-51064369</td>
      <td>The last decade was a big one for artificial i...</td>
      <td>0.281200</td>
      <td>0.506788</td>
    </tr>
    <tr>
      <th>5</th>
      <td>http://www.bbc.co.uk/news/technology-25000756</td>
      <td>Singularity: The robots are coming to steal ou...</td>
      <td>0.365842</td>
      <td>0.506259</td>
    </tr>
    <tr>
      <th>6</th>
      <td>https://www.bbc.com/news/technology-52415775</td>
      <td>UK spies will need to use artificial intellige...</td>
      <td>0.414651</td>
      <td>0.505149</td>
    </tr>
  </tbody>
</table>
</div>



## Conclusion

We have seen how to create a reranker using the sentence-transformers library and how to deploy it as a microservice on Hugging Face Spaces. Next steps will be to create a model that can be used to generate a response to a query.

## Next Steps

- Continue - with [Generate a responses based on retrieved documents using a SmolLM](https://github.com/huggingface/ai-blueprint/blob/main/rag/generate.ipynb).
- Contribute - missing something? PRs are always welcome.
- Learn - theories behind the approaches in [Hugging Face courses](https://huggingface.co/learn) or [smol-course](https://github.com/huggingface/smol-course?tab=readme-ov-file).
- Explore - notebooks with similar techniques on [the Hugging Face Cookbook](https://huggingface.co/learn/cookbook/index).
