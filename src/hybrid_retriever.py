from pinecone import Pinecone, PodSpec
import os
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone_text.sparse import BM25Encoder
from tqdm.auto import tqdm
import csv
import gradio as gr
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate


# Instantiate Pinecone instance with API key
load_dotenv()

# Access the environment variables
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))


df = pd.read_json('linear_concept_seed0.json')
df['metadata'] = df.apply(lambda row: f"Concept: {row['Concept']}; Tokens: {row['Tokens']}; Summary: {row['Summary']}; Model: {row['Model']}; Size: {row['Size']}", axis=1)

bm25 = BM25Encoder()
bm25.fit(df['metadata'])

model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1',device='cpu')

index = pc.Index(host='https://xai-k9ahswn.svc.gcp-starter.pinecone.io')
llm_model = ChatOllama(model='phi3.5', temperature=0)

# create sparse and dense vectors


# search
# while (question := input("Q: ")) != '/exit':
#     sparse = bm25.encode_queries(question)
#     dense = model.encode(question).tolist()
#     # result = index.query(top_k=5,vector=dense,sparse_vector=sparse,include_metadata=True)
#     result = index.query(top_k=5,vector=dense,include_metadata=True)

#     print(f'\nRetrieved Table Data: \n\n {result}\n')

questions = pd.read_csv('questions.csv')
results = []
question_list = list(questions['questions'])
# '''

def get_retriever_result(question):
    sparse = bm25.encode_queries(question)
    dense = model.encode(question).tolist()
    # result = index.query(top_k=5,vector=dense,sparse_vector=sparse,include_metadata=True)
    result = index.query(top_k=5,vector=dense,include_metadata=True)
    return result

# for question in question_list:
#     result = get_retriever_result(question)
#     results.append((question,result))

template = """
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are an assistant who answers questions based on the retrived document. 
If the document is not related to the question, tell the user you did not find relavent information. 
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Here is the retrieved document: \n\n {document} \n\n

Explanation for the entries in the documents:
Concept: Morphology/Semantic/Syntax/Phonology that encoded in the attention head.
Explainable: You may disregard this entry.
Tokens: Formated as "Words: word_1, word_2, word_3, ...". This contains top 15 words that most activate the neurons.
Summary: The summary of the Tokens.
Model: The LLM model to which the document belongs.
Size: The size of the LLM model to which the document belongs.
Layer: The specific transformer layer of the model to which the document belongs.

You may disregard the rest of the entries.

Here is the user question: {question} 

Your response MUST be formated as following:
According to the documents, ...

[Reference]:
Document 1: ... (reiterate the full document)
Document 2: ...
...
Document 5: ...
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

prompt = PromptTemplate(template=template, input_variables=['question', 'document'])
retrival_chatter = prompt | llm_model | StrOutputParser()

app1 = gr.Interface(
    fn=get_retriever_result,
    #inputs=gr.inputs.Audio(label="Upload audio file", type="filepath"),
    inputs=[gr.Text('Question', value="User Question", label="Input Question"),
            ],     
    outputs="text",
    title="LLM XAI Chat",
    examples=[
            ["Does Llama Model have knowledge of C programming?"],
            ["What is the concept learned by 21st layer of Vicuna?" ],
            ['What is the height of Mt. Everest?'],
    ]
    
)

def reload_inputs(text, llm_output, retrieved_doc):
    retrieved_doc = get_retriever_result(text)
    return text, "", retrieved_doc

def get_llm_response(text, llm_output, retrieved_doc):
    print("Retriever Doc: ", retrieved_doc)
    llm_response = retrival_chatter.invoke({'question': text, 'document': retrieved_doc})
    return llm_response

with gr.Blocks() as app1:
    with gr.Column():
        with gr.Row():
            with gr.Column():
                text = gr.Text('Question', value="User Question", label="Input Question")
                retrieve_documents = gr.Button(value="Retrieve Document")

            with gr.Column():
                llm_output = gr.Text("", label="LLM Result", min_width=250)
        with gr.Row():
            retrieved_doc = gr.Text("", label="Rerrieved Docs")

        with gr.Column():
            btn = gr.Button(value="Submit")
            btn.click(get_llm_response, inputs=[text,llm_output,retrieved_doc], outputs=[llm_output])
            retrieve_documents.click(reload_inputs, inputs=[text,llm_output,retrieved_doc], outputs=[text,llm_output,retrieved_doc])



# gr.ChatInterface(get_retriever_result).launch(share=True)
demo = gr.TabbedInterface([app1], ["LLM XAI"])
demo.launch(server_name="0.0.0.0",server_port=8013, debug=True, share=False, ssl_verify=False)


# filename = "hybrid_retrieval_samples.csv"
# with open(filename, mode='w', newline='', encoding='utf-8') as file:
#     writer = csv.writer(file)
#     # Write the header
#     writer.writerow(["Question", "Hybrid Retriever"])
#     # Write the data
#     writer.writerows(results)
# '''