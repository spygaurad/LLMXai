from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.output_parsers import StrOutputParser
import csv
import pandas as pd

DB_DIR = './db'
DB_NAME = 'linear_concept_db'

""" Convert json to vector db """
# json_file_path = "linear_data.json"
# loader = JSONLoader(json_file_path, jq_schema='.[].Tokens', text_content=False)
# documents = loader.load()

# vector_store = Chroma.from_documents(
#     documents = documents,
#     collection_name = DB_NAME,
#     persist_directory = DB_DIR,
#     embedding = GPT4AllEmbeddings(model_name="all-MiniLM-L6-v2.gguf2.f16.gguf"),
# )
# vector_store.persist()

vector_store = Chroma(
    collection_name = DB_NAME,
    persist_directory = DB_DIR, 
    embedding_function=GPT4AllEmbeddings(model_name="all-MiniLM-L6-v2.gguf2.f16.gguf")
)
retriever = vector_store.as_retriever()

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

rag_template = """
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
Your are a retriever assistant who convert user question into a concised sentense or a set of words.
Your response will then be used as a prompt to search relevant documents from a vector database using cosine similarity.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Make sure your response is as concise as possible.
Your response should only contain relevent information.
An example entry from the database which your prompt will be used to compare against looks like:
"concept": ..., "tokens": "word_1, word_2, word_3, ...", "summary": "summary of the tokens", "model": "llm models", "layer": "4"

Explaination of the entris:
Concept: Morphology/Semantic/Syntax/Phonology that encoded in the attention head.
Explainable: You may disregard this entry.
Tokens: Formated as "Words: word_1, word_2, word_3, ...". This contains top 15 words that most activate the neurons.
Summary: The summary of the Tokens.
Model: The LLM model to which the document belongs.
Size: The size of the LLM model to which the document belongs.
Layer: The specific transformer layer of the model to which the document belongs.

For example, if user prompted "Does vicuna model has the concept of C++?",
Your response might be: "vicuna, C++" or "vicuna, GCC" or "vicuna, C", etc.

For another example, if user prompt "Does llama model has the concept of Progamming?"
Your response might be: "llama, C++, Python, javascript" or "llama, code, program", etc.

Here is the user question: {question}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

prompt = PromptTemplate(template=template, input_variables=['question', 'document'])
retrieve_prompt = PromptTemplate(template=rag_template, input_variables=['question'])

print(prompt)
model = ChatOllama(model='llama3', temperature=0)
# model2 = ChatOllama(model='llama3', temperature=0)
retrival_chatter = prompt | model | StrOutputParser()
retrive_summerizer = retrieve_prompt | model | StrOutputParser()


questions = pd.read_csv('questions.csv')
results = []
question_list = [list(questions['questions'])[18]]
# print(question_list)
# quit()
# '''
for question in question_list:

    rag_prompt = retrive_summerizer.invoke({'question': question})
    # print(f'\nRAG_Prompt: {rag_prompt}')

    docs = retriever.invoke(rag_prompt)
    doc_txt = "\n".join([ doc.page_content for doc in docs ])
    print(doc_txt)
    # ans = retrival_chatter.invoke({'question': question, 'document': doc_txt})
    # results.append((question, doc_txt, ans))

filename = "hybrid_retrieval_samples.csv"
with open(filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(["Question", "Simple Retriever", "Answer"])
    # Write the data
    writer.writerows(results)
