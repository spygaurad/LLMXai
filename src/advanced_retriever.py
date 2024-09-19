from pinecone import Pinecone, PodSpec
import os
import pandas as pd
from dotenv import load_dotenv
from tqdm.auto import tqdm
from langchain import PromptTemplate, LLMChain
from indexing.db_setup import Database
import requests
import json
import re
import asyncio


load_dotenv()


# Define the template for generating the prompt
template1 = """<s>[INST] You are a key value extraction assistant. The user enters a text, your task is to extract the key-value pairs from the text.
{examples}
Based on above example responses, provide a dictionary of key (model and layer) and their value. Give short and clear answer. [/INST]
User: {question}
Assistant:"""


template2 = """<s>[INST] You will have information in json format. Based on values of the json, answer the question asked by the user
Json Information:
{context}

Above information is present in {model} model in its {layer} layer. Based on this answer following question.
User: {question}
Assistant:"""


# Define the generation examples
generation_examples = """
{
Example 1:
User: Does Vicuna model has information about mount everest?
Assistant: {
  "model": "Vicuna",
  "layer": None
}
},
{
Example 2:
User: What is the concept captured by the 30th layer of mistral?
Assistant: {
  "model": "mistral",
  "layer": 30
}
}
"""

def generate_template_1(question,template, examples):
    return template.format(question=question, examples=examples)

def generate_template_final(question, template,context, model, layer):
    return template.format(question=question, context=context, model=model, layer=layer)


def generate_response(prompt,api_url):
    # Prepare the prompt with the given template and examples
    
    # Prepare the payload for the API request
    payload = {
        "prompt": prompt,
        "max_tokens": 50,  # Adjust as needed
        "temperature": 0.5  # Adjust as needed
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # Make the API request
    response = requests.post(api_url, headers=headers, data=json.dumps(payload))
    
    # Check if the request was successful
    if response.status_code == 200:
        result = response.json()
        # Extract the text response from the result
        return result.get("choices", [{}])[0].get("text", "").strip()
    else:
        # Handle errors
        return f"Error: {response.status_code} - {response.text}"

def clean_response(input_string, question):
    # Find the starting index of the question
    index = input_string.find(question)
    return input_string[index:].split("User")[0].split("Assistant")[-1]


def extract_models_layers(text):
    model_pattern = r'"model":\s*"([^"]+)"'
    layer_pattern = r'"layer":\s*(\d+|None)'
    model_matches = re.findall(model_pattern, text)
    layer_matches = re.findall(layer_pattern, text)
    layers = [int(layer) if layer.isdigit() else None for layer in layer_matches]

    return model_matches[0], layers[0]


# Example usage
async def main():
    db = Database()
    await db.connect()
    api_url = ""  # Replace with your API endpoint
    question = "What information does the 31st layer of Vicuna model capture?"

    generation_prompt = generate_template_1(question, template1, generation_examples)
    generated_response = generate_response(generation_prompt, api_url)
    final_res = clean_response(generated_response, question)
    model, layer = extract_models_layers(final_res)
    model = model.lower()

    context = await db.semantic_with_keyword_search(question, model)

    final_prompt = generate_template_final(question, template2, context, model, layer)
    final_response = generate_response(question, template2)
    print(final_response)

asyncio.run(main())






