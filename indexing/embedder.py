import itertools
import requests
# from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel

class EmbeddingModel:
    def __init__(self, model_name='BAAI/bge-m3', api_url=False):
        self.model_name = model_name
        self.api_url = api_url
        self.model = None
        
        if not api_url:
            self.model = BGEM3FlagModel(self.model_name )
    
    def encode(self, sentences):
        if self.api_url:
            return self._encode_api(sentences)
        else:
            return self._encode_local(sentences)
    
    def _encode_local(self, sentence):
        return self.model.encode(sentence)['dense_vecs']
    
    def _encode_api(self, sentences):
        # Assuming the API expects JSON with a 'sentences' key
        response = requests.post(self.api_url, json={'sentences': sentences})
        if response.status_code == 200:
            return response.json()['embeddings']
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")