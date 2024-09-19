from pinecone import Pinecone, PodSpec
import os
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone_text.sparse import BM25Encoder
from tqdm.auto import tqdm
from indexing.db_setup import Database
load_dotenv()



db = Database()
await db.connect()



