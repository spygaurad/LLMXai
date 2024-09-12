from pinecone import Pinecone, PodSpec
import os
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone_text.sparse import BM25Encoder
from tqdm.auto import tqdm

# Instantiate Pinecone instance with API key
load_dotenv()

# Access the environment variables
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

# Create a pod index. By default, Pinecone indexes all metadata.
'''
pc.create_index( 
  name=os.getenv('PINECONE_INDEX_NAME'), 
  dimension=384, 
  metric='dotproduct', 
  spec=PodSpec( 
    environment='gcp-starter', 
    pod_type='p1.x1', 
    pods=1, ))

# Show the information about the newly-created index
print(pc.describe_index(os.getenv('PINECONE_INDEX_NAME')))

quit()
# '''

df = pd.read_json('linear_concept_seed0.json')
df['metadata'] = df.apply(lambda row: f"Concept: {row['Concept']}; Tokens: {row['Tokens']}; Summary: {row['Summary']};", axis=1)
# df['metadata'] = df.apply(lambda row: f"Summary: {row['Summary']} Concept: {row['Concept']}", axis=1)
print(df['metadata'])
# quit()
bm25 = BM25Encoder()
bm25.fit(df['metadata'])
  
# dense vector

model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1',device='cpu')

batch_size = 32
# Loop through the DataFrame 'df' in batches of size 'batch_size'
for i in tqdm(range(0, len(df), batch_size)):
    i_end = min(i+batch_size, len(df))# Determine the end index of the current batch
    df_batch = df.iloc[i:i_end]# Extract the current batch from the DataFrame
    df_dict = df_batch.to_dict(orient="records")  # Convert the batch to a list of dictionaries
    # Create a batch of metadata by concatenating all columns
    # except 'Filetype', 'Element Type', and 'Date Modified'
    meta_batch = [
        " ".join(map(str, x)) for x in df_batch.loc[
            :, ~df_batch.columns.isin(['Summary','Concept','Size', 'Type', 'Rank','Score','TopK','Head'])
        ].values.tolist()
    ]
    # Extract the 'text' column from the current batch as a list
    text_batch = df['metadata'][i:i_end].tolist()
    # Encode the metadata batch using the bm25 algorithm to create sparse embeddings
    sparse_embeds = bm25.encode_documents([text for text in meta_batch])
    # Encode the text batch using a model to create dense embeddings
    dense_embeds = model.encode(text_batch).tolist()
    # Generate a list of IDs for the current batch
    ids = [str(x) for x in range(i, i_end)]
    # Initialize a list and iterate over each item in the batch to prepare the upsert data
    upserts = []
    for _id, sparse, dense, meta in zip(ids, sparse_embeds, dense_embeds, df_dict):
        upserts.append({
            'id': _id,
            'sparse_values': sparse,
            'values': dense,
            'metadata': meta
        })
    # Connect to the Pinecone index and upsert the batch data
    index = pc.Index(host='https://xai-k9ahswn.svc.gcp-starter.pinecone.io')
    index.upsert(upserts)

