from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from llama_index.legacy.embeddings.langchain import LangchainEmbedding
import psycopg2

embedding_model_name = "BAAI/bge-large-en-v1.5"

def load_embed_model():
    embed_model = LangchainEmbedding(
    HuggingFaceBgeEmbeddings(model_name=embedding_model_name)
    )


connection_string = "postgresql://spygaurad:test123@localhost:5432"

# Created a database with psql first
db_name = "llm_xai"
table_name = 'embeddings'


# Connect to the database
conn = psycopg2.connect(connection_string+ f"/{db_name}")
# Set autocommit to True to avoid having to commit after every command
conn.autocommit = True

cursor = conn.cursor()

cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id SERIAL PRIMARY KEY,
        embedding_data BYTEA
    )
""")
               
cursor.execute(f"""
    DROP TABLE {table_name};
""")

# Commit changes if necessary (autocommit is True in this case, so it's optional)
# conn.commit()

# Close the cursor and connection when done
cursor.close()
conn.close()


