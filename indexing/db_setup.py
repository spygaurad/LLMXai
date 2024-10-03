import asyncio
import itertools
from pgvector.psycopg import register_vector_async
import psycopg
from sentence_transformers import SentenceTransformer, CrossEncoder
from embedder import EmbeddingModel

class Database:
    # Initialize the database connection and the embedding model

    def __init__(self):
        self.conn = None
        self.embedder = EmbeddingModel()  # Embedding model used to generate vector representations
        self.DBNAME = 'vectordb'  # Database name
        self.USER = "postgres"  # Database username
        self.PASSWORD = "password"  # Database password
        self.HOST = "127.0.0.1"  # Database host (local)
        self.PORT = 5434  # Database port
        self.TABLE_DOCUMENTS = 'linear_concept'  # Table name for storing document embeddings
        self.VECTOR_DIM = 1024  # Dimension of the vectors (size of the embeddings)


    async def connect(self):
        # Asynchronously connect to the PostgreSQL database

        self.conn = await psycopg.AsyncConnection.connect(
            dbname = self.DBNAME,
            user = self.USER,
            password = self.PASSWORD,
            host = self.HOST,
            port = self.PORT,
            autocommit=True) # Automatically commit transactions

    async def create_schema(self):
        # Create the schema, register the vector extension, and create the documents table if it doesn't exist

        await self.conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
        await register_vector_async(self.conn)
        
        # Create the table if it does not exist

        await self.conn.execute(f'''
            CREATE TABLE IF NOT EXISTS {self.TABLE_DOCUMENTS} (
                id bigserial PRIMARY KEY, 
                concept text,
                explainable text,
                tokens text,
                summary text,
                model text,
                size text,
                layer text,
                type text,
                head text,
                rank text,
                score numeric,
                topk text,
                metadata text,
                embedding vector({self.VECTOR_DIM})
            )
        ''')

        # await self.conn.execute(f"CREATE INDEX IF NOT EXISTS {self.TABLE_DOCUMENTS} USING GIN (to_tsvector('english', content))")

    async def insert_data(self, df):
        # Insert data into the database from a DataFrame

        # Check if data already exists in the table
        async with self.conn.cursor() as cur:
            await cur.execute(f'SELECT COUNT(*) FROM {self.TABLE_DOCUMENTS}')
            count = (await cur.fetchone())[0]
            if count > 0:
                return  # If data exists, exit the function
                
        # Extract 'metadata' column to embed using the embedding model
        sentences = df['metadata'].tolist()  # List of metadata to be embedded
        embeddings = self.embedder.encode(sentences) # Generate embeddings for each sentence
        embeddings = [embedding.tolist() for embedding in embeddings] # Convert embeddings to lists


        # Prepare SQL query for inserting data into the table
        sql = f'''INSERT INTO {self.TABLE_DOCUMENTS} 
                (concept, explainable, tokens, summary, model, size, layer, type, head, rank, score, topk, metadata, embedding) 
                VALUES ''' + ', '.join(['(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)' for _ in range(len(embeddings))])

        # Flatten the parameters and prepare them for insertion
        params = list(itertools.chain(*[
            (
                row['Concept'], 
                row['Explainable'], 
                row['Tokens'], 
                row['Summary'], 
                row['Model'], 
                row['Size'], 
                row['Layer'], 
                row['Type'], 
                row['Head'], 
                row['Rank'], 
                row['Score'], 
                row['TopK'], 
                row['metadata'],    # Metadata text
                embedding # Corresponding embedding vector
            )
            for (_, row), embedding in zip(df.iterrows(), embeddings)  # Iterate over both DataFrame and embeddings
        ]))

        # Execute the query to insert data into the table
        await self.conn.execute(sql, params)


    async def semantic_search(self, query):
        # Perform a semantic search using a vector similarity match
        query_embedding = self.embedder.encode([query])[0].tolist()  # Ensure it's a list
        
        async with self.conn.cursor() as cur:
            # Find the top 5 most similar embeddings using cosine similarity

            # Use pgvector's <=> operator to find nearest vectors (cosine similarity by default)
            await cur.execute('''
                SELECT id, concept, explainable, tokens, summary, model, size, layer, type, head, rank, score, topk, metadata
                FROM linear_concept
                ORDER BY embedding <=> %s::vector
                LIMIT 5
            ''', (query_embedding,))
            
            # Fetch and return results
            results = await cur.fetchall()
            return results

    async def close(self):
        # Close the database connection

        await self.conn.close()

    async def keyword_search(self, query):
        # Perform a keyword-based search

        async with self.conn.transaction():  # Ensures proper transaction management
            async with self.conn.cursor() as cur:
                # Perform a case-insensitive search on the 'model' field

                await cur.execute('''
                    SELECT id, concept, explainable, tokens, summary, model, size, layer, type, head, rank, score, topk, metadata
                    FROM linear_concept
                    WHERE model ILIKE %s  -- Case-insensitive search
                ''', (f'%{query}%',))
                
                results = await cur.fetchall()
                return results

    async def semantic_with_keyword_search(self, query, keyword):
        # Perform a combined semantic and keyword search
        query_embedding = self.embedder.encode([query])[0].tolist()  # Ensure it's a list

        async with self.conn.transaction():  # Ensures proper transaction management
            async with self.conn.cursor() as cur:
                # Perform a case-insensitive search for the keyword and order by vector similarity

                # Use pgvector's <=> operator to find nearest vectors (cosine similarity by default)
                await cur.execute('''
                    SELECT id, concept, explainable, tokens, summary, model, size, layer, type, head, rank, score, topk, metadata
                    FROM linear_concept
                    WHERE model ILIKE %s  -- Case-insensitive search
                    ORDER BY embedding <=> %s::vector
                    LIMIT 5
                ''', (f'%{keyword}%', query_embedding))
                
                results = await cur.fetchall()
                return results