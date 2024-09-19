import asyncio
import itertools
from pgvector.psycopg import register_vector_async
import psycopg
from sentence_transformers import SentenceTransformer, CrossEncoder
from embedder import EmbeddingModel

class Database:

    def __init__(self):
        self.conn = None
        self.embedder = EmbeddingModel()
        self.DBNAME = 'vectordb'
        self.USER = "postgres"
        self.PASSWORD = "password"
        self.HOST = "127.0.0.1"
        self.PORT = 5434
        self.TABLE_DOCUMENTS = 'linear_concept'
        self.VECTOR_DIM = 1024


    async def connect(self):
        self.conn = await psycopg.AsyncConnection.connect(
            dbname = self.DBNAME,
            user = self.USER,
            password = self.PASSWORD,
            host = self.HOST,
            port = self.PORT,
            autocommit=True)

    async def create_schema(self):
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

        # Check if data already exists
        async with self.conn.cursor() as cur:
            await cur.execute(f'SELECT COUNT(*) FROM {self.TABLE_DOCUMENTS}')
            count = (await cur.fetchone())[0]
            if count > 0:
                return  # Data already exists
                # Extract 'Tokens' column for embedding

        sentences = df['metadata'].tolist()  # Assuming you're embedding the 'Tokens' field
        embeddings = self.embedder.encode(sentences)
        embeddings = [embedding.tolist() for embedding in embeddings]


        # Prepare SQL insert for each field
        sql = f'''INSERT INTO {self.TABLE_DOCUMENTS} 
                (concept, explainable, tokens, summary, model, size, layer, type, head, rank, score, topk, metadata, embedding) 
                VALUES ''' + ', '.join(['(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)' for _ in range(len(embeddings))])

        # Prepare parameters by iterating over each row in the DataFrame
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
                row['metadata'],
                embedding 
            )
            for (_, row), embedding in zip(df.iterrows(), embeddings)  # Iterate over both DataFrame and embeddings
        ]))

        # Execute the query with the params
        await self.conn.execute(sql, params)


    async def semantic_search(self, query):
    # Encode the query to get its embedding
        query_embedding = self.embedder.encode([query])[0].tolist()  # Ensure it's a list
        
        async with self.conn.cursor() as cur:
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
        await self.conn.close()

    async def keyword_search(self, query):
        async with self.conn.transaction():  # Ensures proper transaction management
            async with self.conn.cursor() as cur:
                await cur.execute('''
                    SELECT id, concept, explainable, tokens, summary, model, size, layer, type, head, rank, score, topk, metadata
                    FROM linear_concept
                    WHERE model ILIKE %s  -- Case-insensitive search
                ''', (f'%{query}%',))
                
                results = await cur.fetchall()
                return results

    async def semantic_with_keyword_search(self, query, keyword):
    # Encode the query to get its embedding
        query_embedding = self.embedder.encode([query])[0].tolist()  # Ensure it's a list

        async with self.conn.transaction():  # Ensures proper transaction management
            async with self.conn.cursor() as cur:
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