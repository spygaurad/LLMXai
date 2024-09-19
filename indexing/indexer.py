import asyncio
from db_setup import Database
import pandas as pd

def load_data(json_path):
    df = pd.read_json(json_path)
    return df



async def main():
    db = Database()
    await db.connect()

    df = load_data(json_path='../data/linear_concept_seed0.json')
    # Concatenate 'concept' and 'tokens' into a single sentence
    df['metadata'] = [f"{concept} - {tokens}" for concept, tokens in zip(df['Concept'], df['Tokens'])]
    await db.create_schema()
    # df = df[:24]
    await db.insert_data(df)

    query = 'Does LLAMA model has bias towards any religion?'
    keyword = 'vicuna'

    results = await db.keyword_search(keyword)
 
    # Perform queries in parallel
    # results = await asyncio.gather(
    #     db.semantic_search(query),
    #     db.keyword_search(query)
    # )
    # results = db.rerank(query, results)
    # print(results)

asyncio.run(main())




