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
    df = df[:200]
    await db.insert_data(df)

    query = 'Does Vicuna model has bias towards any programming language?'
    keyword = 'vicuna'

    results = await db.keyword_search(keyword)
    # print(results)
    results = await db.semantic_search(query)
    results = await db.semantic_with_keyword_search(query, keyword)
    print(results)
 
    # Perform queries in parallel
    # results = await asyncio.gather(
    #     db.semantic_search(query),
    #     db.keyword_search(query)
    # )
    # results = db.rerank(query, results)
    # print(results)

asyncio.run(main())




