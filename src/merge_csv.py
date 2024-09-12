import pandas as pd

simple_retr = pd.read_csv("hybrid_retrieval_samples_test.csv")
hybrid_retr = pd.read_csv("hybrid_retrieval_samples_test.csv")

hybrid_retr['simple_retrieval'] = ''
print(hybrid_retr.head())

hybrid_retr['simple_retrieval'] = simple_retr['hybrid_retrieval_docs']
print(hybrid_retr.head())
