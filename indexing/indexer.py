# from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
# from llama_index.embeddings import LangchainEmbedding
# # Create the embedding model using the HuggingFaceBgeEmbeddings class

# embedding_model_name = "BAAI/bge-large-en-v1.5"


# embed_model = LangchainEmbedding(
#   HuggingFaceBgeEmbeddings(model_name=embedding_model_name)
# )

# # Get the embedding dimension of the model by doing a forward pass with a dummy input
# embed_dim = len(embed_model.get_text_embedding("Hello world")) # 1024

