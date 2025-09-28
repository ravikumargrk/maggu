# [Incomplete]
# query = "How do I use embeddings for retrieval?"
# q_emb = ollama.embeddings(model=MODEL, prompt=query)["embedding"]

# results = collection.query(query_embeddings=[q_emb], n_results=2)
# print(results)

MODEL='embeddinggemma:300m'
N_CHUNKS=5
import json 
with open(r'embedding/raw_data/DMAS_CHUNK_META.json', 'r') as meta_data_fp:
    meta_data = json.load(meta_data_fp)

import ollama
import chromadb

client = chromadb.PersistentClient(path=r'embedding/embedded_data/chroma_db', settings=chromadb.Settings(anonymized_telemetry=False))
collection = client.get_collection('DMAS') # this should be variable?

def sim_search(query:str):
    q_emb = ollama.embeddings(model=MODEL, prompt=query)['embedding']
    results = collection.query(query_embeddings=[q_emb], n_results=N_CHUNKS, include=['documents'])
    print(results)
    

import sys 
sim_search(' '.join(sys.argv[1:]))
