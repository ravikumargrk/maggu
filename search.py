# [Incomplete]
# query = "How do I use embeddings for retrieval?"
# q_emb = ollama.embeddings(model=MODEL, prompt=query)["embedding"]

# results = collection.query(query_embeddings=[q_emb], n_results=2)
# print(results)

MODEL='embeddinggemma:300m'
N_CHUNKS=5
import json 
with open(r'embedding/raw_data/meta_data.json', 'r') as meta_data_fp:
    meta_data = json.load(meta_data_fp)

import ollama
import chromadb

client = chromadb.PersistentClient(path=r'embedding/embedded_data/chroma_db')
collection = client.get_collection('DMAS') # this should be variable?

def sim_search(query:str):
    q_emb = ollama.embeddings(model=MODEL, prompt=query)['embedding']
    results = collection.query(query_embeddings=[q_emb], n_results=N_CHUNKS, include=[])
    for idx in results['ids'][0]:
        page_idx, chunk_idx = [int(_idxs) for _idxs in idx.split('/')]
        chunk_meta = meta_data[page_idx]['chunks'][chunk_idx]
        chunk_path = r'embedding/raw_data/chunks/' + chunk_meta['chunk_data_filename']
        print(json.dumps(chunk_meta), end=' ')
        with open(chunk_path) as cf:
            text = cf.read()
            print(len(text))
            print(text)

sim_search('DE 23')