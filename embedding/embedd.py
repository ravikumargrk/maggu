MODEL='mxbai-embed-large:335m'
import ollama
import chromadb
import os 

from glob import glob
if glob(r'embedded_data/chroma_db'):
    os.system(r'rm -rf embedded_data/chroma_db')

# Start client (creates local .chroma DB folder)
client = chromadb.PersistentClient(path=r'./embedded_data/chroma_db')

collection = client.create_collection('DMAS')
import json 
with open(r'raw_data/DMAS_CHUNK_META.json', 'r') as meta_data_fp:
    meta_data = json.load(meta_data_fp)

# Insert into DB
from tqdm import tqdm
chunk_idx = 0
for chunk_meta in tqdm(meta_data):
    chunk_path = r'raw_data/chunks/' + chunk_meta['filename']
    with open(chunk_path, 'r') as cf:
        text = cf.read()
        text_embedding = ollama.embeddings(model=MODEL, prompt=text)["embedding"]
        meta_ = {'title': '\n'.join(chunk_meta['breadcrumbs'])}
        collection.add(documents=[text], embeddings=[text_embedding], ids=[f"{chunk_idx}"], metadatas=meta_)
    chunk_idx += 1
