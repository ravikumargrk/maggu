# page meta:
# {
#     "title": "Page title Main",
#     "children": [], # list[int]
#     "chunks": [
#         {
#             "chunk_data_filename" : "chunk_filename.md",
#             "page_links" : [ ],  # list[int]
#             "images_idx" : [ ],  # list[int]
#         },
#         ...
#     ]
# }

MODEL='embeddinggemma:300m'
import ollama
import chromadb
import os 
os.system(r'rm -rf embedded_data/chroma_db')

# Start client (creates local .chroma DB folder)
client = chromadb.PersistentClient(path=r'./embedded_data/chroma_db')

# should be a variable
collection = client.create_collection('DMAS')
import json 
with open(r'raw_data/meta_data.json', 'r') as meta_data_fp:
    meta_data = json.load(meta_data_fp)

# Insert into DB
from tqdm import tqdm
page_idx = 0
for page in tqdm(meta_data):
    chunk_idx = 0
    for chunk in page['chunks']:
        filename = chunk['chunk_data_filename']
        chunk_path = r'raw_data/chunks/' + filename
        with open(chunk_path) as cf:
            text = cf.read()
            text_embedding = ollama.embeddings(model=MODEL, prompt=text)["embedding"]
            collection.add(documents=[text], embeddings=[text_embedding], ids=[f"{page_idx}/{chunk_idx}"])
        chunk_idx += 1
    page_idx += 1
