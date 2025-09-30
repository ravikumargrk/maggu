MODEL='mxbai-embed-large:335m'
DB_PATH=r'./embedded_data/faiss'

import ollama
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model=MODEL, base_url="http://localhost:11434")

# load metadata
import json 
with open(r'raw_data/DMAS_CHUNK_META.json', 'r') as metadatas_fp:
    metadatas = json.load(metadatas_fp)

texts = []
ids = [str(i) for i in range(len(metadatas))]

print('Loading documents: ')
for chunk_meta in metadatas:
    chunk_path = r'raw_data/chunks/' + chunk_meta['filename']
    with open(chunk_path, 'r') as cf:
        text = cf.read()
        texts.append(text)
print('Done.')
print('Adding to db: ', end='')
vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
vectorstore.save_local(DB_PATH)

print('Done')