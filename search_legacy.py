# [Incomplete]
# query = "How do I use embeddings for retrieval?"
# q_emb = ollama.embeddings(model=MODEL, prompt=query)["embedding"]

# results = collection.query(query_embeddings=[q_emb], n_results=2)
# print(results)

MODEL='mxbai-embed-large:335m'
N_CHUNKS=5
DB_PATH=r'./embedding/embedded_data/faiss'

from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.callbacks import StdOutCallbackHandler

embeddings = OllamaEmbeddings(model=MODEL, base_url="http://localhost:11434")
vectorstore = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

# Use Ollama locally (make sure Ollama is running: `ollama serve`)
llm = ChatOllama(
    model='mistral:latest',    # or nomic-embed-text, mistral, etc.
    temperature=0.7,
    base_url="http://localhost:11434"  # Ollama default API server
)

# set up conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# retriever comes from your VectorStore (e.g., FAISS, Chroma, etc.)
retriever = vectorstore.as_retriever(search_kwargs={'k': 6})

# combine LLM + retriever + memory into a conversation chain
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    # callbacks=[StdOutCallbackHandler()]
)

# Example query
# response = conversation_chain.run("What did we talk about earlier?")
# print(response)

def chat(question, history):
    result = conversation_chain.invoke({"question": question})
    print(result["answer"])
    return result["answer"]

import gradio
view = gradio.ChatInterface(chat, type="messages").launch(inbrowser=True)