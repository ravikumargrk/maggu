
MODEL='mxbai-embed-large:335m'
N_CHUNKS=6
DB_PATH=r'./embedding/embedded_data/faiss'
STEP_BACK = True

from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage

embeddings = OllamaEmbeddings(model=MODEL, base_url="http://localhost:11434")
vectorstore = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

# Use Ollama locally (make sure Ollama is running: `ollama serve`)
llm = ChatOllama(
    model='mistral:latest',    # or nomic-embed-text, mistral, etc.
    temperature=0.7,
    base_url="http://localhost:11434"  # Ollama default API server
)

# set up conversation memory

# retriever comes from your VectorStore (e.g., FAISS, Chroma, etc.)
# retriever = vectorstore.as_retriever(search_kwargs={'k': 6})

# combine LLM + retriever + memory into a conversation chain

# Example query
# response = conversation_chain.run("What did we talk about earlier?")
# print(response)


# -------- Utilities --------
def make_stepback_query(question: str) -> str:
    """
    Ask the LLM to produce a broader, higher-level 'step-back' version of the user's question.
    Keep it to one sentence, no preamble.
    """
    system = (
        "You rewrite user questions into a broader, high-level version to improve search recall. "
        "Return ONLY the rewritten query, one sentence, no quotes."
    )
    user = f"Rewrite this question more generally: {question}"
    sb_msg = [SystemMessage(content=system), HumanMessage(content=user)]
    try:
        resp = llm.invoke(sb_msg)  # non-stream call
        text = (resp.content or "").strip()
        # Safety: if the model parrots or returns empty, just skip
        if not text or text.lower() == question.lower():
            return ""
        return text
    except Exception:
        return ""

def chat(question, history):
    if STEP_BACK:
        step_back_query = make_stepback_query(question)
    else:
        step_back_query = question
    db_search_results = vectorstore.similarity_search_with_score(step_back_query, k=N_CHUNKS)
    if not db_search_results:
        yield "I couldn't find anything relevant in the indexed documents."
        return

    for i, (doc, score) in enumerate(db_search_results, start=1):
        md = doc.metadata or {}
        fname = md.get("filename", "<no filename in metadata>")
        print(f"{i} : Source file : {fname} FAISS score : {score:.6f} (lower = more similar)")
        # print("Preview     :", pretty_preview(doc.page_content, CHUNK_PREVIEW_CHARS))
    
    top_docs = [doc for (doc, _score) in db_search_results]
    context = "\n\n---\n\n".join(d.page_content for d in top_docs)

    system_prompt = (
        "You are a careful assistant. Answer the user's question "
        "using ONLY the provided context. If the answer is not in the context, "
        "say you don't know based on the indexed documents. Be concise dont answer anything other than what is needed."
        "Also, Do not tell user that context is provided"
        f"Context (retrieved chunks):\n{context}\n\n"
        "Answer:"
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
    )

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    # answer_chunks = []
    answer = ''
    for chunk in llm.stream(messages):
        if chunk and chunk.content:
            answer += chunk.content
            yield answer

import gradio
view = gradio.ChatInterface(chat).launch(inbrowser=True)