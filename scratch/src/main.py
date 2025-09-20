import sys
from pathlib import Path
from .document_loader import load_documents
from .text_splitter import TextSplitter
from .vector_store import VectorStore
from .retriever import Retriever

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

def main():
    splitter = TextSplitter()
    store = VectorStore()
    retriever = Retriever(store)

    print("Simple RAG CLI (commands: add, search, info, quit).")
    print("Quick use: type a question directly or 'search your query'.")

    while True:
        raw = input("> ").strip()
        if not raw:
            continue
        cmd_lower = raw.lower()

        # Direct exit
        if cmd_lower in ("quit", "exit"):
            break

        # Inline search pattern: search <query>
        if cmd_lower.startswith("search "):
            query = raw[len("search "):].strip()
            _run_search(retriever, query)
            continue

        # Exact commands
        if cmd_lower == "search":
            query = input("Query: ").strip()
            if query:
                _run_search(retriever, query)
            continue

        if cmd_lower == "add":
            path = input("File path (.pdf/.txt): ").strip()
            _run_add(path, splitter, store)
            continue

        if cmd_lower == "info":
            print(store.stats())
            continue

        if cmd_lower == "help":
            print("Commands: add, search, info, quit (or just type a question).")
            continue

        # Fallback: treat entire line as a search query
        _run_search(retriever, raw)

    print("Done.")

def _run_add(path: str, splitter: TextSplitter, store: VectorStore):
    from .document_loader import load_documents
    from pathlib import Path
    if not Path(path).exists():
        print("Not found.")
        return
    try:
        docs = load_documents(path)
        chunks = splitter.split(docs)
        store.add_documents(chunks)
        print(f"Added {len(chunks)} chunks.")
    except Exception as e:
        print(f"Add error: {e}")

def _run_search(retriever: Retriever, query: str, k: int = 4):
    if not query:
        print("Empty query.")
        return
    results = retriever.retrieve(query, k=k)
    if not results:
        print("No results.")
        return
    print(retriever.build_context(results))

if __name__ == "__main__":
    main()