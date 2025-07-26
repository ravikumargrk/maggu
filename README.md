# RAG
A retrieval augmented generation project

## Retrieval Augmented Generation
* RAG (Retrieval-Augmented Generation) is an AI technique that combines document retrieval with text generation to produce accurate, context-aware responses. 
* In this approach, documents are first chunked, embedded into high-dimensional vectors using embedding models, and stored in a vector database. At query time, the input is also embedded and used to retrieve the most relevant document chunks, which are then fed into a generative model to produce the final response.

## Process
### 1. Document Loads
For processing documents converting them into text.
### 2. Split documents into chunks
Use `langchain_text_splitters.RecursiveCharacterTextSplitter` to split document into smaller chunks for storing
### 3. Embedding function
Use same embedding function to store and search, Use `langchain_community.embeddings.ollama.OllamaEmbeddings` to do this completely off-line. 
### 4. Build Database
If you can tag each chunk with a unique chunk id, it will help us to update the database when needed.
You may use `langchain.vectorstores.chroma.Chroma` to store all the chunks.
### 5. Search 
You may use `langchain.vectorstores.chroma.Chroma.similrity_search_with_score` with optimal k (Number of chunks to return)
to search query text in the db for chunks.
Then use the results inside the context window of the prompt to LLM

## Ways to improve RAG:
### 1. Use cleaner data
Its always better to have clean data, but during this step, we often feed-in PDF files or documents that when parsed, can break the document content. If possible, work on cleaning data manually. 
### 2. Chunk size 
Lesser chunk size probably generates the chunks with incomplete contexts and hence causing information loss.
Larger chunk implies that the AI model needs to parse larger tokens, this affects performance and when the prompt size is too large, AI models tend to focus on start and end of the prompt, causing information in the middle to be ignored.
### 3. Filter chunks
If we set the vector search to return large number of chunks, it means again the AI models takes in a lot more tokens and have lot of noise, as we could be sending irrelevant chunks. 
Its better to have a model to re-rank the chunks on relevancy to filter again before sending them it into the LLM. 
You may also use different search methods and combine them for mor relevancy.

*NOTE* You can index data with some parameters (time, chapters etc) and then use the models to generate meta-data from the prompt so we can search at the right place.

### 4. Step bank prompting
User prompt string may not be the optimal way to search the vector db, for this, we can use LLM to first re-phrase the question so that it is more "researchable" and then use the re-phrased prompt string to search for chunks.
