==============================================
SIMPLE RAG PROFILE SEARCH (Windows Instructions)
==============================================

1️⃣ Install dependencies
------------------------
Open Command Prompt inside this folder and run:

    pip install -r requirements.txt

(If FAISS fails to install, use:
    pip install faiss-cpu==1.7.4
)

2️⃣ Run the FastAPI server
--------------------------
    uvicorn app:app --reload --port 8000

3️⃣ Open the front-end
----------------------
Open 'index.html' in your browser.

4️⃣ Try it out!
---------------
Type something like:
    "Python developer with ML experience"
and press 'Search'.

Results will appear instantly below.

==============================================
This app demonstrates a minimal Retrieval-Augmented
Generation (RAG) search using:
- SentenceTransformers (embeddings)
- FAISS (vector similarity search)
- FastAPI (backend)
==============================================
