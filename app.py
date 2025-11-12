from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import numpy as np

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load CSV dataset
df = pd.read_csv('profiles.csv')
texts = df.apply(lambda x: f"{x['name']} - {x['role']} - {x['skills']} - {x['experience']}", axis=1).tolist()

# Create embeddings
embeddings = model.encode(texts, normalize_embeddings=True)
embeddings = np.array(embeddings, dtype='float32')
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

class Query(BaseModel):
    text: str
    top_k: int = 3

@app.post('/search')
def search_profiles(query: Query):
    q_emb = model.encode([query.text], normalize_embeddings=True).astype('float32')
    scores, idxs = index.search(q_emb, query.top_k)
    results = []
    for i, score in zip(idxs[0], scores[0]):
        row = df.iloc[i].to_dict()
        row['similarity'] = round(float(score), 3)
        results.append(row)
    return {"query": query.text, "results": results}

@app.get('/')
def root():
    return {"message": "Simple RAG Profile Search API running. Use /search or open index.html."}
