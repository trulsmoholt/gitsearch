import openai
import numpy as np
from typing import List, Dict

def get_embedding(text: str, api_key: str) -> List[float]:
    """Get embedding for a text using OpenAI's API."""
    client = openai.OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_relevant_commits(commits: List[Dict], query: str, api_key: str, top_k: int = 5) -> List[Dict]:
    """Find the most relevant commits using embeddings."""
    # Get query embedding
    query_embedding = get_embedding(query, api_key)
    
    # Get embeddings for each commit (combining message and diff)
    commit_scores = []
    for commit in commits:
        commit_text = f"{commit['message']} {commit['diff']}"
        commit_embedding = get_embedding(commit_text, api_key)
        similarity = cosine_similarity(query_embedding, commit_embedding)
        commit_scores.append((commit, similarity))
    
    # Sort by similarity and return top k
    commit_scores.sort(key=lambda x: x[1], reverse=True)
    return [commit for commit, _ in commit_scores[:top_k]] 