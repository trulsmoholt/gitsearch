import os
import logging
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from .cache import EmbeddingCache

# Configure logging to not interfere with tqdm
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)
# Disable httpx logging
logging.getLogger('httpx').setLevel(logging.WARNING)

# Initialize cache
cache = EmbeddingCache()

def get_embedding(text: str, client: OpenAI) -> list[float]:
    """Get embedding for a single text using OpenAI's API."""
    # Try to get from cache first
    cached = cache.get(text)
    if cached is not None:
        return cached

    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
            encoding_format="float"
        )
        embedding = response.data[0].embedding
        # Store in cache
        cache.set(text, embedding)
        return embedding
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        return None

def get_embeddings(texts: list[str], client: OpenAI) -> list[list[float]]:
    """Get embeddings for multiple texts with progress bar."""
    embeddings = []
    for text in tqdm(texts, desc="Generating embeddings", unit="commit"):
        embedding = get_embedding(text, client)
        if embedding:
            embeddings.append(embedding)
        else:
            # If embedding fails, add a zero vector of the same dimension
            embeddings.append([0.0] * 1536)  # text-embedding-3-small dimension
    return embeddings

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_relevant_commits(commits: list[dict], query: str, client: OpenAI, top_k: int = 5) -> list[dict]:
    """Find the most relevant commits for a given query using embeddings."""
    # Get query embedding
    query_embedding = get_embedding(query, client)
    if not query_embedding:
        logger.error("Failed to get query embedding")
        return []

    # Get commit embeddings with progress bar
    commit_texts = [f"{commit['message']}\n{commit['diff']}" for commit in commits]
    commit_embeddings = get_embeddings(commit_texts, client)

    # Calculate similarities
    similarities = []
    for i, commit_embedding in enumerate(commit_embeddings):
        similarity = cosine_similarity(query_embedding, commit_embedding)
        similarities.append((similarity, commits[i]))

    # Sort by similarity score (first element of tuple) and return top k
    similarities.sort(key=lambda x: x[0], reverse=True)
    return [commit for _, commit in similarities[:top_k]] 