#!/usr/bin/env python3
# LLM Git Commit Search Prototype

import openai
import git
import os
import argparse
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict

def find_git_root(path):
    """Find the git repository root directory."""
    try:
        git_repo = git.Repo(path, search_parent_directories=True)
        return git_repo.git.rev_parse("--show-toplevel")
    except git.InvalidGitRepositoryError:
        return None

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

# --- Parse command line arguments ---
def parse_args():
    parser = argparse.ArgumentParser(description='LLM Git Commit Search Tool')
    parser.add_argument('--repo', help='Path to local git repo (defaults to current directory)')
    parser.add_argument('--api-key', help='OpenAI API Key (defaults to OPENAI_API_KEY env var)')
    parser.add_argument('--model', default="gpt-4", help='OpenAI model to use')
    parser.add_argument('--days', type=int, default=30, help='Number of days to look back')
    parser.add_argument('--top-k', type=int, default=5, help='Number of most relevant commits to analyze')
    parser.add_argument('query', nargs='?', help='The bug/feature to investigate')
    args = parser.parse_args()
    
    # If no repo specified, try to find git repo in current directory
    if not args.repo:
        repo_path = find_git_root(os.getcwd())
        if not repo_path:
            parser.error("No git repository found in current directory. Please specify a repository with --repo")
        args.repo = repo_path
    
    return args

# --- Collect recent commits ---
def collect_commits(repo_path, days_back):
    repo = git.Repo(repo_path)
    cutoff_date = datetime.now() - timedelta(days=days_back)
    commits = []
    for commit in repo.iter_commits(since=cutoff_date.isoformat()):
        diff_str = ""
        for diff in commit.diff(None, create_patch=True):
            diff_str += diff.diff.decode("utf-8", errors="ignore")
        commits.append({
            "hash": commit.hexsha[:7],
            "author": commit.author.name,
            "date": datetime.fromtimestamp(commit.committed_date).isoformat(),
            "message": commit.message.strip(),
            "diff": diff_str[:1000]  # Truncate large diffs
        })
    return commits

# --- Prepare prompt ---
def build_prompt(commits, user_query):
    prompt = (
        f"A user is investigating a bug in the following feature: '{user_query}'. "
        f"Based on the commit history below, identify which commits could be relevant to this feature or bug."
        "\n\n"
    )
    for c in commits:
        prompt += f"Commit {c['hash']} by {c['author']} on {c['date']}\n"
        prompt += f"Message: {c['message']}\n"
        prompt += f"Diff (truncated):\n{c['diff']}\n\n"
    prompt += "\nRespond with a list of possibly related commits and a brief explanation for each."
    return prompt

# --- Send to LLM ---
def query_llm(prompt, api_key, model):
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a senior software engineer helping with code archaeology."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.3
    )
    return response.choices[0].message.content

# --- Main function ---
def main():
    args = parse_args()
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key must be provided either via --api-key or OPENAI_API_KEY environment variable")
    
    # Collect all commits
    all_commits = collect_commits(args.repo, args.days)
    print(f"\nFound {len(all_commits)} commits in the last {args.days} days")
    
    # Get query from args or prompt user
    user_query = args.query
    if not user_query:
        user_query = input("What bug or feature are you investigating? ")
    
    # Find most relevant commits using embeddings
    print("\nFinding most relevant commits using semantic search...")
    relevant_commits = find_relevant_commits(all_commits, user_query, api_key, args.top_k)
    print(f"Selected top {len(relevant_commits)} most relevant commits")
    
    # Build prompt with only relevant commits
    prompt = build_prompt(relevant_commits, user_query)
    print("\n--- Querying LLM ---\n")
    response = query_llm(prompt, api_key, args.model)
    print(response)

if __name__ == "__main__":
    main()