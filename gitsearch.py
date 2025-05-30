#!/usr/bin/env python3
# LLM Git Commit Search Prototype

import openai
import git
import os
import argparse
from datetime import datetime, timedelta

def find_git_root(path):
    """Find the git repository root directory."""
    try:
        git_repo = git.Repo(path, search_parent_directories=True)
        return git_repo.git.rev_parse("--show-toplevel")
    except git.InvalidGitRepositoryError:
        return None

# --- Parse command line arguments ---
def parse_args():
    parser = argparse.ArgumentParser(description='LLM Git Commit Search Tool')
    parser.add_argument('--repo', help='Path to local git repo (defaults to current directory)')
    parser.add_argument('--api-key', help='OpenAI API Key (defaults to OPENAI_API_KEY env var)')
    parser.add_argument('--model', default="gpt-4", help='OpenAI model to use')
    parser.add_argument('--days', type=int, default=30, help='Number of days to look back')
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
    commits = collect_commits(args.repo, args.days)
    
    # Get query from args or prompt user
    user_query = args.query
    if not user_query:
        user_query = input("What bug or feature are you investigating? ")
    
    prompt = build_prompt(commits, user_query)
    print("\n\n--- Querying LLM ---\n")
    response = query_llm(prompt, args.api_key or os.getenv("OPENAI_API_KEY"), args.model)
    print(response)

if __name__ == "__main__":
    main()