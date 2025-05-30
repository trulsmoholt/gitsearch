# LLM Git Commit Search Prototype

import openai
import git
import os
from datetime import datetime, timedelta

# --- Config ---
REPO_PATH = "./your-repo"  # Path to local git repo
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Or hardcode if testing
MODEL = "gpt-4"
DAYS_BACK = 30

# --- Collect recent commits ---
repo = git.Repo(REPO_PATH)
cutoff_date = datetime.now() - timedelta(days=DAYS_BACK)

commits = []
for commit in repo.iter_commits(since=cutoff_date.isoformat()):
    commits.append({
        "hash": commit.hexsha[:7],
        "author": commit.author.name,
        "date": datetime.fromtimestamp(commit.committed_date).isoformat(),
        "message": commit.message.strip(),
        "diff": commit.diff(None, create_patch=True).decode("utf-8", errors="ignore")[:1000]  # Truncate large diffs
    })

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
def query_llm(prompt):
    openai.api_key = OPENAI_API_KEY
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a senior software engineer helping with code archaeology."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.3
    )
    return response["choices"][0]["message"]["content"]

# --- Example run ---
if __name__ == "__main__":
    user_query = input("Hva slags bug/feature undersøker du? ")
    prompt = build_prompt(commits, user_query)
    print("\n\n--- Spør LLM ---\n")
    response = query_llm(prompt)
    print(response)