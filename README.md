# GitSearch

A command-line tool that uses OpenAI's embeddings and GPT models to help you find relevant commits in your git repository based on natural language queries.

## Features

- Semantic search through git commit history using OpenAI embeddings
- Automatic detection of git repository
- Configurable number of relevant commits to analyze
- Natural language querying of commit history

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd gitsearch
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Install the package locally:
```bash
pip install --user .
```

## Configuration

Set your [OpenAI API key](https://platform.openai.com/docs/overview) as an environment variable:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or add it to your `~/.zshrc` or `~/.bashrc`:
```bash
echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.zshrc
source ~/.zshrc
```

## Usage

### Basic Usage

From within a git repository:
```bash
gitsearch "fix login button not working"
```

### Command Line Options

- `--repo`: Path to git repository (defaults to current directory)
- `--api-key`: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
- `--model`: OpenAI model to use (default: "gpt-4")
- `--days`: Number of days to look back (default: 30)
- `--top-k`: Number of most relevant commits to analyze (default: 5)

Examples:
```bash
# Search in specific repository
gitsearch "fix login button" --repo /path/to/repo

# Look back 60 days
gitsearch "fix login button" --days 60

# Analyze top 3 most relevant commits
gitsearch "fix login button" --top-k 3

# Use specific API key
gitsearch "fix login button" --api-key your-api-key-here
```

## Development

For development and testing, you can run the script directly:
```bash
python gitsearch.py "your query"
```

## Requirements

- Python 3.7+
- OpenAI API key
- Git repository

## Dependencies

- openai
- gitpython
- numpy
