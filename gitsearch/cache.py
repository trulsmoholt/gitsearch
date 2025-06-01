import os
import json
import hashlib
from pathlib import Path
from typing import Optional, List

class EmbeddingCache:
    def __init__(self, cache_dir: str = ".gitsearch_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a unique key for the text."""
        return hashlib.sha256(text.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get the cache file path for a key."""
        return self.cache_dir / f"{key}.json"
    
    def get(self, text: str) -> Optional[List[float]]:
        """Get embedding from cache if it exists."""
        key = self._get_cache_key(text)
        cache_path = self._get_cache_path(key)
        
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception:
                return None
        return None
    
    def set(self, text: str, embedding: List[float]) -> None:
        """Store embedding in cache."""
        key = self._get_cache_key(text)
        cache_path = self._get_cache_path(key)
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(embedding, f)
        except Exception:
            pass  # Silently fail if we can't write to cache 