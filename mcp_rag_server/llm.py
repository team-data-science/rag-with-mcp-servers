import requests
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the model once at module level
_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_embedding(text: str) -> list:
    """Generate 384-dim embeddings using all-MiniLM-L6-v2."""
    return _model.encode(text).tolist() 