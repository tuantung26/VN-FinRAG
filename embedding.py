import json
import requests
import numpy as np


class Jina:
    def __init__(self):
        self.url = "https://api.jina.ai/v1/embeddings"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer jina_b37191e39117417898aa38e8d43cbbf2sADPlFKvIulBYHXEk0ci1I9aAvPz",
        }

    def EmbeddingBysentence(self, sentence):
        data = {
            "model": "jina-embeddings-v5-text-small",
            "task": "retrieval.query",
            "normalized": True,
            "input": [sentence], 
        }

        response = requests.post(self.url, headers=self.headers, json=data)
        result = response.json()

        
        embeddings = [item["embedding"] for item in result["data"]]
        return embeddings[0]


