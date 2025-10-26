import requests
import time
import sys
import numpy as np

class RealWorldInterface:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def get_real_digital_metrics(self, complexity_factor: float) -> np.ndarray:
        # 1. Latenz einer echten API messen
        headers = {"Authorization": f"Bearer {self.api_key}"}
        start = time.time()
        try:
            response = requests.get("https://api.openai.com/v1/models", headers=headers)
            latency = time.time() - start
            error_rate = 0 if response.ok else 1
        except Exception:
            latency = 1.0
            error_rate = 1

        # 2. Token-Kosten (hier: Antwortlänge als Proxy)
        try:
            token_cost = len(response.text)
        except Exception:
            token_cost = 0

        # 3. Baue den Vektor für das restliche System
        return np.array([latency * 10, token_cost / 5000, error_rate * 50])