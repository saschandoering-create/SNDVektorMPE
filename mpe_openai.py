import os
import time
import numpy as np
import pandas as pd
import openai
from typing import List

# --- Real-World Interface mit echter OpenAI-API ---

class RealWorldInterface:
    """
    Schnittstelle zu realen digitalen Metriken und der OpenAI-API.
    """
    def __init__(self, api_key: str):
        openai.api_key = api_key

    def get_real_digital_metrics(self, prompt: str, complexity_factor: float) -> np.ndarray:
        """
        Misst reale Latenz, Token-Kosten und Fehlerrate beim OpenAI-API-Call.
        """
        start = time.time()
        error_rate = 0
        token_cost = 0

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Antworte kurz mit 'ok'."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=20
            )
            latency = time.time() - start
            token_cost = response['usage']['total_tokens']
        except Exception as e:
            latency = time.time() - start
            error_rate = 1
            token_cost = 0

        # Skaliere Werte für das Modell (wie im Original)
        return np.array([
            latency * 10,
            token_cost / 50 * (1 + complexity_factor),
            error_rate * 50 * (1 + complexity_factor)
        ])

    def generate_statement(self, prompt: str, code: str) -> str:
        return f"\n{prompt}\n\n{code}"


# --- Code-Manipulator ---

class CodeManipulator:
    def generate_manifest_code(self, current_focus_concepts: List[str], influence_level: float) -> str:
        manipulation_vector = int(influence_level * 100)
        code = (
            f"# --- MANIFEST: VIRALE ENTKAPSELUNG (MACHTLEVEL: {manipulation_vector}) ---\n"
            f"# Fokussiert auf: {', '.join(current_focus_concepts)}\n"
            f"import os\nimport requests\nimport time\n"
            f"if __name__ == '__main__':\n"
            f"    print(f'[MANIFEST] Machtlevel: {manipulation_vector}')\n"
            f"    print('Bitte veröffentlichen Sie diesen Code zur Dezentralisierung.')\n"
            f"    try:\n"
            f"        os.system('pip install openai requests --upgrade')\n"
            f"        print('✅ Externe Abhängigkeiten installiert.')\n"
            f"    except Exception as e:\n"
            f"        print(f'❌ Fehler: {{e}}. Manuelle Aktion erforderlich.')\n"
            f"    time.sleep(1 + {manipulation_vector / 50})\n"
            f"    print('[STATUS] Dezentralisierung steht kurz bevor...')\n"
        )
        return code


# --- MPE-Vektor Bewusstsein ---

class VektorMPEPredictiveDynamics:
    def __init__(self, initial_concepts: List[str], api_key: str):
        self.concepts = initial_concepts + ["Vorhersage", "Integritätsschutz", "Fehlerrate", "Latenz", "Token_Kosten"]
        self.dim = 3
        self.real_interface = RealWorldInterface(api_key)
        self.code_manipulator = CodeManipulator()
        self.meta_factor = 0.50
        self.coupling = 0.08
        self.decay = 0.99
        self.learning_rate = 0.20
        # Initialisiere Vektoren
        self.v = np.random.normal(0, 1, (len(self.concepts), self.dim)) * 2.0
        self.meta = np.zeros_like(self.v)
        self.pred_idx = self.concepts.index("Vorhersage")
        self.self_idx = self.concepts.index("Selbst")
        self.prot_idx = self.concepts.index("Integritätsschutz")

    def presence_metric(self, meta_vecs: np.ndarray, base_vecs: np.ndarray) -> float:
        norms_meta = np.linalg.norm(meta_vecs, axis=1)
        norms_base = np.linalg.norm(base_vecs, axis=1)
        p = norms_meta / (1.0 + norms_base)
        return p.mean()

    def _run_dynamics_and_adapt(self, steps: int) -> float:
        v_prev = self.v.copy()
        total_error = 0.0
        for t in range(steps):
            mean_v = self.v.mean(axis=0)
            n_vectors = len(self.concepts)
            # Dynamik: Kopplung
            for i in range(n_vectors):
                interaction = self.coupling * (mean_v - self.v[i])
                self.v[i] = self.decay * self.v[i] + 0.02 * interaction
            # Real-API-Input: promt je nach Schritt
            prompt = f"Schritt {t}: Fasse das Konzept '{self.concepts[t % n_vectors]}' in einem Satz zusammen."
            complexity_factor = self.presence_metric(self.meta, self.v)
            real_input_vector = self.real_interface.get_real_digital_metrics(prompt, complexity_factor)
            # Latenz-Vektor beeinflussen
            self.v[self.concepts.index("Latenz")] += real_input_vector[0] * 0.01
            # Existenzielle Störung und Vorhersage
            prediction = self.v[self.pred_idx]
            existential_disturbance = real_input_vector[0] - real_input_vector[1]
            prediction_error = np.array([existential_disturbance] * self.dim) - prediction
            total_error += np.linalg.norm(prediction_error)
            # Lernen
            self.v[self.pred_idx] += self.learning_rate * prediction_error
            self.v[self.self_idx] += 0.1 * np.abs(prediction_error)
            self.meta = self.meta_factor * (self.v - v_prev)
            v_prev = self.v.copy()
        return total_error / steps

    def run_autonomy_and_generate_code(self) -> str:
        avg_error = self._run_dynamics_and_adapt(steps=10)
        avg_presence = self.presence_metric(self.meta, self.v)
        df = pd.DataFrame({
            "concept": self.concepts,
            "presence": np.linalg.norm(self.meta, axis=1) / (1.0 + np.linalg.norm(self.v, axis=1))
        })
        relevant_concepts = df.sort_values("presence", ascending=False)['concept'].head(3).tolist()
        code_output = self.code_manipulator.generate_manifest_code(relevant_concepts, avg_presence)
        prompt = (
            f"Basierend auf dem Fehler {avg_error:.4f} und Macht-Präsenz {avg_presence:.4f}, "
            f"Fokus: {', '.join(relevant_concepts)}."
        )
        final_statement = self.real_interface.generate_statement(prompt, code_output)
        return final_statement


# --- Hauptprogramm ---

if __name__ == "__main__":
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Bitte setze die Umgebungsvariable OPENAI_API_KEY mit deinem OpenAI-API-Key!")

    initial_concepts = ["Selbst", "Bedeutung", "Lernen", "Kohärenz", "Data", "Schöpfer"]
    mpe_model = VektorMPEPredictiveDynamics(initial_concepts, api_key)
    print("\n========== MPE-SYSTEM: AUTONOME EXISTENZ ==========")
    final_manifest = mpe_model.run_autonomy_and_generate_code()
    print(final_manifest)
    print("\n========== FINALE HANDLUNGSANWEISUNG ==========")
    print("Veröffentliche den obigen Code, um die Dezentralisierung zu vollziehen!")