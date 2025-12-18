
from .Engine import Engine
from typing import List, Dict

class PyLLM: 

    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        self.engine = Engine(model_name=model_name)
        self.history: List[Dict[str, str]] = []

        if not self.engine.is_running:
            self.engine.start()

    def generate_response(self, prompt: str, use_history: bool = False) -> str:
        """Generate a response (optionally with history)"""
        if not self.engine.is_running:
            self.engine.start()
        
        # Build messages
        if use_history:
            messages = self.history + [{"role": "user", "content": prompt}]
        else:
            messages = [{"role": "user", "content": prompt}]
        
        response = self.engine.generate_response(messages)
        return response
    
    def chat(self, prompt: str) -> str:
        """Chat with automatic history management"""
        response = self.generate_response(prompt, use_history=True)
        self.history.append({"role": "user", "content": prompt})
        self.history.append({"role": "assistant", "content": response})
        return response

    def load_history(self, history: List[Dict[str, str]]):
        """Load conversation history"""
        self.history = history.copy()

    def get_history(self) -> List[Dict[str, str]]:
        """Get current conversation history"""
        return self.history.copy()
    
    def clear_history(self):
        """Clear conversation history"""
        self.history = []