import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List, Dict

class Engine:
    """Stateless LLM Engine for generating responses"""
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        """Initialize the engine with a model name"""
        self.model_name = model_name
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.is_running = False
        # Remove history - Engine is stateless!
        
    def start(self, 
              load_in_8bit: bool = True,
              torch_dtype = torch.float16,
              device_map: str = "auto") -> None:
        """Load and start the model"""
        if self.is_running:
            print(f"Engine already running with model: {self.model_name}")
            return
            
        print(f"Starting engine with model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            load_in_8bit=load_in_8bit
        )
        self.is_running = True
        print("Engine started successfully!")
        
    def stop(self) -> None:
        """Stop the engine and free up memory"""
        if not self.is_running:
            print("Engine is not running")
            return
            
        print("Stopping engine...")
        del self.model
        del self.tokenizer
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.model = None
        self.tokenizer = None
        self.is_running = False
        print("Engine stopped successfully!")
        
    def generate_response(self, 
                         messages: List[Dict[str, str]],
                         max_length: int = 1000,
                         temperature: float = 0.7) -> str:
        """
        Generate a response given a message history
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response string
        """
        if not self.is_running:
            raise RuntimeError("Engine is not running. Call start() first.")
            
        # Format messages using chat template
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def is_ready(self) -> bool:
        """Check if engine is ready to use"""
        return self.is_running and self.model is not None and self.tokenizer is not None
    
    def get_status(self) -> Dict[str, any]:
        """Get engine status information"""
        return {
            "running": self.is_running,
            "model": self.model_name,
            "cuda_available": torch.cuda.is_available(),
            "cuda_memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        }