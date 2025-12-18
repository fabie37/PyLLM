import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name):
    """Load model and tokenizer with GPU support"""
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with 8-bit quantization to fit in 12GB VRAM
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Automatically uses GPU
        torch_dtype=torch.float16,  # Use half precision
        load_in_8bit=True  # 8-bit quantization for memory efficiency
    )
    
    return model, tokenizer

def generate_response(model, tokenizer, messages, max_length=1000, temperature=0.7):
    """Generate response from messages using chat template"""
    
    # Use the model's chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()

def chat_loop(model, tokenizer):
    """Interactive chat loop"""
    print("\n" + "="*50)
    print("Chat started! Type 'quit', 'exit', or 'q' to end.")
    print("="*50 + "\n")
    
    messages = []
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        # Check for exit commands
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        # Add user message
        messages.append({"role": "user", "content": user_input})
        
        print("\nAssistant: ", end="", flush=True)
        
        # Generate response
        assistant_response = generate_response(model, tokenizer, messages, max_length=150)
        print(assistant_response + "\n")
        
        # Add assistant response to history
        messages.append({"role": "assistant", "content": assistant_response})

if __name__ == "__main__":
    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
    
    # Load model
    model, tokenizer = load_model(MODEL_NAME)
    
    # Start chat loop
    chat_loop(model, tokenizer)