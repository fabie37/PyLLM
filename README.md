# PyLLM - Local LLM Chat API

A FastAPI-based microservice for running local Large Language Models (LLMs) with conversation history management.

## Features

- 🚀 **FastAPI REST API** - Modern async API with automatic documentation
- 💬 **Multi-chat support** - Manage multiple conversation threads by chat_id
- 🧠 **Mistral-7B** - Uses Mistral-7B-Instruct-v0.2 (configurable)
- 🎯 **8-bit quantization** - Runs on consumer GPUs (12GB VRAM)
- 📝 **Conversation history** - Persistent chat history per session

## Project Structure

```
LLM/
├── src/
│   ├── Core/
│   │   ├── Engine.py      # Stateless LLM engine
│   │   ├── PyLLM.py       # Stateful chat wrapper
│   │   └── __init__.py
│   ├── Server/
│   │   └── main.py        # FastAPI application
│   └── prompt.py          # CLI chat interface
├── tests/
│   ├── test_basic.py
│   └── test_pyllm.py
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support (12GB+ VRAM recommended)
- Windows/Linux/Mac

## Installation

1. **Clone the repository**
```bash
git clone git@github.com:fabie37/PyLLM.git
cd PyLLM
```

2. **Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

### Start the API Server

```bash
 fastapi dev src/Server/main.py 
```

Access the API documentation at: http://localhost:8000/docs

### API Endpoints

#### Get Chat History
```bash
GET /chat/{chat_id}
```

#### Send Message
```bash
POST /chat/{chat_id}
Body: {"message": "Hello!"}
```

#### Clear History
```bash
DELETE /chat/{chat_id}
```

### Example Usage

```python
import requests

# Send a message
response = requests.post(
    'http://localhost:8000/chat/user123',
    json={'message': 'Hello! How are you?'}
)
print(response.json()['response'])

# Get chat history
history = requests.get('http://localhost:8000/chat/user123')
print(history.json())

# Clear history
requests.delete('http://localhost:8000/chat/user123')
```

## Development

### Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src tests/

# Verbose
pytest -v
```

### Code Structure

- **Engine** - Stateless LLM wrapper, handles model loading and inference
- **PyLLM** - Stateful chat manager, handles conversation history
- **Server** - FastAPI application with REST endpoints

## Configuration

### Change Model

Edit the model name in [`src/Core/PyLLM.py`](src/Core/PyLLM.py):

```python
def __init__(self, model_name: str = "your-hugging-face-model-here"):
```

### Adjust Performance

In [`src/Core/Engine.py`](src/Core/Engine.py):

```python
def start(self, 
          load_in_8bit: bool = True,  # 8-bit quantization
          torch_dtype = torch.float16,  # Half precision
          device_map: str = "auto"):  # Auto GPU mapping
```

## Performance

On **RTX 3060 12GB**:
- First token: ~1-3 seconds
- Generation speed: ~15-30 tokens/second
- Memory usage: ~7-8GB VRAM (8-bit mode)

## Troubleshooting

### CUDA Out of Memory
- Reduce `max_length` in generation parameters
- Use 4-bit quantization: `load_in_4bit=True`
- Clear GPU cache: `torch.cuda.empty_cache()`

### Import Errors
- Run from project root: `c:\Users\Wintermute\Documents\Programming\LLM\`
- Ensure virtual environment is activated

### Slow Generation
- Reduce `max_new_tokens` (default: 1000 → 150)
- Remove repetition penalties
- Enable `use_cache=True`

## License

MIT

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## Acknowledgments

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Mistral AI](https://mistral.ai/)