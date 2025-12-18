
from src.Core.PyLLM import PyLLM


def test_generate_response():
    llm = PyLLM()
    prompt = "Hello, how are you?"
    response = llm.generate_response(prompt)
    assert isinstance(response, str)

def test_clear_history():
    llm = PyLLM()
    response = llm.chat("Hello")
    assert len(llm.get_history()) == 2
    llm.clear_history()
    assert len(llm.get_history()) == 0

def test_get_history():
    llm = PyLLM()
    llm.chat("Hello")
    history = llm.get_history()
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Hello"
    assert history[1]["role"] == "assistant"

def test_load_history():
    llm = PyLLM()
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello! How can I help you?"}
    ]
    llm.load_history(history)
    loaded_history = llm.get_history()
    assert loaded_history == history