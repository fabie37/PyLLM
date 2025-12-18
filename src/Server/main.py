
from fastapi import FastAPI
from pydantic import BaseModel
from src.Core.PyLLM import PyLLM

llm = PyLLM()
app = FastAPI()

history_db = {
    # chat_id: [ {"role": "user"/"assistant", "content": str}, ... ]
}

class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    history: list

class ChatHistory(BaseModel):
    history: list

# Endpoint to get chat history
@app.get("/chat/{chat_id}", response_model=ChatHistory, tags=["chat"])
async def chat_endpoint(chat_id: str):
    return ChatHistory(history=history_db.get(chat_id, []))

# Endpoint to chat to the LLM
@app.post("/chat/{chat_id}", response_model=ChatResponse, tags=["chat"])
async def post_chat_endpoint(chat_id: str, message: ChatMessage):
    history = history_db.get(chat_id, [])
    llm.load_history(history)
    response = llm.chat(message.message)
    history_db[chat_id] = llm.get_history()
    return ChatResponse(response=response, history=history_db[chat_id])

# Endpoint to clear chat history
@app.delete("/chat/{chat_id}", response_model=dict, tags=["chat"])
async def clear_chat_history(chat_id: str):
    if chat_id in history_db:
        del history_db[chat_id]
    return {"status": "cleared"}   

