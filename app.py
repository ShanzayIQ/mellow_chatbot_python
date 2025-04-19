import os
import json
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain.llms.base import LLM
from typing import ClassVar, List, Tuple
from topsecret import DEEPSEEK_API_KEY 

# Initialize FastAPI
app = FastAPI()

# In-memory chat history (can later move this to a DB or per-user session)
chat_history: List[Tuple[str, str]] = []

# 💬 Data Model for Chat Request
class ChatRequest(BaseModel):
    message: str

# 🧽 Clean response
def clean_response(text):
    text = text.strip()
    if text.endswith("(") and text.count("(") > text.count(")") :
        return text[:-1].strip()
    return text

@app.post("/")
async def root(data: dict):
    return {"message": "Received POST request!", "data": data}

# DeepSeek LLM class
class DeepSeekLLM(LLM):
    model: ClassVar[str] = "deepseek-chat"

    def _call(self, prompt, stop=None, run_manager=None):
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }

        messages = [{"role": "system", "content": (
            "You are a compassionate and thoughtful AI therapist. Try to stick closely to what the user has shared, and ask gentle clarifying questions if something seems vague or unclear. You may gently infer meaning if it helps guide the conversation in a supportive way, but avoid making bold assumptions. Keep your responses concise, emotionally supportive, and easy to understand. Use friendly tone and soft emojis if appropriate. Avoid technical language."
        )}]

        for role, content in chat_history:
            messages.append({
                "role": "user" if role == "user" else "assistant",
                "content": content
            })

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 300,
            "temperature": 0.7
        }

        response = requests.post(url, headers=headers, json=payload)

        if response.status_code != 200:
            return f"Error {response.status_code}: {response.text}"

        try:
            data = response.json()
            content = data['choices'][0]['message']['content'].strip()
            if "Note:" in content or "As an AI" in content:
                content = content.split("Note:")[0].split("As an AI")[0].strip()
            return clean_response(content)
        except Exception as e:
            return f"Error parsing response: {str(e)}\nRaw response: {response.text}"

    @property
    def _identifying_params(self):
        return {"model": self.model}

    @property
    def _llm_type(self):
        return "deepseek-custom"

# 🔁 Remove VectorStore loader and related retrieval logic

# ✅ API route to chat
@app.post("/chat")
def chat(request: ChatRequest):
    user_message = request.message
    chat_history.append(("user", user_message))

    try:
        # Since we removed the vector store, directly interact with the LLM
        response = DeepSeekLLM()._call(user_message)
        response = clean_response(response)
        response = validate_context(response, chat_history)
        chat_history.append(("assistant", response))
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

def validate_context(response, chat_history):
    # Example: avoid repeating last response
    if chat_history and response == chat_history[-1][1]:
        return "Let’s talk more about how you’re feeling. Can you tell me more?"
    
    # Could also add filters or rephrasing logic here
    return response
