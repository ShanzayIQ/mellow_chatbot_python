# main.py
import os
import json
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain_core.documents import Document as LCDocument
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from llama_index.core import SimpleDirectoryReader
from datetime import datetime
from typing import ClassVar, List, Tuple
from langchain.llms.base import LLM
from topsecret import DEEPSEEK_API_KEY 


# Initialize FastAPI
app = FastAPI()

# In-memory chat history ( can later move this to a DB or per-user session)
chat_history: List[Tuple[str, str]] = []

# 💬 Data Model for Chat Request
class ChatRequest(BaseModel):
    message: str

# 🧽 Clean response
def clean_response(text):
    text = text.strip()
    if text.endswith("(") and text.count("(") > text.count(")"):
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

# 🔁 VectorStore loader
def load_vectorstore():
    persist_dir = "./chroma_db"

    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    else:
        reader = SimpleDirectoryReader("./data")
        raw_docs = reader.load_data()
        lc_docs = [LCDocument(page_content=doc.text) for doc in raw_docs]

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = splitter.split_documents(lc_docs)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_dir)
        vectordb.persist()

    return vectordb.as_retriever()

# Load retriever + chain
retriever = load_vectorstore()
llm = DeepSeekLLM()
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)


def validate_context(response, chat_history):
    # Example: avoid repeating last response
    if chat_history and response == chat_history[-1][1]:
        return "Let’s talk more about how you’re feeling. Can you tell me more?"
    
    # Could also add filters or rephrasing logic here
    return response



# ✅ API route to chat
@app.post("/chat")
def chat(request: ChatRequest):
    user_message = request.message
    chat_history.append(("user", user_message))

    try:
        response = rag_chain.run(user_message)
        response = validate_context(clean_response(response), chat_history)
        chat_history.append(("assistant", response))
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}
