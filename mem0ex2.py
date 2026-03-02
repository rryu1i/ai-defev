import os

from dotenv import load_dotenv
from mem0 import Memory
from openai import OpenAI

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

config = {
    "llm": {
        "provider": "openai",
        "config": {
            "openai_base_url": "https://api.groq.com/openai/v1",
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "temperature": 0,
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "memory",
            "url": QDRANT_URL,
            "api_key": QDRANT_API_KEY,
            "embedding_model_dims": 384,
        },
    },
    "embedder": {
        "provider": "fastembed",
        "config": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
        },
    },
}

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY não encontrada no ambiente.")

client = OpenAI(api_key=groq_api_key, base_url="https://api.groq.com/openai/v1")

memory = Memory.from_config(config)


def chat_with_memories(message: str, user_id: str = "daniel") -> str:
    relevant_memories = memory.search(query=message, user_id=user_id, limit=3)
    memories_str = "\n".join(
        f"- {entry['memory']}" for entry in relevant_memories["results"]
    )

    input_prompt = f""" Você é um assistente pessoal.
    Responda à pergunta considerando as memórias do usuário.
    Memórias do usuário: {memories_str}
    Pergunta: {message}"""

    response = client.responses.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        input=input_prompt,
    )

    assistant_response = response.output_text

    messages = [
        {"role": "user", "content": message},
        {"role": "assistant", "content": assistant_response},
    ]

    memory.add(messages, user_id=user_id)

    return assistant_response


def main():
    print("Chat com IA (digite 'sair'para encerrar)")
    while True:
        user_input = input("Você: ").strip()
        if user_input.lower() == "sair":
            print("Adeus!")
            break
        print(f"IA: {chat_with_memories(user_input)}")


if __name__ == "__main__":
    main()
