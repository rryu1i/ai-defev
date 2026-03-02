from mem0 import MemoryClient
from dotenv import load_dotenv

load_dotenv()

client = MemoryClient()

client.add("Sou o Roger!", user_id="roger")

query = "Qual o meu nome?"
response = client.search(query, filters={"user_id": "roger"})
response
response["results"][0]["memory"]
