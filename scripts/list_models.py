"""Quick script to list all models available on your Groq account."""
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
models = client.models.list()

for m in sorted(models.data, key=lambda x: x.id):
    print(m.id)
