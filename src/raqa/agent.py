from openai import OpenAI
from utils import get_openai_key

class RAGAgent:
    def __init__(self):
        self.db = VectorDB()
        self.db.load()

        api_key = get_openai_key()
        self.client = OpenAI(api_key=api_key)

    def retrieve(self, query: str):
        results = self.db.search(query)
        filtered = self.db.nucleus_filter(results)

        context = "\n\n".join(
            f"[{r['data']['source']}]\n{r['data']['text']}"
            for r in filtered
        )

        return context

    def chat(self):
        print("💬 RAG Agent ready. Type 'exit' to quit.")

        messages = [{"role": "system", "content": "You are a helpful assistant."}]

        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ("exit", "quit"):
                break

            context = self.retrieve(user_input)

            augmented_prompt = f"""
Use the context below if relevant:

{context}

User question:
{user_input}
"""

            messages.append({"role": "user", "content": augmented_prompt})

            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages
            )

            reply = response.choices[0].message.content
            messages.append({"role": "assistant", "content": reply})

            print("\nAssistant:", reply)