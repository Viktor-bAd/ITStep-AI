import dotenv
import os
from langchain_google_genai import GoogleGenerativeAI


dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

llm = GoogleGenerativeAI(
    model="gemini-3.6-flash",
    api_key=api_key,
    temperature=0.1
)

# response = llm.invoke("Привіт - тільки одне слово")
# print(response)

with open("/Users/macbook/ITStep-AI/data/lesson_data/lesson9/rules.txt", "r", encoding="utf-8") as f:
    rules = f.read()


responses = []
questions = []

while True:
    question = input("Question: ")
    response = llm.invoke(f"""
    Ти консультант атракціону, відповідай на запитання клієнтів на основі правил {rules}.
    Відповідай тільки на ті положення що є у правилах, якщо цього нема у правилах відповідай що ти не знаєш відповіді
    Відповідь повинна бути короткою.
    Ось питання від користувача {question}
    """)
    questions.append(question)
    responses.append(response)

    print(f"Відповідь: {response}")
