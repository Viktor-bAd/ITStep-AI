import os
import dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate


dotenv.load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

llm = GoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=api_key,
    )

# prompt_template = PromptTemplate(
#     input_variables=["programming_language", "customer_task"],
#     template="""
# Твоє завдання — написати функцію для заданого опису.
#
# Ось приклади того, як виглядають очікувані рішення:
#
# Приклад 1:
# Мова: python
# Опис задачі: Знайти суму всіх елементів у списку.
# Код:
# def calculate_sum(numbers):
#     return sum(numbers)
#
# Приклад 2:
# Мова: python
# Опис задачі: Перевірити, чи є рядок паліндромом.
# Код:
# def is_palindrome(text):
#     cleaned = ''.join(e for e in text.lower() if e.isalnum())
#     return cleaned == cleaned[::-1]
#
# ---
# Тепер виріши нову задачу в такому ж стилі:
# Мова: {programming_language}
# Опис задачі: {customer_task}
#
# Напиши лише чистий код функції.
# """
# )
#
# data = {
#     "programming_language": "python",
#     "customer_task": "customer_task"
# }
#
# final_prompt = prompt_template.format(**data)
#
# response = llm.invoke(final_prompt)
# print(response)


prompt_template = PromptTemplate(
     input_variables=["Text"],
   template="""
   Ти є професійним розробником промптів та експертом
   [КОНТЕКСТ] 
   Користувач звертається із запитом створити структурований промпт для переведення тексту з неформального 
   стилю у формальний стиль за допомогою штучного інтелекту, використовуючи два підходи: Zero-shot та Few-shot.
   
   [ОБМЕЖЕННЯ] Правила:

    Зберігати початковий зміст тексту під час перекладу.

    Виправляти сленг, скорочення та розмовні конструкції на професійніші формулювання.

    Не додавати зайвої води, код чи пояснення поза межами запитуваної структури.
   
   Твоє завдання — перекладати тексти з неформального стилю у формальний (діловий) стиль.

    Ось приклади того, як слід виконувати переклад:

    Приклад 1:
    Неформальний: Привіт! Слухай, я завтра не зможу прийти на зустріч, бо в мене форс-мажор. Давай перенесемо на інший день?
    Формальний: Шановні колеги, на жаль, я не зможу бути присутнім на завтрашній зустрічі через непередбачувані обставини. 
    Пропоную перенести її на іншу дату.

    Приклад 2:
    Неформальний: Короче, ми зробили ту задачу, яку ви просили. Гляньте, чи все норм, і дайте знати.
    Формальний: Повідомляємо про успішне виконання поставленого завдання. 
    Просимо перевірити результати та надати зворотний зв'язок.

---
Тепер переведи наступний текст у формальний стиль за цим же прикладом:
Неформальний: {text}
   """
)

input_text = "Вітаю, напиши код і поясни"

data = {
    "text": input_text,
}

final_prompt = prompt_template.format(**data)

response = llm.invoke(final_prompt)
print(response)