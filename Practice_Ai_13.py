import dotenv
import os

import langchain
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


# завантадити дані з .env
dotenv.load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")


# # модель
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=api_key
)


# class BooksName(BaseModel):
#     genre: str = Field(description="визначає жанр книги")
#
#
#
# parser = PydanticOutputParser(pydantic_object=BooksName)
#
# instrustions = parser.get_format_instructions()
#
# prompt1 = PromptTemplate.from_template("""
#     Ти — професійний літературний експерт та бібліограф
#     Твоя задача — визначити точний жанр заданої книги.
#
#     ###ІНСТРУНКЦІЇ###
#     Відпиши лише назвою жанру одним-двома словами, без зайвих речень.
#
#     ###ФОРМАТ ВІДПОВІДІ###
#     {format_instrustions}
#
#     ###ВХІДНІ ДАНІ###
#     Назва книги: {book_name}
# """,
#     partial_variables={"format_instrustions": instrustions}
#
#                                        )
#
# chain1 = prompt1 | llm | parser
#
#
# class Recommendations(BaseModel):
#     recommendations_same: list[str] = Field(description="список цікавих книг за визначеним жанром")
#     recommendations_other: list[str] = Field(description="список цікавих схожих книг іншого жанру")
#
# parser = PydanticOutputParser(pydantic_object=Recommendations)
#
# instrustions = parser.get_format_instructions()
#
#
# prompt = PromptTemplate.from_template("""
#     Ти — досвідчений книжковий рекомендатель.
#     Твоя задача — підібрати список із 4-5 схожих книг (як у тому ж жанрі, так і суміжних),
#     базуючись на назві книги та її жанрі.
#
#
#     ###ФОРМАТ ВІДПОВІДІ###
#     {format_instructions}
#
#     ###ВХІДНІ ДАНІ###
#     Назва книги: {book_name} | Жанр: {genre}
# """,
#     partial_variables={"format_instructions": instrustions}
#                                       )
#
#
# chain2 = prompt | llm | parser
#
#
# user_book = "Гаррі Поттер"
#
# data = {
#     "book_name": user_book,
# }
#
#
# response1 = chain1.invoke(data)
#
# print(f"Відповідь на питання: {response1.genre}")
#
# data = {
#     "book_name": user_book,
#     "genre": response1.genre
# }
#
# response2 = chain2.invoke(data)
# print(f"Рекомендації: {response2}")
#


class Skills(BaseModel):
    experience: float = Field(description="Досвід роботи")
    english_level: str = Field(description="Рівень англійської")
    frameworks: list[str] = Field(description="список бібліотек")
    technologies: list[str] = Field(description="Список технологій")
    language_programing: str = Field(description="Мова програмування")


parser = PydanticOutputParser(pydantic_object=Skills)

instrustions = parser.get_format_instruments()


prompt = PromptTemplate.from_template(f"""
    Ти - досвідчений рекрутер 
    Потрібно повернути основні навички з опису вакансії
    
    ###ФОРМАТ ВІДПОВІДІ###
    {instrustions}
    
    ###ВХІДНІ ДАНІ###
    Опис вакансії: {vacancy_description}
""")

chain1 = prompt | llm | parser

vacancy = """
Are you a Data Scientist with a love of LLMs, generative AI?
 
We are looking for a passionate Data Scientist to implement AI solutions aimed at achieving business goals.
 
This role offers the opportunity to work on cutting-edge AI adoption projects that helps to improve current business processes.
 
     You'll be a great fit if you have:
 Strong Python Experience (2 year +);
Experience with LLM , Diffusion models;
Knowledge of Prompt engineering;
Experience with Gen AI-related technologies such as LangChain and RAG;
Experience with Neural Networks (Optional) ;
Experience with NLP , Predictive analytics and Machine learning;
Experience with Pandas;
Experience with SQL, including experience with large datasets;
Strong experience in statistics;
Bachelor's degree in Computer Science or a related field.
What you'll do:
Develop AI agents that utilize LLM, RAG and langchain approach;
Implement LLM and Diffusion models to boost business productivity;
Utilize LLM (LLM Vision) to improve object detection, text classification and extraction;
Create forecasting, recommendation, and classification models;
Transform business challenges to AI applications.

     We ensure your growth with:
Competitive salary fixed in USD;
Flexible working schedule and fully remote work format;
Paid vacation days and sick leave days ;
Personal and professional development opportunities;
Participation in building innovative projects from scratch using modern technologies;
Team-building activities and corporate events;
English classes and educational events.
 
"""

data = {
    "vacancy_description": vacancy,
}
response1 = chain1.invoke(data)
print(response1)