
from uuid import uuid4
import dotenv
import os
import json
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.agents import create_agent
from langchain_core.tools import tool
from pinecone import ServerlessSpec
from pinecone import Pinecone
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage,
    trim_messages,
)

dotenv.load_dotenv()


pinecone_api_key = os.getenv("PINECONE_API_KEY")
api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=api_key
)

embedding = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    api_key=api_key,
)

pc = Pinecone(api_key=pinecone_api_key)

index_name = "my-data"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=3072,    # кількість чисел у векторі
        metric="cosine",   # формула для пошуку схожих текстів
        spec=ServerlessSpec(
            cloud="aws",        # хмарна платформа(амазон)
            region="us-east-1"  # регіон
        ),
    )

index = pc.Index(index_name)

vector_store = PineconeVectorStore(
    index=index,          # база даних
    embedding=embedding   # модель для кодування
)

with open("/Users/macbook/ITStep-AI/data/lesson8/lesson_rag/files/future_of_ai.txt", "r", encoding="utf-8") as f:
    text1 = f.read()
    doc1 = Document(
        page_content=text1,)
with open("/Users/macbook/ITStep-AI/data/lesson8/lesson_rag/files/intro.txt", "r", encoding="utf-8") as f:
    text2 = f.read()
    doc2 = Document(
        page_content=text2, )

with open("/Users/macbook/ITStep-AI/data/lesson8/lesson_rag/files/machine_learning.txt", "r", encoding="utf-8") as f:
    text3 = f.read()
    doc3 = Document(
        page_content=text3, )

with open("/Users/macbook/ITStep-AI/data/lesson8/lesson_rag/files/neural_networks.txt", "r", encoding="utf-8") as f:
    text4 = f.read()
    doc4 = Document(
        page_content=text4, )


documents = [doc1, doc2, doc3, doc4]
uuids = [str(uuid4()) for _ in range(len(documents))]

# # добавити документи в базу даних
vector_store.add_documents(
    documents=documents,
    ids=uuids
)

@tool
def document_search(query:str):
    """
    Пошук документів у векторній базі даних
    :param query: str -- запит користувача
    :return: документи
    """
    result = vector_store.similarity_search(query, k=1)
    return result

agent = create_agent(model=llm, tools=[document_search])

messages = [
    SystemMessage("""
    Ти -- ввічлиіий чат бот

    ###ІНСТРУКЦІЯ###
    1.
    2. якщо користувач питає щось про штучний інтелект шукай в document_search
    """)
]

while True:
    user_query = input("Ви: ")

    if user_query == "":
        break

    user_message = HumanMessage(user_query)

    messages.append(user_message)

    data = {
        "messages": messages
    }

    data = agent.invoke(data)

    messages = data["messages"]

    response = messages[-1]

    print(response.text)

    print()
    print("----------ІСТОРІЯ-----------")

    for message in messages:
        print(repr(message))

    print("-----------------------------")
    print()
