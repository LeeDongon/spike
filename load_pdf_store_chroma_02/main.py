from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import chroma
from langchain.retrievers import MultiQueryRetriever
from langchain_openai import ChatOpenAI


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


loader = PyPDFLoader("./lucky_day.pdf")
pages = loader.load_and_split()

splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=10)
docs = splitter.split_documents(pages)

embedding = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)

db = chroma.Chroma.from_documents(docs, embedding=embedding)

llm = ChatOpenAI(openai_api_key = openai_api_key, temperature=0)

retriever_from_llm = MultiQueryRetriever.from_llm(retriever=db.as_retriever(), llm=llm)

query = "아내가 먹고 싶어하는 음식은?"

docs = retriever_from_llm.get_relevant_documents(query=query)

print(docs)
print(len(docs))
