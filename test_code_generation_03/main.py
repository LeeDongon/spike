from langchain.text_splitter import Language
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI

import os
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

src_path = os.getcwd() + "/sample.java"
loader = TextLoader(src_path)
documents = loader.load()

java_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JAVA, chunk_size=1000, chunk_overlap=10
)
texts = java_splitter.split_documents(documents)
db = Chroma.from_documents(texts, OpenAIEmbeddings(disallowed_special=()))
retriever = db.as_retriever(
    search_type="mmr",  # Also test "similarity"
    search_kwargs={"k": 8},
)

llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4", temperature=1)

memory = ConversationSummaryMemory(
    llm=llm, memory_key="chat_history", return_messages=True
)

qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)
question = "Can you write a test code for those methods? also edge case"
for i in range(1, 3):
    result = qa(question)
    print(result["answer"])
