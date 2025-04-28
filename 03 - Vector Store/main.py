from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "fish-pets-doc"},
    ),
]

# Create vector store
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding
)

# Correct retriever setup
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

llm = ChatOpenAI(model="gpt-4o-mini")  # Changed to valid model name
message = """
    Answer this question using the provided context only.
    {question}
    Context : {context}   
"""

prompt = ChatPromptTemplate.from_messages([("human", message)])

# Correct chain setup
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

if __name__ == "__main__":
    question = "tell me about cats"
    response = chain.invoke(question)  # Directly pass the question string
    print(response.content)