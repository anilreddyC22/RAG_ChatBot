from importlib.metadata import version
import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()


# Ensure the OPENAI_API_KEY environment variable is set
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise EnvironmentError("The OPENAI_API_KEY environment variable is not set.")


os.environ["LANGCHAIN_TRACING"] = "true"
#os.environ["LANGCHAIN_API_KEY"] = os.environ["OPENAI_API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = "langchain-Conversational_RAG"

# FROM LANGCHAIN import OpenAI
from langchain_openai import ChatOpenAI

llm=ChatOpenAI(model="gpt-4o-mini",openai_api_key=openai_api_key, temperature=0.5, max_tokens=1000)
#llm_response=llm.invoke("Tell me a Funny Joke")
#print(llm_response.content)



#definiation of the OutputParser : 
# OutputParser definition:
# OutputParsers in LangChain are used to convert the raw output from language models
# into a desired format (e.g., string, list, dictionary). StrOutputParser is a simple
# parser that extracts and returns the output as a plain string.
from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()
#output_parser.invoke(llm_response)
#print(output_parser)


#combine llm and output_parser
# Chain definition:
# In LangChain, a "chain" is a sequence of components (such as LLMs and output parsers)
# that process data step by step. Here, we combine the language model (llm) and the
# output parser so that the output from the LLM is automatically parsed into a plain string.
chain = llm | output_parser
# Invoke the chain with a prompt
#response = chain.invoke("Tell me a Funny Joke")
#print(response)



# ChatPromptTemplate definition:
# In LangChain, a ChatPromptTemplate is used to create structured prompts for chat-based interactions
# It allows you to define how the input should be formatted, including system messages, user messages, and any additional context.
#  Here, we define a template that includes a system message and a user message.

from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("Tell me a short joke about {topic}")
#prompt.invoke({"topic": "programming"})

#chain = prompt | llm | output_parser
# Invoke the chain with a specific topic
#response1 = chain.invoke({"topic": "programming"})
#print(response1)


# we can use chatprompttemplate in many ways like passing messages through a list or tuples
from langchain_core.messages import HumanMessage,SystemMessage

system_message = SystemMessage(content="You are a helpful assistant that tells a joke.")
human_message = HumanMessage(content = "Tell me about programming.")

#llm.invoke([system_message, human_message])

# We can also use the ChatPromptTemplate to create a more complex prompt with multiple messages
template=ChatPromptTemplate([
    ("system","You are helpful assistant that tells jokes"),
    ("human", "Tell me a joke about {topic}")])
#prompt_value=template.invoke(
    #{"topic":"Parrot"})
#prompt_value
# Now we can use this prompt value with the LLM
#response2 = llm.invoke(prompt_value)
#print(response2.content)



# Take a single document and do chunks of that document by using openAIEmdedding and recrusive text splitter

#Defination of RecursiveCharacterTextSplitter:
# The RecursiveCharacterTextSplitter is a text splitter that divides a document into smaller chunks
# based on character count, allowing for overlap between chunks. This is useful for processing large texts
# in manageable pieces while retaining context across chunks.

#Defination of OpenAIEmbeddings:
# OpenAIEmbeddings is a class that generates embeddings for text using OpenAI's models.

from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from typing import List
from langchain_core.documents import Document

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=200,
    length_function=len
)

docx_loader = Docx2txtLoader("RAG_doc/product_manual.docx")
documents = docx_loader.load()
#print(f"Loaded {len(documents)} document(s)")
splits = text_splitter.split_documents(documents)
#print(f" Split the documents into {len(splits)} chunks") #o/p : Loaded 1 document(s) and  Split the documents into 3 chunks



# Function to load all the documents from a folder:

def load_documents(folder_path: str) -> List[Document]:
    documents = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            print(f"Skipping unsupported file: {filename}")
            continue

        docs = loader.load()
        documents.extend(docs)

    return documents

#  Use the folder path relative to your current script
folder_path = "RAG_doc"
docs = load_documents(folder_path)
#print(f" Loaded {len(docs)} documents from the folder.")
splits=text_splitter.split_documents(docs)
#print(f"Split the documents into {len(splits)} chunks")        #O/P : Loaded 5 documents from the folder and Split the documents into 9 chunks...


# Now we can use the OpenAIEmbeddings to create embeddings for these chunks
#embeddings = OpenAIEmbeddings()
#document_embeddings = embeddings.embed_documents([split.page_content for split in splits])
#print(f"Created embeddings for {len(document_embeddings)} documents chunks.")



#Now we can use the HuggingFaceEmbeddings to create embeddings for these chunks
from langchain_community.embeddings import HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Generate embeddings for your chunked documents
document_embeddings = embedding_model.embed_documents([split.page_content for split in splits])

print(f"Created embeddings for {len(document_embeddings)} document chunks.")


# Now we can use the embeddings to create a vector store
from langchain_community.vectorstores import Chroma

embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embedding_function,
    persist_directory="./chroma_db"
)

#vectorstore.persist()  # ✅ should now work

print("✅ Vectorstore persisted to './chroma_db'")



# 5. Perform similarity search

query = "When did Salaar movie released?"
search_results = vectorstore.similarity_search(query, k=5)

print(f"\nTop 5 most relevant chunks for the query: '{query}'\n")


#for i, result in enumerate(search_results, 1):
   #print(f"Result {i}:")
   #print(f"Source: {result.metadata.get('source', 'Unknown')}")
   #print(f"Content: {result.page_content}")
   #print()


retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
retriever.invoke("When did Salaar movie released?")

template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer: """
prompt = ChatPromptTemplate.from_template(template)

def docs2str(docs):
    return "\n\n".join(doc.page_content for doc in docs)

from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {"context": retriever | docs2str, "question": RunnablePassthrough()} | prompt | llm | output_parser
)

response = rag_chain.invoke("When did Salaar movie released?")
print(f"RAG Response: {response}")

question = "When did Salaar movie released?"






# This section builds a retrieval-augmented generation (RAG) system using LangChain.
# Purpose: To answer questions by retrieving relevant context from a document store and generating responses.
#Now Building a Conversational RAG system
from langchain_core.messages import AIMessage, HumanMessage
chat_history=[]
chat_history.extend([
    HumanMessage(content=question),
    AIMessage(content=response)
])


# This section builds a "contextualization" chain for conversational retrieval-augmented generation (RAG).
# Purpose:
# When a user asks a follow-up question in a chat, it may refer to previous context (e.g., "Who is the hero?").
# This chain reformulates such questions into standalone questions that make sense without chat history.
# 
# - contextualize_q_system_prompt: Instructs the LLM to rewrite the user's question as a standalone question, using chat history if needed.
# - contextualize_q_prompt: A ChatPromptTemplate that structures the prompt with a system message, chat history, and the latest user input.
# - contextualize_chain: Combines the prompt, LLM, and output parser so that when invoked, it returns a reformulated standalone question.
# - contextualize_chain.invoke(...): Runs the chain with a sample input and chat history to demonstrate how a follow-up question is rewritten.


from langchain_core.prompts import MessagesPlaceholder

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. DO NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# history_aware_retriever = create_history_aware_retriever(
#     llm, retriever, contextualize_q_prompt
# )

contextualize_chain = contextualize_q_prompt | llm | StrOutputParser()
contextualize_chain.invoke({"input": "Who is the hero?", "chat_history": chat_history})
#print("Contextualized Question:", contextualize_chain.invoke({"input": "Who is the hero?", "chat_history": chat_history}))


# This section creates a history-aware retriever that uses the LLM to contextualize questions based on chat history.
from langchain.chains import create_history_aware_retriever

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

history_aware_retriever.invoke({"input": "Who is the hero?", "chat_history": chat_history})





# This section creates a retrieval-augmented generation (RAG) chain that combines a retriever and a question-answering chain.
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use the following context to answer the user's question."),
    # ("system", "Tell me joke on Programming"),
    ("system", "Context: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

res1=rag_chain.invoke({"input": "Who is the hero?", "chat_history": chat_history})
print(res1)

#i want to use the logging module to log the response
import logging
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Log the response
logging.info(f"RAG Response: {res1}")
# Log the chat history
logging.info(f"Chat History: {chat_history}")
# Log the question
logging.info(f"Question: {question}")
# Log the vectorstore information
logging.info(f"Vectorstore: {vectorstore}")
# Log the prompt used for the RAG chain
logging.info(f"Prompt used for RAG chain: {qa_prompt}")