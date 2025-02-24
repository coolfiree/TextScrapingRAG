import streamlit as st
from langchain import PromptTemplate
from langchain import hub
from langchain.docstore.document import Document
from langchain.document_loaders import WebBaseLoader
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

#API KEY
GOOGLE_API_KEY = ""
#GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_API_KEY == ""
if (GOOGLE_API_KEY == "") :
    GOOGLE_API_KEY = st.text_input("Google API Key", type="password")
    st.info("Please add your Google API key to continue.", icon="🗝️")
    
#defining the app title and description
st.title('RAG App')
st.write('Enter the website URL and ask a question about it. The RAG model will answer!')

#asking the user for the website URL
website_url = st.text_input("Website URL")
#website_url = "https://blog.google/technology/ai/google-gemini-ai/"

#asking the user for the question
question = st.text_area(
        "Now give a keyword or ask a Question!",
        placeholder="Enter a Keyword?",
        disabled=not website_url,
)

if(website_url):
    loader = WebBaseLoader(website_url)
    docs = loader.load()

    # Extract the text from the website data document
    final_text = docs[0].page_content

    # Convert the text to LangChain's `Document` format
    docs =  [Document(page_content=final_text, metadata={"source": "local"})]

    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

    # Save to disk
    vectorstore = Chroma.from_documents(
                         documents=docs,                 # Data
                         embedding=gemini_embeddings,    # Embedding model
                         persist_directory="./chroma_db" # Directory to save data
                         )

    # Load from disk
    vectorstore_disk = Chroma(
                            persist_directory="./chroma_db",       # Directory of db
                            embedding_function=gemini_embeddings   # Embedding model
                       )
    # Get the Retriever interface for the store to use later.
    # When an unstructured query is given to a retriever it will return documents.
    retriever = vectorstore_disk.as_retriever(search_kwargs={"k": 1})

    llm = ChatGoogleGenerativeAI(model="gemini-pro",api_key=GOOGLE_API_KEY,
                     temperature=0.7, top_p=0.85)

    # Prompt template to query Gemini
    llm_prompt_template = """You are an assistant.
    Use the following context to answer the question.
    Try to give atleast 5 sentence and Keep the answer concise.\n
    Question: {question} \nContext: {context} \nAnswer:"""

    llm_prompt = PromptTemplate.from_template(llm_prompt_template)

    print(llm_prompt)

    # Combine data from documents to readable string format.
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
  
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | llm_prompt
        | llm
        | StrOutputParser()
    )
    
if(question):
    stream = rag_chain.invoke(question)
    st.write(stream)
