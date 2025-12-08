# business logic for chat with pdf 
from langchain_classic.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
from langchain_classic.chains import LLMChain
from langchain_classic.vectorstores import FAISS
from langchain_classic.memory import ConversationBufferMemory
from dotenv import load_dotenv
from langchain_aws import ChatBedrock, BedrockEmbeddings   
from langchain_classic.prompts import ChatPromptTemplate

load_dotenv()

# Load pdf 
def load_pdf(file_paths):
    text = ""
    for pdf in file_paths:
        loader = PdfReader(pdf)
        pages = loader.pages
        for page in pages:
            text += page.extract_text()
    return text

# split pages into chunks
def split_into_chunks(pdf_content, chunk_size=1000, chunk_overlap=200):
    from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(pdf_content)
    return chunks

# chunks into embeddings and store in vectorstore 
def create_vectorstore(chunks):
    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

# retrie data from vectorstore 
def retrieve_data(vectorstore, query, k=4):
    results = vectorstore.similarity_search(query, k=k)
    return results

# create chat model and retrieval chain 
# temperature is set to 0 for deterministic responses
def create_chat_chain():
    systemInstruction = """You are an IT support assistant. Use the following context to answer the user's question.
            If the context does not contain the answer, respond with "I'm sorry, I don't have that information." Do not make up answers, answer in few sentenses only.
            Context: {context}
            Question: {question}"""
    prompt_template = ChatPromptTemplate.from_template(systemInstruction)
    chat_model = ChatBedrock(model_id="mistral.mistral-7b-instruct-v0:2", temperature=0)
    qa_chain = LLMChain(
        llm=chat_model,
        prompt=prompt_template,
        verbose=False,
    )
    return qa_chain 


# LLM RAG -> we get different chunks -> you process the rerived data [chunks] with the help of llm -> Grammer, language 