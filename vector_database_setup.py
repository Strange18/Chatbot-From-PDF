from dotenv import load_dotenv  # for api key
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings # for gemini embedding model loading
import PyPDF2   # for extracting text from pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter # splitting text 
from langchain_community.vectorstores import FAISS  # vector database
load_dotenv()
import os

# to get text from the pdf pages
def get_text_from_pdf(file):
    texts = ""
    pdfReader = PyPDF2.PdfReader(file)
    for i in pdfReader.pages:
        texts += i.extract_text()
    return texts

# to create chunks of the all the texts in the pdf
def create_text_chunks(texts):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = text_splitter.split_text(texts)
    return chunks

# to store the chunks in the vector database 
def get_vector_store(text_chunks):
    try:
        embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", google_api_key=os.getenv("GEMINI_API_KEY")
        )
        vector_store = FAISS.from_texts(texts=text_chunks, embedding=embedding_model, normalize_L2=True)
        vector_store.save_local('./db')
        print('stored to db!')
        return True
    except:
        return False
    
#  combining all the process  
def store_pdf_to_vector_db(file):
    print(f'file is {file}')
    text_output = get_text_from_pdf(file)
    chunks = create_text_chunks(text_output)
    return get_vector_store(chunks)
    

# store_pdf_to_vector_db(data_directory)


