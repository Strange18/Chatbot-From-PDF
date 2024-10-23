from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
load_dotenv()

# to get text from the pdf pages
def get_text_from_pdf(files):
    texts = ""
    for pdf in files:
        pdfReader = PyPDF2.PdfReader(pdf)
        for i in pdfReader.pages:
            texts += i.extract_text()
    return texts

# to create chunks of the all the texts in the pdf
def create_text_chunks(texts):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = text_splitter.split_text(texts)
    return chunks

# to store the chunks in the vector database 
def get_vector_store(text_chunks):
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=os.getenv("GEMINI_API_KEY")
    )
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embedding_model)
    vector_store.save_local('./db')
    print('stored to db!')
    
#  combine all the process  
def store_pdf_to_vector_db(location):
    files = os.listdir(location)
    files = [
        open(os.path.join(location, x), "rb") for x in files if x.endswith(".pdf")
    ]
    text_output = get_text_from_pdf(files)
    chunks = create_text_chunks(text_output)
    get_vector_store(chunks)
    

data_directory = "./data/"

store_pdf_to_vector_db(data_directory)


