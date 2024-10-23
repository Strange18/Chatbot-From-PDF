from dotenv import load_dotenv
import os 
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from  langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
load_dotenv()

def conversational_chain():
    prompt_template = '''
    Assume you are expert in the topic {context}. 
    Answer the {question} based on your expertise and say not in the context if the question is irrelavant.
    Talk in a natural tone.
    '''
    prompt = PromptTemplate(template=prompt_template, input_variables=['context','question'])
    model = ChatGoogleGenerativeAI(model='gemini-pro', google_api_key=os.getenv('GEMINI_API_KEY'))
    chain = create_stuff_documents_chain(llm=model, prompt=prompt)
    
    return chain


    
def user_query(user_question):
    embedding_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001',google_api_key=os.getenv('GEMINI_API_KEY'))
    vector_store = FAISS.load_local("./db",embeddings=embedding_model, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)
    chain = conversational_chain()
    
    response = chain.invoke(
        {"context":docs, "question":user_question}, return_only_outputs=False)
    return response
    
