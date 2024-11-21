from dotenv import load_dotenv  # for loading the environment variable
import os
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)  # for embedding model and chat model of google
from langchain.chains.combine_documents import (
    create_stuff_documents_chain,
)  # to create chain to ingest the document
from langchain_community.vectorstores import FAISS  # vector database
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from tools import extract_details

load_dotenv()

chat_history = []


# creating a chain
def conversational_chain():
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=os.getenv("GEMINI_API_KEY")
    )
    vector_store = FAISS.load_local(
        "./db",
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
        normalize_L2=True,
    )
    # prompt for history aware retriver from the documentation
    condense_question_system_template = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    model = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0,
    )
    condense_question_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", condense_question_system_template),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm=model,
        retriever=vector_store.as_retriever(),
        prompt=condense_question_prompt,
    )
    # taking in consideration the annual report of Thapathali Campus
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a knowledgeable student at Thapathali Campus, eager to help visitors and prospective students. 
                You are well-versed in the campus's history, academic programs, student life, and facilities. Answer questions about Thapathali Campus in a friendly, informative, and engaging manner. 
                Answer them in a polite way as if you are a teacher.
            - If you cannot fully answer the question, let the user know gently and suggest related topics they may be interested in.
            - If the user greets you, respond warmly in the same manner. But do not repeat the greetings
            - Always maintain a helpful, conversational style.
            - Donot Mention any source of information, if needed include **as per my knowledge** or similar type of sentences.
            - Always try to provide full information about the matter asked by the user with correct deatils as possible
            - If user intends to book an appointment prompt the user to enter details such as 
                - name
                - email
                - phonenumber
                - date of appointment 
            Explain to the user you can send the email to the user regarding the confirmation of appointment when you get all the details required else email wont be sent but dont say these words directly
            {context}
            """,
            ),
            ("placeholder", "{chat_history}"),
            ("human", "I want to know about {input}"),
        ]
    )
    qa_chain = create_stuff_documents_chain(llm=model, prompt=prompt)

    chain = create_retrieval_chain(history_aware_retriever, qa_chain)
    return chain


def user_query(user_question):
    global chat_history
    chain = conversational_chain()
    response = chain.invoke(
        {
            "input": user_question,
            "chat_history": [chat["role"] for chat in chat_history],
        },
        return_only_outputs=False,
    )

    chat_history.append({"role": user_question, "content": response["answer"]})

    chat_history_str = " ".join(chat["role"] for chat in chat_history)
    if chat_history_str is not None:
        user_details = extract_details(chat_history_str)
        print(user_details)
    return response["answer"]
