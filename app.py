import streamlit as st
from extract_data import user_query

# from tools import extract_details
from datetime import datetime
from vector_database_setup import store_pdf_to_vector_db
import io
import os


def chat_bot(query):
    response = user_query(query)
    return response


def main():
    st.title("Chat Bot")

    # custom CSS for chat history
    st.markdown(
        """
        <style>
        .chat-container {
            max-height: 400px;  /* Set max height for scrollable area */
            overflow-y: auto;   /* Enable vertical scroll */
            padding: 10px;
            margin-bottom: 10px;
        }
        .user-message {
            text-align: right;
            padding: 8px;
            border-radius: 10px;
            margin: 5px;
        }
        .bot-message {
            text-align: left;
            padding: 8px;
            border-radius: 10px;
            margin: 5px;
        }
        .message-time {
            font-size: 0.8em;
            color: gray;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "extracted_details" not in st.session_state:
        st.session_state.extracted_details = {}

    st.sidebar.title("Upload PDF")
    uploaded_pdf = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

    # the uploaded pdf is stored in the vector database
    if uploaded_pdf is not None:
        pdf_file = io.BytesIO(uploaded_pdf.read())
        full_path = os.path.join("data", "output.pdf")
        with open(full_path, "wb") as file:
            file.write(pdf_file.getvalue())
        store_pdf_to_vector_db("./data/output.pdf")
        st.sidebar.success("PDF uploaded and stored successfully!")

    # container for chat display
    chat_display = st.empty()

    def render_chat():
        with chat_display.container():
            st.subheader("Chat History")
            st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
            for user_msg, bot_msg, time in st.session_state.chat_history:
                st.markdown(
                    f"<div class='user-message'>{user_msg}<div class='message-time'>{time.strftime('%Y-%m-%d %H:%M:%S')}</div></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='bot-message'>{bot_msg}<div class='message-time'>{time.strftime('%Y-%m-%d %H:%M:%S')}</div></div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

    render_chat()

    if "user_data" not in st.session_state:
        st.session_state.user_data = ""  
    if "clear_input" not in st.session_state:
        st.session_state.clear_input = False 

    # Input box for the user
    user_input = st.text_area(
        "Enter your message:",
        value="" if st.session_state.clear_input else st.session_state.user_data,
        key="user_data",
        label_visibility="collapsed",
        placeholder="Ask a question",
    )

    if st.button("Send", key="send_button"):
        user_text = st.session_state.user_data
        print(user_text)

        st.session_state.clear_input = True

        bot_response = chat_bot(user_text)

        st.session_state.chat_history.append((user_text, bot_response, datetime.now()))

        # re-render chat history with the new message
        render_chat()


if __name__ == "__main__":
    main()
