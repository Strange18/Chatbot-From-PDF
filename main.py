import streamlit as st
from extract_data import user_query

def chat_bot(query):
    response = user_query(query)
    return response


def main():
    st.title("Chat Bot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    col1, col2 = st.columns([1, 2])

    with col1:
        user_input = st.text_input("Enter your message:")
        if st.button("Send") and user_input:
            bot_response = chat_bot(user_input)
            st.session_state.chat_history.append((user_input, bot_response))

    with col2:
        st.subheader("Chat History")
        if st.session_state.chat_history:
            for i, (user_msg, bot_msg) in enumerate(
                st.session_state.chat_history[::-1]
            ):
                st.markdown(f"**You:** {user_msg}")
                st.markdown(f"**Bot:** {bot_msg}")
                st.markdown("---")  


if __name__ == "__main__":
    main()
