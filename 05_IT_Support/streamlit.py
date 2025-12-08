# front end for chat with pdf
import streamlit as st
from app import  split_into_chunks, create_vectorstore, create_chat_chain, retrieve_data
from langchain_classic.prompts import ChatPromptTemplate

def load_data_into_VectorStore():
    with open("content/it_sector.txt", "r") as file:
        text = file.read()
    chunks = split_into_chunks(text)
    vectorstore = create_vectorstore(chunks)
    return vectorstore  

def main():
    # Streamlit app title
    st.title("Chat with IT Support Bot")

    # main chat interface
    if 'messages' not in st.session_state:
        st.session_state.messages = []
       
    user_input = st.text_input("You: ", key="input")
    if user_input:
        # Get response from chat chain
        # prepare a prompt to use only the data from the vector store and respond accordingly
        with st.spinner("Thinking..."): 
            context = retrieve_data(load_data_into_VectorStore(), user_input, k=4)
            chat_chain = create_chat_chain()
            response = chat_chain.predict(context=context, question=user_input)
            # Store messages in session state
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": response})

        # Display chat messages
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(msg["content"])


if __name__ == "__main__":
    main()