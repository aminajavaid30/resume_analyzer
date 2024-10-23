import streamlit as st
from backend.analysis import llm
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# Chat interface section of the application - displayed at the right
def render_chat_interface():
    st.header("Chat with the Resume")  # Header for the chat interface

    # Add CSS for fixing chat input position at the bottom
    st.markdown("""
        <style>
        .stChatInput {
            position: fixed;
            bottom: 0;
            right: 0;
            width: 40%;
            padding: 1rem;
            background-color: white;
            z-index: 1000;
        }
        .stChatFloatingInputContainer {
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)  # Injecting custom CSS for styling

    # Initialize empty chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = []  # Initialize messages in session state

    # Check if the vector store is available 
    if "vector_store" in st.session_state:
        # Setting up the vector store as retriever
        retriever = st.session_state.vector_store.as_retriever(
            search_type="mmr",  # Uses Maximum Marginal Relevance for search
            search_kwargs={
                "k": 3,        # Fetch top 3 chunks
            }
        )

        # Chat logic setup for contextualizing user questions
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        
        # Creating a prompt template for contextualizing questions
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        # Creating a history-aware retriever with the language model
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        # System prompt for answering questions
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        
        # Creating a prompt template for question-answering
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        # Setting up the question-answering chain
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        retrieval_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Chat history management using a dictionary
        store = {}

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in store:
                store[session_id] = ChatMessageHistory()  # Create a new history if not exists
            return store[session_id]  # Return the chat history

        # Creating a runnable chain with message history
        conversational_retrieval_chain = RunnableWithMessageHistory(
            retrieval_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        # Create a container for messages with bottom padding for input space
        chat_container = st.container()
        
        # Add space at the bottom to prevent messages from being hidden behind input
        st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)
        
        # Input box - will be fixed at bottom due to CSS
        prompt = st.chat_input("Ask about the resume")  # Input for user queries

        # Display messages in the container
        with chat_container:
            for message in st.session_state.messages:  # Iterate through session messages
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])  # Display message content

        if prompt:  # Check if there is a user input
            st.session_state.messages.append({"role": "user", "content": prompt})  # Store user message
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)  # Display user input

                with st.chat_message("assistant"):
                    # Prepare input data for the conversational chain
                    input_data = {
                        "input": prompt,
                        "chat_history": st.session_state.messages,
                    }
                    response = conversational_retrieval_chain.invoke(
                        input_data,
                        config={
                            "configurable": {"session_id": "abc123"}  # Setting session ID
                        },
                    )
                    answer_text = response['answer']  # Extract the assistant's response
                    st.markdown(answer_text)  # Display the response

            st.session_state.messages.append({"role": "assistant", "content": answer_text})  # Store assistant response
            
            # Force a rerun to update the chat immediately
            st.rerun()  # Refresh the Streamlit app

    else:
        st.info("Please upload a resume and analyze it to start chatting.")