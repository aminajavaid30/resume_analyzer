import streamlit as st
from frontend.main_app import render_main_app
from frontend.chat_interface import render_chat_interface

# Set the page layout to wide for better visual presentation
st.set_page_config(layout="wide")

def main():
    st.title("Resume Analyzer")

    # Create two columns with a 3:2 ratio for layout
    col1, col2 = st.columns([3, 2])

    with col1:
        # Render the main app in the larger column
        render_main_app()

    with col2:
        # Render the chat interface in the smaller column
        render_chat_interface()

# Script execution through the 'main' function
if __name__ == "__main__":
    main()