import streamlit as st
import asyncio
from main import chatbot
import shutil

# --- STREAMLIT UI ---
st.set_page_config(page_title="Web Reader Chatbot", page_icon="🤖", layout="wide")
st.title("📖 Web Reader Chatbot")

st.sidebar.header("Settings")
url = st.sidebar.text_input("Enter Website URL", "https://playwright.dev")
query = st.sidebar.text_input("Ask a Question", "Describe playwright and its feature briefly")

st.markdown("---")
st.write("### 🤖 Chatbot Response")

col1, col2 = st.sidebar.columns(2)
if col1.button("Run Chatbot"):
    try:
        with st.spinner('Processing...'):
            response = chatbot(url, query)
        st.write(f"> {response['result']}")
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.write("> click **run** to start...")

if col2.button("Clear Cache"):
    shutil.rmtree("cache")
    st.success("Cache cleared successfully.")

st.sidebar.markdown("---")
st.sidebar.markdown("Developed by **Siddharth Chandel** 🚀")
st.sidebar.markdown("Let's connect on [LinkedIn](https://www.linkedin.com/in/siddharth-chandel-001097245/) !!!!")
