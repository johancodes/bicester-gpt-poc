
# import langchain dependencies
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma 
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain import OpenAI 
from langchain.chains import RetrievalQA 
from langchain.document_loaders import DirectoryLoader

# import other dependencies
import streamlit as st 
import openai
import os

# if using streamlit to store key:
# use key in streamlit secret: https://blog.streamlit.io/secrets-in-sharing-apps/ (NB: this is hosted by Streamlit on their cloud server)
os.environ["OPENAI_API_KEY"] == st.secrets["OPENAI_API_KEY"]
openai.api_key = st.secrets["OPENAI_API_KEY"]

# load the source documents / dataset
loader = DirectoryLoader('./', glob='**/*.txt')
docs = loader.load()

# chunk text
#char_text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#if running into token issues use: 
char_text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0) 
doc_texts = char_text_splitter.split_documents(docs)

# embed the dataset and store in local ChromaDB
openAI_embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
vStore = Chroma.from_documents(doc_texts, openAI_embeddings)

# define model for Q and A using OpenAI's LLM
model = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=vStore.as_retriever())

# to run app in browser, $ streamlit run path/filename.py (ensure you are on the right conda environment with all the dependencies)
st.title('"Got questions on Bicester Village London? Ask me anything, in multiple languages"')

query = st.text_input('Type your query below and press Enter')
if query:
    response = model.run(query + '. Respond in same language as query language.')
    st.write(response)

st.text('')
st.write('(Text from selected Bicester Village London webpages were used in this demo. Proof of Concept using LLM Gen AI created by joel.lim@zensar.com)')