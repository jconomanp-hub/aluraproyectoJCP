import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Cargar variables de entorno
load_dotenv()

class RateLimitedEmbeddings:
    """
    Clase ultra-conservadora para gestionar límites de la API gratuita.
    """
    def __init__(self, model="models/gemini-embedding-001", batch_size=80, delay_seconds=10):
        self.underlying_embeddings = GoogleGenerativeAIEmbeddings(model=model)
        self.batch_size = batch_size
        self.delay_seconds = delay_seconds

    def embed_documents(self, texts):
        embeddings = []
        total_texts = len(texts)
        
        for i in range(0, total_texts, self.batch_size):
            batch = texts[i:i + self.batch_size]
            retries = 5
            
            for attempt in range(retries):
                try:
                    batch_embeddings = self.underlying_embeddings.embed_documents(batch)
                    embeddings.extend(batch_embeddings)
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < retries - 1:
                        sleep_time = (2 ** attempt) + 15 # Espera exponencial más larga
                        time.sleep(sleep_time)
                    else:
                        raise e
            
            time.sleep(self.delay_seconds) # Pausa larga para respetar 15 RPM
            
        return embeddings

    def embed_query(self, text):
        return self.underlying_embeddings.embed_query(text)

# Configuración de página
st.set_page_config(page_title="Alura Agente - OCI", page_icon="🤖", layout="centered")
st.title("🤖 Alura Agente Corporativo")

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("🔑 API Key no configurada en .env")
    st.stop()

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

with st.sidebar:
    st.header("📂 Configuración")
    uploaded_file = st.file_uploader("Sube un PDF o CSV", type=["pdf", "csv"])
    
    if uploaded_file is not None:
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        with st.spinner("Procesando con máxima optimización (paciencia, esto evita errores de cuota)..."):
            try:
                if temp_file_path.endswith('.pdf'):
                    loader = PyPDFLoader(temp_file_path)
                else:
                    loader = CSVLoader(temp_file_path)
                
                docs = loader.load()
                # Aumentamos chunk_size a 4000 para reducir el número de peticiones
                splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=300)
                chunks = splitter.split_documents(docs)
                
                embeddings = RateLimitedEmbeddings(delay_seconds=10) # 10s de espera
                st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
                
                st.success("✅ ¡Indexado con éxito!")
                if os.path.exists(temp_file_path): os.remove(temp_file_path)
            except Exception as e:
                st.error(f"Error crítico: {e}")

if st.session_state.vector_store is not None:
    if user_query := st.chat_input("Pregunta sobre el documento:"):
        with st.chat_message("user"): st.markdown(user_query)
        with st.chat_message("assistant"):
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
            prompt = ChatPromptTemplate.from_messages([("system", "Eres un asistente útil.\nContexto:\n{context}"), ("human", "{input}")])
            response = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt)).invoke({"input": user_query})
            st.markdown(response["answer"])
