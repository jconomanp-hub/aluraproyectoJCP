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
from langchain_core.embeddings import Embeddings

# 1. Cargar variables de entorno
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# 2. Clase para manejar embeddings con límite de velocidad
class RateLimitedEmbeddings(Embeddings):
    def __init__(self, model="text-embedding-004", batch_size=5, delay_seconds=2):
        # Inicializamos explícitamente sin prefijo "models/"
        self.underlying_embeddings = GoogleGenerativeAIEmbeddings(
            model=model, 
            google_api_key=api_key
        )
        self.batch_size = batch_size
        self.delay_seconds = delay_seconds

    def embed_documents(self, texts):
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            embeddings.extend(self.underlying_embeddings.embed_documents(batch))
            time.sleep(self.delay_seconds)
        return embeddings

    def embed_query(self, text):
        return self.underlying_embeddings.embed_query(text)

# 3. Configuración de Streamlit
st.set_page_config(page_title="Alura Agente - OCI", page_icon="🤖", layout="centered")
st.title("🤖 Alura Agente Corporativo")

if not api_key:
    st.error("🔑 API Key no configurada. Asegúrate de tener: GOOGLE_API_KEY='tu_clave' en el archivo .env")
    st.stop()

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# 4. Lógica de procesamiento
with st.sidebar:
    st.header("📂 Configuración")
    uploaded_file = st.file_uploader("Sube un PDF o CSV", type=["pdf", "csv"])
    
    if uploaded_file is not None:
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        with st.spinner("Procesando documento..."):
            try:
                if temp_file_path.endswith('.pdf'):
                    loader = PyPDFLoader(temp_file_path)
                else:
                    loader = CSVLoader(temp_file_path)
                
                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = splitter.split_documents(docs)
                
                # Usamos el modelo estándar sin prefijo
                embeddings_indexer = RateLimitedEmbeddings(model="text-embedding-004", delay_seconds=1)
                st.session_state.vector_store = FAISS.from_documents(chunks, embeddings_indexer)
                
                st.success("✅ ¡Indexado con éxito!")
                if os.path.exists(temp_file_path): os.remove(temp_file_path)
            except Exception as e:
                st.error(f"Error crítico: {e}")

# 5. Lógica de Chat
if st.session_state.vector_store is not None:
    if user_query := st.chat_input("Pregunta sobre el documento:"):
        with st.chat_message("user"): 
            st.markdown(user_query)
        with st.chat_message("assistant"):
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", 
                temperature=0.3,
                google_api_key=api_key
            )
            
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Eres un asistente útil.\nContexto:\n{context}"), 
                ("human", "{input}")
            ])
            
            chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))
            response = chain.invoke({"input": user_query})
            st.markdown(response["answer"])
