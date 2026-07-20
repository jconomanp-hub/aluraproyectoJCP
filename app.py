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

# Cargar variables de entorno (Local o en el Servidor)
load_dotenv()

class RateLimitedEmbeddings:
    """
    Clase contenedora para GoogleGenerativeAIEmbeddings que maneja los límites de cuota (Rate Limits / Error 429).
    Envía los textos en lotes pequeños y añade pausas de seguridad y reintentos automáticos.
    """
    def __init__(self, model="models/gemini-embedding-001", batch_size=20, delay_seconds=3):
        self.underlying_embeddings = GoogleGenerativeAIEmbeddings(model=model)
        self.batch_size = batch_size
        self.delay_seconds = delay_seconds

    def embed_documents(self, texts):
        embeddings = []
        total_texts = len(texts)
        
        # Procesar los fragmentos en lotes (batches) controlados
        for i in range(0, total_texts, self.batch_size):
            batch = texts[i:i + self.batch_size]
            retries = 5
            
            for attempt in range(retries):
                try:
                    # Intenta obtener los embeddings para el lote actual
                    batch_embeddings = self.underlying_embeddings.embed_documents(batch)
                    embeddings.extend(batch_embeddings)
                    break
                except Exception as e:
                    # Si recibimos un error de límite de cuota (429), aplicamos retroceso exponencial
                    if "429" in str(e) and attempt < retries - 1:
                        sleep_time = (2 ** attempt) + 5
                        time.sleep(sleep_time)
                    else:
                        raise e
            
            # Pausa de seguridad fija entre lotes exitosos para no saturar la API
            time.sleep(self.delay_seconds)
            
        return embeddings

    def embed_query(self, text):
        # Para consultas individuales no suele haber problemas de límite de cuota
        return self.underlying_embeddings.embed_query(text)

# Configuración de página de Streamlit
st.set_page_config(page_title="Alura Agente - OCI", page_icon="🤖", layout="centered")
st.title("🤖 Alura Agente Corporativo")
st.subheader("Tu asistente inteligente de documentos en la nube")

# Verificar si existe la API Key de Gemini
if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("🔑 API Key de Google/Gemini no configurada. Asegúrate de definir la variable de entorno 'GOOGLE_API_KEY'.")
    st.stop()

# Inicializar estados de la sesión
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Barra lateral para carga de archivos
with st.sidebar:
    st.header("📂 Configuración del Documento")
    uploaded_file = st.file_uploader("Sube un archivo corporativo (PDF o CSV)", type=["pdf", "csv"])
    
    if uploaded_file is not None:
        # Guardar archivo de manera temporal
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        # Contenedor de progreso visual para el procesamiento
        progress_placeholder = st.empty()
        with progress_placeholder.container():
            with st.spinner("Procesando y vectorizando documento de forma segura..."):
                try:
                    # 1. Cargar el tipo de archivo correspondiente
                    if temp_file_path.endswith('.pdf'):
                        loader = PyPDFLoader(temp_file_path)
                    else:
                        loader = CSVLoader(temp_file_path)
                    
                    docs = loader.load()
                    
                    # 2. Fragmentar el documento (Optimizamos tamaños para reducir la cantidad de fragmentos)
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
                    chunks = splitter.split_documents(docs)
                    
                    # 3. Generar Embeddings usando la clase controladora de cuotas
                    embeddings = RateLimitedEmbeddings(
                        model="models/gemini-embedding-001",
                        batch_size=15,    # Enviar en paquetes de 15 textos
                        delay_seconds=3   # Esperar 3 segundos entre paquetes para respetar los límites de la API
                    )
                    
                    st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
                    
                    st.success("✅ ¡Documento indexado con éxito!")
                    
                    # Eliminar archivo temporal
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                except Exception as e:
                    st.error(f"Error procesando el archivo: {e}")

# Zona principal de Chat interactivo
if st.session_state.vector_store is None:
    st.info("👈 Sube un archivo PDF o CSV en la barra lateral para comenzar a chatear con el Agente.")
else:
    # Mostrar historial de conversación
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Entrada de pregunta del usuario
    if user_query := st.chat_input("¿Qué deseas saber sobre este documento?"):
        # Mostrar consulta del usuario en el chat
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        # Generar respuesta del agente RAG
        with st.chat_message("assistant"):
            with st.spinner("Buscando en el documento..."):
                try:
                    # Configurar cadena RAG
                    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
                    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
                    
                    system_prompt = (
                        "Eres un asistente inteligente corporativo.\n"
                        "Usa los siguientes fragmentos de contexto para responder de forma concisa y profesional.\n"
                        "Si no encuentras la respuesta en el contexto proporcionado, indícalo de forma educada y profesional.\n\n"
                        "Contexto:\n{context}"
                    )
                    
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", "{input}"),
                    ])
                    
                    document_chain = create_stuff_documents_chain(llm, prompt)
                    rag_chain = create_retrieval_chain(retriever, document_chain)
                    
                    # Invocar cadena
                    response = rag_chain.invoke({"input": user_query})
                    answer = response["answer"]
                    
                    st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error generando respuesta: {e}")
