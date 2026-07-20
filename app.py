import streamlit as st
import os
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
            
        with st.spinner("Procesando y vectorizando documento..."):
            try:
                # 1. Cargar el tipo de archivo correspondiente
                if temp_file_path.endswith('.pdf'):
                    loader = PyPDFLoader(temp_file_path)
                else:
                    loader = CSVLoader(temp_file_path)
                
                docs = loader.load()
                
                # 2. Fragmentar el documento
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_documents(docs)
                
                # 3. Generar Embeddings y almacenar en FAISS
                embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
                st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
                
                st.success("✅ ¡Documento indexado con éxito!")
                
                # Eliminar archivo temporal
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
                        "Si no encuentras la respuesta en el contexto proporcionado, indícalo educadamente.\n\n"
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
