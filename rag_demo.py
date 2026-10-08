
import os
import streamlit as st

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama


# -----------------------------------------
# PAGE SETTINGS
# -----------------------------------------

st.set_page_config(
    page_title="College RAG Assistant",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 College RAG Assistant")
st.caption(
    "Learn how Retrieval-Augmented Generation works, step by step."
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #101827, #172554);
}
h1, h2, h3, p, label {
    color: #F1F5F9;
}
div[data-testid="stExpander"] {
    border: 1px solid #475569;
    border-radius: 12px;
}
.answer-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #16352C;
    border: 1px solid #22C55E;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------
# EMBEDDING MODEL
# -----------------------------------------

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# -----------------------------------------
# SIDEBAR SETTINGS
# -----------------------------------------

st.sidebar.header("⚙️ RAG Settings")

top_k = st.sidebar.slider(
    "Number of retrieved chunks (Top-K)",
    min_value=1,
    max_value=5,
    value=3
)

chunk_size = st.sidebar.slider(
    "Chunk size",
    min_value=100,
    max_value=800,
    value=300,
    step=50
)

chunk_overlap = st.sidebar.slider(
    "Chunk overlap",
    min_value=0,
    max_value=100,
    value=50,
    step=10
)

st.sidebar.info(
    "Ollama must be installed, and the llama3.2 model "
    "must be available on your computer."
)


# -----------------------------------------
# DOCUMENT INPUT
# -----------------------------------------

st.header("📄 1. Choose your college document")

uploaded_file = st.file_uploader(
    "Upload college.txt",
    type=["txt"]
)

local_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "college.txt"
)

if uploaded_file is not None:
    document_text = uploaded_file.getvalue().decode("utf-8")
    source_name = uploaded_file.name

elif os.path.exists(local_path):
    with open(local_path, "r", encoding="utf-8") as file:
        document_text = file.read()
    source_name = "college.txt"

else:
    document_text = ""
    source_name = ""

if document_text.strip():
    st.success(f"Document ready: {source_name}")

    with st.expander("👀 View original document"):
        st.text(document_text)
else:
    st.warning(
        "Place college.txt beside app.py or upload the file above."
    )


# -----------------------------------------
# QUESTION INPUT
# -----------------------------------------

st.header("💬 2. Ask a question")

question = st.text_input(
    "Your question",
    placeholder="What programming languages are taught?"
)

run_button = st.button(
    "🚀 Run RAG Pipeline",
    type="primary",
    use_container_width=True
)


# -----------------------------------------
# RUN THE RAG PIPELINE
# -----------------------------------------

if run_button:

    if not document_text.strip():
        st.error("Please provide a non-empty college.txt file.")

    elif not question.strip():
        st.error("Please enter a question.")

    else:
        try:
            # STEP 1: DOCUMENT LOADING
            with st.expander(
                "✅ Step 1: Document Loading",
                expanded=True
            ):
                documents = [
                    Document(
                        page_content=document_text,
                        metadata={"source": source_name}
                    )
                ]

                st.success("Document loaded successfully!")
                st.write("Number of documents:", len(documents))
                st.write("Source:", source_name)
                st.text_area(
                    "Loaded content",
                    documents[0].page_content,
                    height=150,
                    disabled=True
                )

            # STEP 2: TEXT SPLITTING
            with st.expander(
                "✂️ Step 2: Text Splitting",
                expanded=True
            ):
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )

                chunks = splitter.split_documents(documents)

                st.success("Text splitting completed!")
                st.write("Total chunks:", len(chunks))

                for i, chunk in enumerate(chunks, start=1):
                    st.markdown(f"**Chunk {i}**")
                    st.code(chunk.page_content)

            if not chunks:
                st.error("No text chunks were created.")
                st.stop()

            # STEP 3: EMBEDDINGS
            with st.expander(
                "🔢 Step 3: Embedding Generation",
                expanded=True
            ):
                embeddings = load_embeddings()

                sample_vector = embeddings.embed_query(
                    chunks[0].page_content
                )

                st.success("Embedding model loaded!")
                st.write(
                    "Embedding vector dimensions:",
                    len(sample_vector)
                )
                st.write(
                    "First 10 values:",
                    sample_vector[:10]
                )

                st.caption(
                    "Chroma will also generate embeddings for all "
                    "document chunks when they are added."
                )

            # STEP 4: CHROMA VECTOR STORE
            with st.expander(
                "🗄️ Step 4: Chroma Vector Store",
                expanded=True
            ):
                vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    collection_name="college_rag_streamlit"
                )

                st.success("Chunks stored in Chroma!")
                st.write(
                    "Stored chunks:",
                    len(vector_store.get()["ids"])
                )
                st.write("Storage: In-memory vector store")

            # STEP 5: RETRIEVAL
            with st.expander(
                "🔎 Step 5: Top-K Retrieval",
                expanded=True
            ):
                results = (
                    vector_store.similarity_search_with_score(
                        question,
                        k=min(top_k, len(chunks))
                    )
                )

                retrieved_docs = []

                st.write("Question:", question)
                st.write("Requested Top-K:", top_k)
                st.write("Results returned:", len(results))

                for rank, (doc, distance) in enumerate(
                    results, start=1
                ):
                    retrieved_docs.append(doc)

                    st.markdown(f"**Retrieved Chunk {rank}**")
                    st.write("Distance:", round(distance, 4))
                    st.code(doc.page_content)

            # STEP 6: CONTEXT CONSTRUCTION
            with st.expander(
                "🧩 Step 6: Context Construction",
                expanded=True
            ):
                context = "\n\n".join(
                    doc.page_content
                    for doc in retrieved_docs
                )

                st.success("Context constructed successfully!")
                st.text_area(
                    "Context sent to AI",
                    context,
                    height=200,
                    disabled=True
                )

            # STEP 7: FINAL RAG ANSWER
            with st.expander(
                "🤖 Step 7: Final RAG Answer",
                expanded=True
            ):
                prompt = ChatPromptTemplate.from_messages([
                    (
                        "system",
                        """You are a helpful college assistant.
Answer using only the supplied context.
If the context does not contain the answer, say:
'I could not find that information in college.txt.'
Do not invent college facts."""
                    ),
                    (
                        "human",
                        "Context:\n{context}\n\nQuestion:\n{question}"
                    )
                ])

                llm = ChatOllama(
                    model="llama3.2",
                    temperature=0
                )

                chain = prompt | llm

                with st.spinner("AI is generating your answer..."):
                    response = chain.invoke({
                        "context": context,
                        "question": question
                    })

                st.markdown(
                    '<div class="answer-box">'
                    '<h3>💡 Final Answer</h3>'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.markdown(response.content)

                st.success("RAG pipeline completed successfully!")

        except Exception as error:
            st.error("Something went wrong.")
            st.code(str(error))

            st.info(
                "Check that your dependencies are installed, "
                "the embedding model has downloaded, and Ollama "
                "is running with the llama3.2 model."
            )


# -----------------------------------------
# FOOTER
# -----------------------------------------

st.divider()

st.caption(
    "Built with Python, Streamlit, LangChain, "
    "Hugging Face, Chroma and Ollama."
)