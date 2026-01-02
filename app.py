import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Education Assistant",
    page_icon="🎓",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 36px;
    font-weight: 700;
    color: #2c3e50;
}
.subtitle {
    text-align: center;
    font-size: 16px;
    color: #555;
    margin-bottom: 30px;
}
.card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    margin-top: 20px;
}
.answer-title {
    font-size: 20px;
    font-weight: 600;
    color: #1f77b4;
    margin-bottom: 10px;
}
.footer {
    text-align: center;
    font-size: 14px;
    color: #777;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown(
    '<div class="main-title">🎓 AI Education Policy & FAQ Assistant</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Retrieval-Augmented AI for verified education information (SDG-4)</div>',
    unsafe_allow_html=True
)

# ---------------- LOAD EMBEDDINGS ----------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------- LOAD FAISS VECTOR DB ----------------
vector_db = FAISS.load_local(
    "embeddings/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# ---------------- USER INPUT ----------------
question = st.text_input(
    "🔍 Ask your question",
    placeholder="Example: What scholarships are available for students?"
)

# ---------------- BUTTON ACTION ----------------
if st.button("Get Answer"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        # Retrieve documents WITH similarity score
        results = vector_db.similarity_search_with_score(question, k=3)

        # Similarity threshold (IMPORTANT for hallucination control)
        SIMILARITY_THRESHOLD = 0.6   # smaller = stricter
        relevant_docs = []

        for doc, score in results:
            if score < SIMILARITY_THRESHOLD:
                relevant_docs.append(doc)

        # ---------------- DISPLAY ANSWER ----------------
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="answer-title">📌 Retrieved Information</div>',
            unsafe_allow_html=True
        )

        if not relevant_docs:
            st.write("❌ Information not available")
        else:
            for i, doc in enumerate(relevant_docs, 1):
                st.markdown(f"**Source {i}:**")
                st.write(doc.page_content[:500])
                st.markdown("---")

        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown(
    '<div class="footer">🌍 SDG-4: Quality Education | AI for Sustainability Capstone Project</div>',
    unsafe_allow_html=True
)
