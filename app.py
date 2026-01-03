import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings

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
st.markdown('<div class="main-title">🎓 AI Education Policy & FAQ Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Verified education information using RAG (SDG-4)</div>',
    unsafe_allow_html=True
)

# ---------------- LOAD FAISS (SAFE MODE) ----------------
embeddings = FakeEmbeddings(size=384)

vector_db = FAISS.load_local(
    "embeddings/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# ---------------- EDUCATION KEYWORDS ----------------
EDU_KEYWORDS = [
    "education", "scholarship", "student", "college",
    "university", "school", "policy", "scheme",
    "fee", "hostel", "exam", "degree"
]

# ---------------- USER INPUT ----------------
question = st.text_input(
    "🔍 Ask your question",
    placeholder="Example: What scholarships are available for students?"
)

# ---------------- BUTTON ----------------
if st.button("Get Answer"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        q_lower = question.lower()

        # 🔒 Hallucination Control Gate
        if not any(word in q_lower for word in EDU_KEYWORDS):
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="answer-title">📌 Retrieved Information</div>', unsafe_allow_html=True)
            st.write("❌ Information not available")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            docs = vector_db.similarity_search(question, k=3)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="answer-title">📌 Retrieved Information</div>', unsafe_allow_html=True)

            for i, doc in enumerate(docs, 1):
                st.markdown(f"**Source {i}:**")
                st.write(doc.page_content[:500])
                st.markdown("---")

            st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown(
    '<div class="footer">🌍 SDG-4: Quality Education | AI for Sustainability</div>',
    unsafe_allow_html=True
)
