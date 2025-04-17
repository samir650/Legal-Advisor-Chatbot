import streamlit as st
import requests
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

# Configure OpenRouter API
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = "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"
MODEL_NAME = "deepseek/deepseek-r1:free"

uploaded_file = "القانون الكويتي.txt"

# Enhanced text extraction with legal structure awareness
def extract_text_from_txt(file):
    with open(file, "r", encoding="utf-8") as f:
        text = f.read()
    # Detect and preserve legal article structure
    text = re.sub(r'(المادة\s+\d+)', r'\n\n\1', text)
    return text

# Intelligent chunking based on legal structure
def split_text(text):
    articles = re.split(r'(المادة\s+\d+)', text)
    chunks = []
    current_chunk = ""
    for part in articles:
        if len(current_chunk) + len(part) > 2048:
            chunks.append(current_chunk.strip())
            current_chunk = part
        else:
            current_chunk += part
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Hybrid semantic search (TF-IDF + Sentence Transformers)
def find_relevant_chunk(question, chunks):
    # TF-IDF filtering
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(chunks + [question])
    tfidf_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    top_tfidf = np.argsort(tfidf_scores)[-5:]  # Get top 5 candidates

    # Semantic similarity with Sentence Transformers
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    question_emb = model.encode([question])
    chunk_embs = model.encode([chunks[i] for i in top_tfidf])
    similarities = cosine_similarity(question_emb, chunk_embs).flatten()
    
    best_index = top_tfidf[similarities.argmax()]
    return chunks[best_index], best_index

# Enhanced system prompt with legal reasoning examples
SYSTEM_PROMPT = """
أنت مستشار قانوني متخصص في قانون الجزاء الكويتي والقوانين المكملة له.
الردود يجب أن:
1. تكون دقيقة قانونيًا مع الإشارة إلى المواد ذات الصلة.
2. تتبع الهيكل القانوني المناسب.
3. تحافظ على مستوى مهني واحترافي.
4. تستخدم المصطلحات القانونية الصحيحة.
5. تعتمد على النصوص القانونية من المستند المرفق (إن وجد).
6. يجب أن تقتصر الإجابة على الأسئلة القانونية فقط، ولا يجوز الإجابة على الأسئلة العامة التي لا تتعلق بالقانون.
"""


# Document processing
if uploaded_file:
    file_text = extract_text_from_txt(uploaded_file)
    text_chunks = split_text(file_text)
    st.session_state["document_chunks"] = text_chunks
    st.session_state["chunk_vectorizer"] = TfidfVectorizer().fit(text_chunks)

# Conversation setup
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

# RTL styling
st.markdown("""
    <style>
    .stApp { direction: rtl; text-align: right; }
    .stChatInput textarea { direction: rtl !important; }
    .stChatMessage { direction: rtl; }
    </style>
    """, unsafe_allow_html=True)

st.title("مستشارك القانوني المتخصص")
st.caption("الرجاء طرح أسئلة محددة حول قانون الجزاء الكويتي")

# Conversation display
for message in st.session_state.messages[1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input handling
if prompt := st.chat_input("ما هو سؤالك القانوني؟"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Context retrieval
    context_text = SYSTEM_PROMPT
    if "document_chunks" in st.session_state:
        best_chunk, chunk_index = find_relevant_chunk(prompt, st.session_state["document_chunks"])
        context_text += f"\n\nالسياق القانوني:\n{best_chunk}"

    # API call with enhanced parameters
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "system", "content": context_text}] + st.session_state.messages[-5:],
        "temperature": 0.2,
        "max_tokens": 2048,
        "top_p": 0.95,
        "frequency_penalty": 0.2
    }

    with st.spinner("جارٍ التحليل القانوني..."):
        try:
            response = requests.post(OPENROUTER_API_URL, json=payload, headers=headers)
            response.raise_for_status()
            ai_response = response.json()['choices'][0]['message']['content']
            
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            with st.chat_message("assistant"):
                st.markdown(ai_response)
        except Exception as e:
            st.error(f"حدث خطأ: {str(e)}")
