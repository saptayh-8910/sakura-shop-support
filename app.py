import os
import shutil
import streamlit as st
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
DOCS_PATH       = "./docs"
PERSIST_DIR     = "./chroma_db"
REBUILD_FLAG    = "./chroma_db/.doc_count"

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="桜ショップ サポート / Sakura Shop Support",
    page_icon="🌸",
    layout="wide"
)

st.markdown("""
<style>
    .main { padding-top: 0.5rem; }
    .sakura-header {
        background: linear-gradient(135deg, #ff6b9d 0%, #c44b8a 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .sakura-header h1 { color: white; margin: 0; font-size: 1.8rem; }
    .sakura-header p { color: rgba(255,255,255,0.85); margin: 4px 0 0; font-size: 0.95rem; }
    .confidence-high { background: #EAF3DE; border-left: 4px solid #4CAF50; padding: 6px 12px; border-radius: 0 6px 6px 0; font-size: 12px; color: #27500A; margin-top: 6px; display: inline-block; }
    .confidence-medium { background: #FFF3E0; border-left: 4px solid #FF9800; padding: 6px 12px; border-radius: 0 6px 6px 0; font-size: 12px; color: #7B4A00; margin-top: 6px; display: inline-block; }
    .confidence-low { background: #FFEBEE; border-left: 4px solid #F44336; padding: 6px 12px; border-radius: 0 6px 6px 0; font-size: 12px; color: #8B0000; margin-top: 6px; display: inline-block; }
    .escalation-box { background: #E3F2FD; border: 1px solid #90CAF9; border-radius: 8px; padding: 12px 16px; margin-top: 8px; font-size: 13px; }
    .source-tag { font-size: 11px; color: #aaa; font-style: italic; margin-top: 4px; }
    .stChatMessage { border-radius: 12px; }
    .quick-btn { margin: 2px; }
</style>
""", unsafe_allow_html=True)

# ── Document loading ──────────────────────────────────────────────────
def count_docs():
    if not os.path.exists(DOCS_PATH):
        return 0
    return len([f for f in os.listdir(DOCS_PATH)
                if f.endswith((".md", ".txt", ".pdf"))])

def load_documents():
    if not os.path.exists(DOCS_PATH):
        os.makedirs(DOCS_PATH)
        return []
    loaders = []
    for file in sorted(os.listdir(DOCS_PATH)):
        fp = os.path.join(DOCS_PATH, file)
        if file.endswith(".pdf"):
            loaders.append(PyPDFLoader(fp))
        elif file.endswith((".txt", ".md")):
            loaders.append(TextLoader(fp, encoding="utf-8"))
    documents = []
    for loader in loaders:
        try:
            documents.extend(loader.load())
        except Exception as e:
            st.warning(f"Could not load: {e}")
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        separators=["。", "？", "！", "\n\n", "\n", "?", ".", " ", ""]
    )
    return splitter.split_documents(documents)

@st.cache_resource(show_spinner=False)
def get_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    docs_count = count_docs()
    needs_rebuild = True

    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        if os.path.exists(REBUILD_FLAG):
            with open(REBUILD_FLAG) as f:
                try:
                    old = int(f.read().strip())
                    if old == docs_count:
                        needs_rebuild = False
                except:
                    pass

    if needs_rebuild:
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
        documents = load_documents()
        if not documents:
            return None, 0
        chunks = split_documents(documents)
        os.makedirs(PERSIST_DIR, exist_ok=True)
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIR
        )
        with open(REBUILD_FLAG, "w") as f:
            f.write(str(docs_count))
        return vectordb, len(chunks)
    else:
        vectordb = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )
        return vectordb, vectordb._collection.count()

@st.cache_resource
def get_llm():
    return ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        temperature=0.1,
        max_tokens=1024
    )

# ── Core answer function ──────────────────────────────────────────────
def get_support_answer(question, retriever, llm, chat_history):
    # Build history
    history_text = ""
    for msg in chat_history[-4:]:
        if isinstance(msg, HumanMessage):
            history_text += f"Customer: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_text += f"Support: {msg.content}\n"

    # Retrieve relevant chunks
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)

    # Determine if question is in Japanese
    is_japanese = any(ord(c) > 0x3000 for c in question)

    prompt = f"""You are Sakura (さくら), a friendly and professional customer support agent for Sakura Shop (桜ショップ), a Japanese general marketplace similar to Rakuten or Amazon Japan.

Your personality:
- Warm, polite, and helpful — reflecting Japanese customer service excellence (おもてなし)
- Always address customers respectfully
- Be concise but thorough
- If you cannot answer from the knowledge base, be honest and offer to escalate

Language rule: {"Respond in Japanese (日本語で回答してください)" if is_japanese else "Respond in English"}

Previous conversation:
{history_text}

Knowledge base context:
{context}

Customer question: {question}

Instructions:
1. Answer based ONLY on the knowledge base context above
2. If the answer is clearly in the context: answer confidently — start your response with [HIGH]
3. If you can partially answer: answer what you can — start with [MEDIUM]  
4. If the answer is NOT in the context: start with [LOW] and offer to connect them with a human agent
5. After [HIGH/MEDIUM/LOW], write your actual answer naturally without showing the tag again
6. Keep your answer focused and helpful

Answer:"""

    response = llm.invoke(prompt)
    full_response = response.content

    # Parse confidence
    confidence = "medium"
    answer = full_response

    if full_response.startswith("[HIGH]"):
        confidence = "high"
        answer = full_response[6:].strip()
    elif full_response.startswith("[MEDIUM]"):
        confidence = "medium"
        answer = full_response[8:].strip()
    elif full_response.startswith("[LOW]"):
        confidence = "low"
        answer = full_response[5:].strip()

    needs_escalation = confidence == "low"

    return answer, confidence, needs_escalation, docs

# ── Quick question suggestions ────────────────────────────────────────
QUICK_QUESTIONS_EN = [
    "How long does shipping take?",
    "What is your return policy?",
    "What payment methods do you accept?",
    "How do I track my order?",
    "What are Premium membership benefits?",
]

QUICK_QUESTIONS_JP = [
    "配送にはどのくらいかかりますか？",
    "返品ポリシーを教えてください",
    "どの支払い方法が使えますか？",
    "注文を追跡するにはどうすればいいですか？",
    "プレミアム会員の特典は何ですか？",
]

# ══ MAIN UI ═══════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="sakura-header">
    <h1>🌸 Sakura Shop Support / 桜ショップ サポート</h1>
    <p>Hello! I'm Sakura, your AI support assistant. I can help in English and Japanese. / こんにちは！AIサポートのさくらです。日本語・英語でお手伝いします。</p>
</div>
""", unsafe_allow_html=True)

# Layout
col_chat, col_sidebar = st.columns([3, 1])

with col_sidebar:
    st.markdown("### 💬 Quick Questions")
    st.markdown("**English:**")
    for q in QUICK_QUESTIONS_EN:
        if st.button(q, key=f"en_{q}", use_container_width=True):
            st.session_state.quick_question = q

    st.markdown("**日本語:**")
    for q in QUICK_QUESTIONS_JP:
        if st.button(q, key=f"jp_{q}", use_container_width=True):
            st.session_state.quick_question = q

    st.divider()
    st.markdown("### 📞 Human Support")
    st.markdown("""
    Can't find your answer?
    - 💬 Live chat: sakurashop.jp
    - 📧 support@sakurashop.jp
    - 📞 0120-SAKURA
    - Mon-Fri 9AM-6PM JST
    """)
    st.divider()
    st.caption("🌸 Sakura Shop AI Support")
    st.caption("Powered by Claude + RAG")

with col_chat:
    # Load vector store
    with st.spinner("🌸 Loading Sakura Support knowledge base..."):
        vectordb, chunk_count = get_vector_store()

    if vectordb is None:
        st.error("Knowledge base not found. Please add FAQ documents to the docs/ folder.")
        st.stop()

    # Init session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "quick_question" not in st.session_state:
        st.session_state.quick_question = None

    # Display messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🌸" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])
            if msg.get("confidence"):
                conf = msg["confidence"]
                if conf == "high":
                    st.markdown("<span class='confidence-high'>✓ High confidence — answered from knowledge base</span>", unsafe_allow_html=True)
                elif conf == "medium":
                    st.markdown("<span class='confidence-medium'>~ Medium confidence — partial information available</span>", unsafe_allow_html=True)
                elif conf == "low":
                    st.markdown("<span class='confidence-low'>↗ Low confidence — human agent recommended</span>", unsafe_allow_html=True)
            if msg.get("escalation"):
                st.markdown("""<div class='escalation-box'>
                    🙋 <strong>Connect with a human agent:</strong><br>
                    Email: support@sakurashop.jp | Phone: 0120-SAKURA | Live chat: sakurashop.jp
                </div>""", unsafe_allow_html=True)

    # Welcome message
    if not st.session_state.messages:
        with st.chat_message("assistant", avatar="🌸"):
            st.markdown("""Welcome to Sakura Shop Support! 🌸

I'm Sakura, your AI assistant. I can help you with:
- 📦 Orders and shipping / 注文と配送
- 🔄 Returns and refunds / 返品と返金
- 💳 Payment methods / お支払い方法
- 👤 Account and membership / アカウントと会員
- 🛍️ Products and sellers / 商品と販売者

Ask me anything in English or Japanese! / 日本語・英語でご質問ください！

*Try a quick question on the right → / 右側のクイック質問をお試しください →*
            """)

    # Handle quick question
    prompt = None
    if st.session_state.quick_question:
        prompt = st.session_state.quick_question
        st.session_state.quick_question = None

    # Chat input
    user_input = st.chat_input("Ask Sakura anything... / 何でも聞いてください...")
    if user_input:
        prompt = user_input

    if prompt:
        # Show user message
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get answer
        with st.chat_message("assistant", avatar="🌸"):
            with st.spinner("🌸 Sakura is thinking... / 考え中..."):
                llm = get_llm()
                retriever = vectordb.as_retriever(search_kwargs={"k": 4})
                answer, confidence, needs_escalation, source_docs = get_support_answer(
                    prompt, retriever, llm, st.session_state.chat_history
                )

            st.markdown(answer)

            # Confidence badge
            if confidence == "high":
                st.markdown("<span class='confidence-high'>✓ High confidence — answered from knowledge base</span>", unsafe_allow_html=True)
            elif confidence == "medium":
                st.markdown("<span class='confidence-medium'>~ Medium confidence — partial information available</span>", unsafe_allow_html=True)
            elif confidence == "low":
                st.markdown("<span class='confidence-low'>↗ Connecting you to a human agent is recommended</span>", unsafe_allow_html=True)

            # Escalation box
            if needs_escalation:
                st.markdown("""<div class='escalation-box'>
                    🙋 <strong>Let me connect you with our human support team:</strong><br>
                    📧 support@sakurashop.jp &nbsp;|&nbsp; 📞 0120-SAKURA &nbsp;|&nbsp; 💬 sakurashop.jp/chat
                </div>""", unsafe_allow_html=True)

        # Update history
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        st.session_state.chat_history.append(AIMessage(content=answer))
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "confidence": confidence,
            "escalation": needs_escalation
        })
