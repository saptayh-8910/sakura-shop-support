# Sakura Shop AI Customer Support / 桜ショップ AIカスタマーサポート

> A production-ready AI customer support bot for a Japanese general marketplace — with confidence scoring, human escalation, and bilingual Japanese/English support.

**Built as Level 2 — Project 2A of an AI Engineer targeting the Japanese tech market.**

---

## Live Demo

 **[Try Sakura Shop Support →](https://sakura-shop-support-mn9v3asmjgcfh7chhberlx.streamlit.app)**

---

##  What Makes This Different

| Feature | Why It Matters |
|---|---|
| **Confidence scoring** (High/Medium/Low) | Honest AI — shows certainty level on every answer |
| **Human escalation** | When AI can't answer, it connects to human support |
| **Japanese-first design** | Answers in the language of the question |
| **Cross-lingual retrieval** | Ask in English, retrieve from Japanese FAQ |
| **Brand persona "Sakura"** | おもてなし (hospitality) customer service style |
| **Quick question buttons** | UX-first — reduces friction for customers |

---

##  How It Works

```
Customer Question (EN or JP)
        ↓
Semantic Search → Sakura Shop FAQ Knowledge Base
        ↓
Confidence Assessment (HIGH / MEDIUM / LOW)
        ↓
HIGH → Answer directly from knowledge base
MEDIUM → Partial answer + caveat
LOW → Honest "I don't know" + human escalation
        ↓
Answer in customer's language + source citation
```

---

## Supported Topics

-  Orders & shipping / 注文と配送
-  Returns & refunds / 返品と返金
-  Payment methods / お支払い方法
-  Account & Premium membership / アカウントと会員
-  Products & sellers / 商品と販売者
-  Customer support escalation / サポートエスカレーション

---

##  Setup

```bash
git clone https://github.com/saptayh-8910/sakura-shop-support.git
cd sakura-shop-support

python3 -m venv venv
source venv/bin/activate

pip install langchain langchain-anthropic langchain-community \
    langchain-text-splitters langchain-huggingface langchain-chroma \
    chromadb pypdf sentence-transformers python-dotenv streamlit
```

Create `.env`:
```
ANTHROPIC_API_KEY=your_key_here
```

Run:
```bash
streamlit run app.py
```

---

##  Stack

- **LLM**: Claude Haiku (`claude-haiku-4-5-20251001`) — fast, cost-efficient
- **Embeddings**: `intfloat/multilingual-e5-large` — Japanese + English
- **Vector Store**: ChromaDB (local, persistent)
- **Framework**: LangChain + Streamlit
- **Knowledge Base**: Bilingual FAQ (Japanese + English)

---

##  Japan Market Relevance

This architecture mirrors customer support AI systems deployed at:
- **PayPay** — 24/7 automated support with human escalation
- **Mercari** — Confidence-scored answers with source attribution
- **Rakuten Ichiba** — Bilingual JP/EN marketplace support

Cost to run: ~¥3,000–5,000/month for a small business deployment.

---

##  Project Structure

```
sakura-shop-support/
├── app.py                  # Main Streamlit application
├── docs/
│   └── sakura_faq.md       # Bilingual FAQ knowledge base
├── .gitignore
└── README.md
```

---
