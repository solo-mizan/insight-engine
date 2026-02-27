# 🚀 InsightEngine Pro: Agentic RAG with Hybrid Fallback

**InsightEngine Pro** is a production-grade research assistant that transforms static PDFs into an interactive knowledge base. Built using **LangGraph**, it moves beyond simple linear chains to implement an autonomous, self-correcting reasoning loop with a multi-cloud fallback strategy.

---

## 🏗️ The Challenge: Why this exists
Standard RAG (Retrieval-Augmented Generation) applications often fail in real-world scenarios due to:
1. **Hallucinations:** The AI ignores the context and makes up answers.
2. **API Fragility:** Rate limits (429 errors) crash the application.
3. **Linearity:** Basic chains can't "retry" if a search result is poor.

**InsightEngine Pro** solves these using a **Stateful Graph Architecture.**

---

## 🛠️ Tech Stack & Modern Tooling
* **Orchestration:** [LangGraph](https://www.langchain.com/langgraph) (Stateful Graph Workflows)
* **Primary Brain:** Google Gemini 2.0 Flash
* **Secondary Brain (Fallback):** Llama 3.3 70B (via Groq)
* **Vector Database:** ChromaDB
* **Package Management:** [uv](https://github.com/astral-sh/uv) (The fastest Python manager in the 2026 ecosystem)
* **Embeddings:** Google `text-embedding-004`

---

## 🧠 Advanced Engineering Features

### 1. Hybrid LLM Fallback (Fault Tolerance)
The system implements a multi-provider strategy. If the primary Gemini API returns a rate-limit error, the system automatically triggers a **graceful fallback** to Groq/Llama-3.3, ensuring zero downtime for the user.

### 2. Self-Correction Loop (The Critic Node)
Instead of delivering the first answer it generates, the agent passes the response to a **Quality Check Node**. If the answer isn't grounded in the PDF context, the graph loops back for a more precise retrieval.

### 3. Loop Protection & State Management
To prevent infinite reasoning loops, the system tracks `loop_count` within the `AgentState` and implements an automated "Safe Exit" after 3 failed attempts, a crucial pattern for production AI.

---

## 🚀 Installation & Usage

### 1. Clone & Setup
```bash
# Clone the repo
git clone [https://github.com/solo-mizan/insight-engine.git](https://github.com/solo-mizan/insight-engine.git)
cd insight-engine

# Sync environment using uv
uv sync