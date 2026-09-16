import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from api_client import ask_question, ApiError

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

EXAMPLE_QUESTIONS = [
    "What is overfitting?",
    "How does gradient descent work?",
    "What is transfer learning?",
    "Bias-variance tradeoff?",
]

st.set_page_config(
    page_title="RAG Document Assistant",
    layout="centered",
)

# =============================================================================
# Custom iMessage-style CSS
# -----------------------------------------------------------------------------
# Chat bubbles are rendered as plain HTML (not st.chat_message / not the
# streamlit-chat package) because both of those box us into fixed styling
# with no CSS hooks for asymmetric bubble tails, hover states, or animation.
# Plain HTML gives full control. st.chat_input is still used for the actual
# text box since building a real text box by hand is not worth it.
# =============================================================================
st.markdown(
    """
    <style>
        html, body {
            background-color: #F2F2F7 !important;
        }
        .stApp {
            background-color: #F2F2F7;
        }
        .stApp, .stApp p, .stApp span, .stApp label, .stApp div {
            color: #000000;
            font-family: -apple-system, BlinkMacSystemFont, "San Francisco",
                "Segoe UI", system-ui, sans-serif;
        }
        .block-container {
            max-width: 700px;
            padding-top: 1rem;
            padding-bottom: 8rem;
        }
        h1, h1 span, .stMarkdown h1 {
            color: #000000 !important;
        }
        .app-subtitle {
            color: #6b7280 !important;
            font-size: 0.95rem;
            margin-top: -0.5rem;
            margin-bottom: 1.25rem;
        }

        /* --- Top header bar: blend into the page instead of a dark strip --- */
        [data-testid="stHeader"] {
            background-color: #F2F2F7 !important;
            height: 2.5rem;
        }
        [data-testid="stHeader"] * {
            color: #000000 !important;
        }

        /* --- "Clear conversation" text link (marker-div + :has(), works
           across Streamlit versions since it doesn't rely on container keys
           and doesn't assume the marker is a *direct* sibling of the button
           -- Streamlit nests it inside stMarkdownContainer). --- */
        :where(div[data-testid="element-container"], div[data-testid="stElementContainer"]):has(div.marker-clear-btn)
            + :where(div[data-testid="element-container"], div[data-testid="stElementContainer"]) button {
            background-color: transparent !important;
            border: none !important;
            color: #007AFF !important;
            font-size: 0.85rem !important;
            padding: 0 !important;
            box-shadow: none !important;
        }
        :where(div[data-testid="element-container"], div[data-testid="stElementContainer"]):has(div.marker-clear-btn)
            + :where(div[data-testid="element-container"], div[data-testid="stElementContainer"]) button:hover {
            text-decoration: underline;
        }

        /* --- Chat history --- */
        .chat-scroll {
            display: flex;
            flex-direction: column;
            padding-bottom: 0.5rem;
        }
        .msg-row {
            display: flex;
            width: 100%;
            margin-top: 10px;
        }
        .msg-row.sent { justify-content: flex-end; }
        .msg-row.received { justify-content: flex-start; }
        .msg-row.tight { margin-top: 2px; }

        .bubble-col {
            display: flex;
            flex-direction: column;
            max-width: 75%;
        }
        .bubble-col.sent { align-items: flex-end; }
        .bubble-col.received { align-items: flex-start; }

        .bubble {
            padding: 10px 14px;
            font-size: 0.95rem;
            line-height: 1.35;
            border-radius: 20px;
            white-space: pre-wrap;
            word-wrap: break-word;
            animation: bubbleIn 0.22s ease-out;
        }
        .bubble.sent {
            background-color: #007AFF;
            color: #FFFFFF;
            border-bottom-right-radius: 4px;
        }
        .bubble.received {
            background-color: #E9E9EB;
            color: #000000;
            border-bottom-left-radius: 4px;
        }

        @keyframes bubbleIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .msg-timestamp {
            font-size: 0.68rem;
            color: #9CA3AF;
            margin-top: 3px;
            padding: 0 4px;
        }

        .source-pill {
            display: inline-block;
            background-color: rgba(0, 122, 255, 0.10);
            color: #007AFF;
            border-radius: 999px;
            padding: 2px 10px;
            font-size: 0.72rem;
            margin: 6px 4px 0 0;
        }
        .sources-row {
            display: flex;
            flex-wrap: wrap;
            justify-content: flex-end;
        }

        /* --- Typing indicator (three bouncing dots inside a received bubble) --- */
        .typing-dots {
            display: flex;
            align-items: center;
            gap: 4px;
            padding: 4px 2px;
        }
        .typing-dots span {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background-color: #9CA3AF;
            animation: typingBounce 1.1s infinite ease-in-out;
        }
        .typing-dots span:nth-child(2) { animation-delay: 0.15s; }
        .typing-dots span:nth-child(3) { animation-delay: 0.3s; }
        @keyframes typingBounce {
            0%, 60%, 100% { transform: translateY(0); opacity: 0.5; }
            30% { transform: translateY(-4px); opacity: 1; }
        }

        /* --- Example question bubbles, sitting just above the input bar --- */
        :where(div[data-testid="element-container"], div[data-testid="stElementContainer"]):has(div.marker-example-btn)
            + :where(div[data-testid="element-container"], div[data-testid="stElementContainer"]) button {
            border-radius: 999px !important;
            border: 1px solid #E5E5EA !important;
            background-color: #FFFFFF !important;
            color: #007AFF !important;
            font-size: 0.82rem !important;
            font-weight: 500 !important;
            padding: 6px 8px !important;
            transition: transform 0.15s ease, background-color 0.15s ease,
                color 0.15s ease, border-color 0.15s ease !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        }
        :where(div[data-testid="element-container"], div[data-testid="stElementContainer"]):has(div.marker-example-btn)
            + :where(div[data-testid="element-container"], div[data-testid="stElementContainer"]) button:hover {
            background-color: #007AFF !important;
            color: #FFFFFF !important;
            border-color: #007AFF !important;
            transform: scale(1.06);
        }
        :where(div[data-testid="element-container"], div[data-testid="stElementContainer"]):has(div.marker-example-btn)
            + :where(div[data-testid="element-container"], div[data-testid="stElementContainer"]) button:active {
            transform: scale(0.98);
        }

        /* --- Input bar (st.chat_input) styled toward the iOS pill --- */
        [data-testid="stBottomBlockContainer"],
        [data-testid="stBottom"],
        div[data-testid^="stBottom"],
        [data-testid*="ottom" i] {
            background-color: #F2F2F7 !important;
        }
        [data-testid="stChatInput"] {
            background-color: #F2F2F7;
            border-top: none;
            box-shadow: none;
        }
        [data-testid="stChatInput"] textarea {
            border-radius: 999px !important;
            border: 1px solid #C6C6C8 !important;
            background-color: #FFFFFF !important;
            padding-left: 16px !important;
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
            box-shadow: none !important;
            outline: none !important;
        }
        /* Streamlit's default focus state draws a red/pink ring here --
           override it so focusing the box shows iOS blue instead. */
        [data-testid="stChatInput"] textarea:focus,
        [data-testid="stChatInput"] textarea:focus-visible {
            border: 1px solid #007AFF !important;
            box-shadow: 0 0 0 1px #007AFF !important;
            outline: none !important;
        }
        [data-testid="stChatInput"]:focus-within {
            border: none !important;
            box-shadow: none !important;
            outline: none !important;
        }
        [data-testid="stChatInput"] textarea::placeholder {
            color: #8E8E93 !important;
            -webkit-text-fill-color: #8E8E93 !important;
            opacity: 1 !important;
        }
        [data-testid="stChatInput"] button {
            background-color: #007AFF !important;
            border-radius: 50% !important;
            border: none !important;
        }
        [data-testid="stChatInput"] button svg {
            fill: #FFFFFF !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("RAG Document Assistant")
st.markdown(
    '<p class="app-subtitle">Ask a question about machine learning concepts — '
    "answers are grounded in the document collection, with sources cited.</p>",
    unsafe_allow_html=True,
)

st.markdown('<div class="marker-clear-btn"></div>', unsafe_allow_html=True)
if st.button("Clear conversation"):
    st.session_state.messages = []
    st.session_state.pending_question = None
    st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []  # {"role", "content", "sources", "time"}
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


def queue_question(question: str) -> None:
    """Append the user's bubble immediately and mark a question pending, so
    the typing indicator renders on the next script pass *before* the
    (blocking) API call happens -- otherwise it never gets a chance to paint."""
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
            "sources": [],
            "time": datetime.now().strftime("%H:%M"),
        }
    )
    st.session_state.pending_question = question


def render_bubble(role: str, content: str, sources: list | None, time_str: str, tight: bool = False) -> None:
    side = "sent" if role == "user" else "received"
    row_classes = f"msg-row {side}" + (" tight" if tight else "")
    safe_content = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    pills_html = ""
    if sources:
        pills = "".join(f'<span class="source-pill">{s}</span>' for s in sources)
        pills_html = f'<div class="sources-row">{pills}</div>'

    st.markdown(
        f"""
        <div class="{row_classes}">
            <div class="bubble-col {side}">
                <div class="bubble {side}">{safe_content}</div>
                {pills_html}
                <div class="msg-timestamp">{time_str}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- Render conversation history ---
st.markdown('<div class="chat-scroll">', unsafe_allow_html=True)
prev_role = None
for msg in st.session_state.messages:
    render_bubble(
        msg["role"],
        msg["content"],
        msg.get("sources"),
        msg.get("time", ""),
        tight=(msg["role"] == prev_role),
    )
    prev_role = msg["role"]

# --- If a question is pending, show the typing indicator now, then make the
# blocking call and rerun once more so the real answer replaces it. ---
if st.session_state.pending_question:
    st.markdown(
        """
        <div class="msg-row received">
            <div class="bubble-col received">
                <div class="bubble received">
                    <div class="typing-dots"><span></span><span></span><span></span></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.pending_question:
    question = st.session_state.pending_question
    try:
        answer, sources = ask_question(question, API_BASE_URL)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "time": datetime.now().strftime("%H:%M"),
            }
        )
    except ApiError as e:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"Couldn't reach the backend: {e}",
                "sources": [],
                "time": datetime.now().strftime("%H:%M"),
            }
        )
    st.session_state.pending_question = None
    st.rerun()

# --- Example question bubbles, sitting just above the input bar ---
cols = st.columns(len(EXAMPLE_QUESTIONS))
for col, eq in zip(cols, EXAMPLE_QUESTIONS):
    with col:
        st.markdown('<div class="marker-example-btn"></div>', unsafe_allow_html=True)
        if st.button(eq, use_container_width=True, key=f"eg_{eq}"):
            queue_question(eq)
            st.rerun()

typed = st.chat_input("Ask a question about ML concepts...")
if typed:
    queue_question(typed)
    st.rerun()