import streamlit as st
from main import RAGSystem

# =========================
# 页面基础设置
# =========================
st.set_page_config(
    page_title="ChefGPT · 知味小厨",
    page_icon="🍳",
    layout="wide"
)

st.title("🍳 知味小厨 ChefGPT")
st.caption("基于 RAG 的智能菜谱问答系统")

# =========================
# Session State 初始化
# =========================
if "rag" not in st.session_state:
    with st.spinner("正在初始化系统，请稍候..."):
        rag = RAGSystem()
        rag.initial_system()
        rag.build_knowledge_base()
        st.session_state.rag = rag
        st.session_state.history = []

st.success("系统已就绪，可以开始提问！")

# =========================
# 聊天记录展示
# =========================
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================
# 用户输入
# =========================
user_input = st.chat_input("请输入你想问的菜谱问题，例如：红烧肉怎么做？")

if user_input:
    # 显示用户输入
    with st.chat_message("user"):
        st.markdown(user_input)

    # 调用 RAG 后端
    with st.spinner("ChefGPT 思考中..."):
        answer = st.session_state.rag.ask_question(
            user_input,
            history=st.session_state.history
        )

    if answer:
        with st.chat_message("assistant"):
            st.markdown(answer)

        # 更新历史
        st.session_state.history.append(
            {"role": "user", "content": user_input}
        )
        st.session_state.history.append(
            {"role": "assistant", "content": answer}
        )
