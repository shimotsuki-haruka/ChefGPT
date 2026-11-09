import streamlit as st
from main import RAGSystem
from food_recognizer import recognize_food
from food_recognizer import make_food_question


# =========================
# 页面基础设置
# =========================
st.set_page_config(
    page_title="ChefGPT · 知味小厨",
    page_icon="🍳",
    layout="wide"
)

st.title("🍳 知味小厨 ChefGPT")
st.caption("基于 RAG 的智能菜谱问答系统 · 支持拍照识菜自动推荐")

# =========================
# Session State 初始化
# =========================
if "rag" not in st.session_state:
    with st.spinner("正在初始化系统，请稍候..."):
        rag = RAGSystem()
        rag.initial_system()
        rag.build_knowledge_base()
        st.session_state.rag = rag
        st.session_state.history = []   # 对话历史
        st.session_state.detected_food = None   # 暂存最近识别结果
    st.success("系统就绪！您可以文字提问，或从侧边栏📷上传食物照片自动提问。")

# =========================
# 侧边栏：独立图片识别 → 自动提问
# =========================
with st.sidebar:
    st.header("📷 拍照识菜 · 自动提问")
    uploaded_file = st.file_uploader("上传食物照片", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="待识别图片", width='stretch')

        if st.button("🔍 开始识别并自动提问"):
            # 保存临时文件
            uploaded_file.seek(0)

            with st.spinner("AI 识别中，即将自动为您推荐菜谱..."):
                try:
                    label = recognize_food(uploaded_file)
                    st.session_state.detected_food = label
                    # 构造默认提问
                    auto_question = make_food_question(label)

                    # 直接调用 RAG 系统获得回答
                    answer = st.session_state.rag.ask_question(
                        auto_question,
                        history=st.session_state.history
                    )

                    # 将问答对加入历史
                    if answer is not None:
                        st.session_state.history.append({"role": "user", "content": auto_question})
                        st.session_state.history.append({"role": "assistant", "content": answer})

                    # 重新加载页面，显示最新对话
                    st.rerun()

                except Exception as e:
                    st.error(f"识别或提问失败：{e}")

    # 展示最近识别结果（如果存在）
    if st.session_state.detected_food:
        st.info(f"📌 最近识别：**{st.session_state.detected_food}**")

# =========================
# 主聊天界面（显示历史记录 + 自由提问）
# =========================
# 展示已有的对话历史
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 手动输入区域
user_input = st.chat_input("请输入你想问的菜谱问题（例如：红烧肉怎么做？）")

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
        st.session_state.history.append({"role": "user", "content": user_input})
        st.session_state.history.append({"role": "assistant", "content": answer})