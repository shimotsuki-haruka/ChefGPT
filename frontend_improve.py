import io
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

import streamlit as st
import streamlit.components.v1 as components

from main import RAGSystem  # 使用你的主程序

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="ChefGPT · 知味小厨",
    page_icon="🍳",
    layout="wide",
)

st.title("🍳 知味小厨 ChefGPT")
st.caption("基于 RAG 的智能菜谱问答系统（Streamlit 前端）")

# —— 新增：让所有锚点跳转都是平滑滚动 —— #
st.markdown("<style>html{scroll-behavior:smooth;}</style>", unsafe_allow_html=True)

# =========================
# Cached system init
# =========================
@st.cache_resource(show_spinner=False)
def load_rag_system():
    rag = RAGSystem()
    rag.initial_system()
    rag.build_knowledge_base()
    return rag


def apply_generation_params(rag, temperature: float, max_tokens: int):
    """
    尽力把前端参数同步到后端；不影响主流程
    """
    try:
        if hasattr(rag, "config") and rag.config is not None:
            try:
                rag.config.temperature = temperature
                rag.config.max_tokens = max_tokens
            except Exception:
                pass
        gm = getattr(rag, "generation_module", None)
        if gm is not None:
            for k, v in [("temperature", temperature), ("max_tokens", max_tokens)]:
                try:
                    setattr(gm, k, v)
                except Exception:
                    pass
    except Exception:
        pass


# =========================
# Session State
# =========================
if "rag" not in st.session_state:
    with st.spinner("正在初始化系统并加载知识库..."):
        st.session_state.rag = load_rag_system()
    st.session_state.history = []
    st.session_state.last_debug = ""

rag: RAGSystem = st.session_state.rag

# =========================
# Sidebar Controls
# =========================
with st.sidebar:
    st.header("⚙️ 控制面板")

    default_temp = getattr(getattr(rag, "config", None), "temperature", 0.7)
    default_max_tokens = getattr(getattr(rag, "config", None), "max_tokens", 1024)

    temperature = st.slider("temperature", 0.0, 1.5, float(default_temp), 0.05)
    max_tokens = st.slider("max_tokens", 128, 4096, int(default_max_tokens), 64)
    show_debug = st.checkbox("显示检索/调试日志（抓取后端 print 输出）", value=True)

    colA, colB = st.columns(2)

    with colA:
        if st.button("🧹 清空对话", use_container_width=True):
            st.session_state.history = []
            st.session_state.last_debug = ""
            st.rerun()

    with colB:
        if st.button("♻️ 重建知识库", use_container_width=True):
            # 清掉 cache，重建 rag & 知识库
            load_rag_system.clear()
            with st.spinner("正在重建知识库..."):
                st.session_state.rag = load_rag_system()
            st.success("已重建完成")
            st.rerun()

    st.divider()
    st.write("📌 小提示")
    st.write("- 右侧会展示 `output/kg_last_query.html`（生成成功后显示）")
    st.write("- CMD 中按 **Ctrl+C** 可停止服务")

    # —— 新增：侧栏“页面内跳转”锚点链接（不会打开新页） —— #
    st.markdown("[➡ 跳到对话输入框](#chat-input-anchor)", unsafe_allow_html=True)
    st.markdown("[➡ 跳到知识图谱](#kg-section)", unsafe_allow_html=True)

# =========================
# 工具：生成并渲染交互式 KG HTML（按你的命名：rag.data_module）
# =========================
def generate_kg_html(rag: RAGSystem):
    """
    1) 取 data_module 的 KG（若未构建则 build_kg）
    2) 选取若干菜名（示例取前 8 个）
    3) 调用 write_kg_html_for_dishes() 生成 output/kg_last_query.html
    """
    out_path = Path(__file__).parent / "output" / "kg_last_query.html"

    # 获取 / 构建 KG
    dpm = rag.data_module
    kg = getattr(dpm, "kg", None) or dpm.build_kg()

    # 选取菜名
    dishes = list(kg.get("dishes", {}).keys())
    if not dishes:
        dishes = ["西葫芦炒鸡蛋"]
    dishes = dishes[:8]

    # 生成交互式 HTML 文件
    dpm.write_kg_html_for_dishes(
        dishes=dishes,
        output_file=str(out_path),
        min_degree=2,
        topk_edges=600,
        label_degree=6,
    )

    return out_path

# =========================
# Layout: Left chat / Right panels
# =========================
left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.subheader("💬 对话")

    # 聊天消息容器（按顺序渲染）
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        # 底部锚点，供自动滚动（若需要）
        st.markdown('<div id="chat-bottom"></div>', unsafe_allow_html=True)

    # —— 新增：对话输入框锚点（供侧栏跳转与自动滚动使用） —— #
    st.markdown('<a id="chat-input-anchor"></a>', unsafe_allow_html=True)

    # 固定在底部的唯一输入框
    user_input = st.chat_input("请输入你的问题，例如：宫保鸡丁怎么做？", key="chat_input")

    if user_input:
        # 先把用户消息展示出来
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)
        st.session_state.history.append({"role": "user", "content": user_input})

        # 同步生成参数
        apply_generation_params(rag, temperature=temperature, max_tokens=max_tokens)

        # 捕获后端日志
        stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
        with st.spinner("ChefGPT 思考中..."):
            try:
                with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                    answer = rag.ask_question(
                        user_input, stream=False, history=st.session_state.history
                    )
            except Exception as e:
                answer = None
                stderr_buf.write(f"\n[Frontend捕获异常] {e}\n")

        st.session_state.last_debug = (stdout_buf.getvalue() + ("\n" + stderr_buf.getvalue() if stderr_buf.getvalue().strip() else "")).strip()

        if answer:
            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(answer)
            st.session_state.history.append({"role": "assistant", "content": answer})

        # 标记滚动至底部并刷新
        st.session_state["_scroll_bottom"] = True
        st.rerun()

    # —— 修改：自动滚动到“对话输入框锚点”，并聚焦输入框（平滑滚动） —— #
    if st.session_state.get("_scroll_bottom"):
        st.session_state["_scroll_bottom"] = False
        components.html("""
            <script>
            const doc = window.parent.document;
            const anchor = doc.getElementById("chat-input-anchor");
            if (anchor) {
              anchor.scrollIntoView({ behavior: "smooth", block: "end" });
            }
            setTimeout(() => {
              const t = doc.querySelector('[data-testid=stChatInput] textarea, textarea');
              if (t) t.focus();
            }, 120);
            </script>
        """, height=0, width=0)


with right:
    st.markdown('<div id="kg_sticky" style="position: sticky; top: 12px; z-index: 2;">',
                unsafe_allow_html=True)\
                
    # —— 新增：知识图谱锚点（供侧栏跳转使用） —— #
    st.markdown('<a id="kg-section"></a>', unsafe_allow_html=True)

    st.subheader("🧠 知识图谱 & 运行信息")

    if st.button("🔁 重新生成交互式知识图谱", use_container_width=True):
        try:
            with st.spinner("正在生成交互式知识图谱..."):
                generate_kg_html(rag)   # 只生成，不渲染
            st.rerun()                  # 刷新后由下方统一渲染
        except Exception as e:
            st.error(f"生成失败：{e}")

    right_wrap = st.container()
    with right_wrap:
        html_path = Path(__file__).parent / "output" / "kg_last_query.html"

        if html_path.exists():
            try:
                html_content = html_path.read_text(encoding="utf-8", errors="ignore")

                # 不要传 key（components.html 不支持 key）
                components.html(html_content, height=520, scrolling=False)

                # 仅给下载按钮设置唯一 key，避免重复 ID
                st.download_button(
                    "下载当前知识图谱 HTML",
                    data=html_content.encode("utf-8"),
                    file_name="kg_last_query.html",
                    mime="text/html",
                    use_container_width=True,
                    key="kg_download_main"   # ✅ 唯一 key
                )
            except Exception as e:
                st.warning(f"知识图谱加载失败：{e}")
        else:
            # 首次无文件时可自动生成，或者给出提示
            st.info("尚未检测到知识图谱文件：output/kg_last_query.html（提问后如果后端生成，它会出现在这里）")
            # 如果想自动生成，放开下面两行：
            # with st.spinner("首次生成交互式知识图谱..."):
            #     generate_kg_html(rag); st.rerun()

    with st.expander("🔎 检索/调试日志（后端 print 输出）", expanded=False):
        if st.session_state.last_debug:
            st.code(st.session_state.last_debug, language="text")
        else:
            st.caption("暂无日志：提问一次后会出现。")
            
    # 右侧粘滞样式保持不变
    st.markdown("""
        <style>
        section.main div[data-testid="column"]:nth-of-type(2) > div:first-child {
            position: sticky;
            top: 0;
            height: calc(100vh - 0.75rem);
            overflow: auto;
            background: white;
            border-left: 1px solid #eee;
            padding-left: 12px;
            z-index: 5;
        }
        </style>
    """, unsafe_allow_html=True)


#streamlit run frontend.py
#streamlit run frontend_improve.py
