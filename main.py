"""
RAG项目主程序
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import List

# 添加模块搜索路径
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
from config import RAGConfig, DEFAULT_CONFIG
from rag_modules import (
    DataPreparationModule,
    GenerationIntegrationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
)
# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class RAGSystem:
    """
    RAG主系统
    """

    def __init__(self, config: RAGConfig = None):
        self.config = config or DEFAULT_CONFIG
        # 四个模块
        self.data_module = None
        self.generation_module = None
        self.index_module = None
        self.retrieval_module = None

        # 检查数据路径
        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f"数据路径不存在: {self.config.data_path}")

        # 检查 API
        if not os.getenv("DASHSCOPY_API_KEY"):
            raise ValueError("请设置API!!")

    # =========================
    # 已见菜品持久化小工具（用于跨会话累计知识图谱）
    # =========================
    def _seen_file(self) -> Path:
        return Path(__file__).parent / "output" / "kg_seen_dishes.json"

    def load_seen_dishes(self) -> List[str]:
        p = self._seen_file()
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return []
        return []

    def save_seen_dishes(self, dishes_list: List[str]) -> None:
        p = self._seen_file()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps([d for d in dishes_list if d], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def clear_seen_dishes(self) -> None:
        p = self._seen_file()
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

    # =========================

    def initial_system(self):
        """
        初始化所有模块
        """
        logger.info("正在加载数据模块.....")
        self.data_module = DataPreparationModule(self.config.data_path)

        logger.info("正在加载索引模块.....")
        self.index_module = IndexConstructionModule(
            self.config.embedding_model, self.config.vector_index_path
        )

        logger.info("正在加载集成模块.....")
        self.generation_module = GenerationIntegrationModule(
            self.config.model_name,
            self.config.temperature,
            self.config.max_tokens,
            self.config.history_window_size,
        )
        logger.info("...系统已经加载完成...")

    def build_knowledge_base(self):
        """
        构建索引与准备检索模块
        """
        # 先尝试加载索引
        vectorstore = self.index_module.load_index()

        if vectorstore is not None:
            print("成功加载已保存的向量索引！")

            print("正在加载食谱文档")
            _ = self.data_module.load_documents()

            print("正在划分子文档")
            chunks = self.data_module.chunk_documents()
        else:
            print("未检测到已有的向量索引，重新建立...")

            print("正在加载食谱文档")
            _ = self.data_module.load_documents()

            print("正在划分子文档")
            chunks = self.data_module.chunk_documents()

            print("正在构建索引")
            self.index_module.set_up_embedding()
            vectorstore = self.index_module.build_vector_index(chunks)

            print("正在保存索引")
            self.index_module.save_index()

        print("...正在准备检索优化模块...")
        self.retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)
        logger.info("RetrievalOptimizationModule 初始化完成")

        # 尝试构建知识图谱基础数据（不影响主流程）
        try:
            self.data_module.build_kg()
        except Exception as e:
            logger.warning(f"构建知识图谱失败: {e}")

    def ask_question(self, question: str, stream: bool = False, history: List[dict] = None):
        """
        回答用户问题
        """
        # 检查模块
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")

        print(f"用户问题： {question} \n")

        # 彩蛋
        if question.strip().lower() == "kyy":
            print(
                "\n🌸 亲爱的 KYY 🌸\n"
                "在这个知识系统中，有一个彩蛋只为你而生。\n\n"
                "💖 你是特别的、可爱的、值得被世界温柔以待的人。\n"
                "🍽️ 无论你想吃什么，我都会尽力帮你找到最棒的做法。\n"
                "☀️ 愿你每天都有好心情，好胃口，还有一点点小幸运～\n\n"
                "💌 —— 来自你的专属美食 AI\n"
            )
            return None

        history = history or []

        # 查询路由
        router_type = self.generation_module.query_router(question)
        if router_type == "chitchat":
            print("闲聊类问题，不触发RAG检索\n")
            return self.generation_module.generate_chitchat_answer(question, history)

        # 重写 query
        query = self.generation_module.query_rewrite(question, history)
        print(f"问题已经自动智能重写为:{query}\n")

        # 检索
        relevant_chunks = self.retrieval_module.hybrid_search(query)
        if relevant_chunks:
            print(f"找到了{len(relevant_chunks)}个相关文档\n")
        else:
            print("没有找到相关文档\n")

        for i, doc in enumerate(relevant_chunks):
            dish_name = doc.metadata.get("dish_name", "未知菜品")
            source = doc.metadata.get("source", "未知路径")
            chunk_index = doc.metadata.get("chunk_index", "无")
            chunk_content = doc.page_content.strip().replace("\n", " ")[:200]
            print(f"[{i+1}] 菜名：{dish_name} | chunk_index：{chunk_index} | 来源：{source}")
            print(f"    🔍 内容片段：{chunk_content}")
            print("-" * 80)

        # 找到相关父文档
        relevant_docs = self.data_module.get_parent_document(relevant_chunks)

        # 展示菜名
        doc_names = []
        print(f"\n🔎 找到了 {len(relevant_docs)} 个相关文档，对应菜品如下：\n")
        for i, doc in enumerate(relevant_docs):
            dish_name = doc.metadata.get("dish_name", "未知菜品")
            source = doc.metadata.get("source", "未知路径")
            chunk_index = doc.metadata.get("chunk_index", "无")
            doc_names.append(dish_name)
            print(f"[{i+1}] 菜名：{dish_name} | chunk_index：{chunk_index} | 来源：{source}")

        print("\n📋 菜品汇总：")
        print("，".join(doc_names))

        # === 用“历史并集”更新知识图谱（关键改动） ===
        try:
            # 1) 读取历史已见菜品
            seen = self.load_seen_dishes()
            seen = [d for d in seen if d]

            # 2) 把本次菜品并入历史（去重并保留顺序，最近的在末尾）
            for d in doc_names:
                if d in seen:
                    seen.remove(d)
                seen.append(d)

            # 3) 限制累计长度（可按需调整）
            MAX_DISHES = 30
            if len(seen) > MAX_DISHES:
                seen = seen[-MAX_DISHES:]

            # 4) 保存并渲染到 HTML
            self.save_seen_dishes(seen)
            out_path = Path(__file__).parent / "output" / "kg_last_query.html"
            self.data_module.write_kg_html_for_dishes(seen, str(out_path))
            print(f"\n🧠 已更新知识图谱（累计 {len(seen)} 道菜）: {out_path}")
        except Exception as e:
            print(f"更新知识图谱失败: {e}")

        # 生成回答
        if router_type == "list":
            return self.generation_module.generate_list_answer(query, relevant_docs)
        elif router_type == "detail":
            return self.generation_module.generate_detail_answer(query, relevant_docs, history)
        else:
            return self.generation_module.generate_general_answer(query, relevant_docs, history)

    def run_interactive(self):
        """
        构建一个交互式系统（命令行）
        支持文字提问和图片识别模式：
        - 输入 'img 图片路径' 可先识别食物类别，再自动生成相关菜谱问题
        - 输入 '退出' / 'exit' / 'quit' 结束对话
        """
        print("=" * 60)
        print("🍳  欢迎来到 知味小厨 （ChefGPT） 🍳".center(60))
        print("=" * 60)
        print("🤖 基于 RAG 技术的智能菜谱问答系统，助您轻松掌厨！".center(60))
        print("=" * 60)

        # 初始化系统与知识库
        self.initial_system()
        print("系统初始化完成\n")
        self.build_knowledge_base()

        history = []

        print("\n交互式问答 (输入'退出'结束):")
        print("💬 直接输入文字提问")
        print("📷 输入 'img 图片路径' 使用图片识别提问（例如: img pizza.jpg）")

        while True:
            try:
                user_input = input("\n您的问题: ").strip()
                if user_input.lower() in ["退出", "exit", "quit"]:
                    print("👋 感谢使用 ChefGPT，再见！")
                    break

                # ========== 图片识别模式（只识别，不自动提问） ==========
                if user_input == "img" or user_input.startswith("img "):
                    img_path = user_input[4:].strip()
                    if not img_path:
                        print("❌ 请提供图片路径，例如: img pizza.jpg")
                        continue
                    try:
                        from food_recognizer import recognize_food, make_food_question
                        food_label = recognize_food(img_path)
                        print(f"📸 识别结果：{food_label}")
                        # 不再自动构造问题，改为让用户自由输入
                        print("💡 您可以基于这个结果向我提问，例如：")
                        print(f'   "最简单的{food_label}做法"')
                        print(f'   "低卡的{food_label}"')
                        custom_question = input("👉 请输入您的问题（直接回车使用默认推荐问题）: ").strip()
                        if not custom_question:
                            # 默认提问
                            user_question = make_food_question(food_label)
                        else:
                            # 用户自定义提问，可以不带标签，也可以拼接标签
                            user_question = custom_question
                    except FileNotFoundError:
                        print(f"❌ 找不到图片文件: {img_path}")
                        continue
                    except Exception as e:
                        print(f"❌ 图片识别失败: {e}")
                        continue
                else:
                    # 普通文字提问
                    user_question = user_input
                # ===================================

                # 调用核心问答
                answer = self.ask_question(user_question, stream=False, history=history)
                if answer is None:
                    continue
                history.append({"role": "user", "content": user_question})
                history.append({"role": "assistant", "content": answer})
                print(f"{answer}\n")

            except KeyboardInterrupt:
                print("\n👋 检测到退出指令，感谢使用 ChefGPT！")
                break
            except Exception as e:
                print(f"处理问题时出错: {e}\n")


def main():
    """
    主函数入口
    """
    try:
        rag = RAGSystem()
        rag.run_interactive()
    except Exception as e:
        logger.error(f"系统运行出错: {e}")
        print(f"处理问题时出错: {e}\n")


if __name__ == "__main__":
    main()


'''def write_kg_html_for_dishes(self, dishes: List[str], output_file: str):
        import json
        from pathlib import Path
        if not self.kg.get("dishes"):
            self.build_kg()
        nodes = []
        links = []
        seen = set()
        for dish in dishes:
            info = self.kg.get("dishes", {}).get(dish, {})
            dish_id = f"dish:{dish}"
            if dish_id not in seen:
                nodes.append({"id": dish_id, "label": dish, "type": "dish"})
                seen.add(dish_id)
            category = info.get("category") or "其他"
            difficulty = info.get("difficulty") or "未知难度"
            cat_id = f"cat:{category}"
            dif_id = f"dif:{difficulty}"
            if cat_id not in seen:
                nodes.append({"id": cat_id, "label": category, "type": "category"})
                seen.add(cat_id)
            if dif_id not in seen:
                nodes.append({"id": dif_id, "label": difficulty, "type": "difficulty"})
                seen.add(dif_id)
            links.append({"source": dish_id, "target": cat_id})
            links.append({"source": dish_id, "target": dif_id})
            for ing in info.get("ingredients", []):
                ing_id = f"ing:{ing}"
                if ing_id not in seen:
                    nodes.append({"id": ing_id, "label": ing, "type": "ingredient"})
                    seen.add(ing_id)
                links.append({"source": dish_id, "target": ing_id})
            for tech in info.get("techniques", []):
                tech_id = f"tech:{tech}"
                if tech_id not in seen:
                    nodes.append({"id": tech_id, "label": tech, "type": "technique"})
                    seen.add(tech_id)
                links.append({"source": dish_id, "target": tech_id})
        data_json = json.dumps({"nodes": nodes, "links": links}, ensure_ascii=False)
        html = (
            "<!DOCTYPE html>\n"
            "<html lang=\"zh\">\n"
            "<meta charset=\"utf-8\">\n"
            "<title>ChefGPT KG Subgraph</title>\n"
            "<style>\n  body { font-family: system-ui, sans-serif; margin: 0; }\n  #chart { width: 100vw; height: 100vh; background: #fff; }\n  .link { stroke: #999; stroke-opacity: 0.6; }\n  .label { font-size: 12px; pointer-events: none; }\n</style>\n"
            "<svg id=\"chart\"></svg>\n"
            "<script>\n"
            f"const data = {data_json};\n"
            "const svg = document.getElementById('chart');\n"
            "const width = (window.innerWidth || document.documentElement.clientWidth || document.body.clientWidth);\n"
            "const height = (window.innerHeight || document.documentElement.clientHeight || document.body.clientHeight);\n"
            "svg.setAttribute('width', width); svg.setAttribute('height', height);\n"
            "const colors = {dish:'#1f77b4',ingredient:'#2ca02c',technique:'#ff7f0e',category:'#9467bd',difficulty:'#8c564b'};\n"
            "const nodes = data.nodes; const links = data.links || [];\n"
            "const types = ['dish','ingredient','technique','category','difficulty'];\n"
            "const radius = Math.min(width, height) / 2 - 60;\n"
            "const center = {x: width/2, y: height/2};\n"
            "const typeR = {dish: radius*0.2, ingredient: radius*0.5, technique: radius*0.7, category: radius*0.9, difficulty: radius*0.95};\n"
            "const grouped = {dish:[], ingredient:[], technique:[], category:[], difficulty:[]};\n"
            "nodes.forEach(n => { grouped[n.type].push(n); });\n"
            "function polar(cx, cy, r, idx, total){ const a = (2*Math.PI)*(idx/Math.max(1,total)); return {x: cx + r*Math.cos(a), y: cy + r*Math.sin(a)}; }\n"
            "const pos = {};\n"
            "types.forEach(t => { const arr = grouped[t]; const r = typeR[t] || radius*0.8; arr.forEach((n,i)=>{ const p = polar(center.x, center.y, r, i, arr.length); pos[n.id] = p; }); });\n"
            "function drawLine(x1,y1,x2,y2){ const l = document.createElementNS('http://www.w3.org/2000/svg','line'); l.setAttribute('x1',x1); l.setAttribute('y1',y1); l.setAttribute('x2',x2); l.setAttribute('y2',y2); l.setAttribute('class','link'); l.setAttribute('stroke','#999'); l.setAttribute('stroke-width','1'); svg.appendChild(l); }\n"
            "function drawNode(x,y,color,label){ const g = document.createElementNS('http://www.w3.org/2000/svg','g'); const c = document.createElementNS('http://www.w3.org/2000/svg','circle'); c.setAttribute('cx',x); c.setAttribute('cy',y); c.setAttribute('r',10); c.setAttribute('fill',color||'#7f7f7f'); c.setAttribute('stroke','#333'); c.setAttribute('stroke-width','1'); const text = document.createElementNS('http://www.w3.org/2000/svg','text'); text.setAttribute('x', x+12); text.setAttribute('y', y+4); text.setAttribute('class','label'); text.textContent = label; g.appendChild(c); g.appendChild(text); svg.appendChild(g); }\n"
            "links.forEach(e => { const s = pos[e.source], t = pos[e.target]; if(s && t){ drawLine(s.x,s.y,t.x,t.y); }});\n"
            "nodes.forEach(n => { const p = pos[n.id]; const color = colors[n.type] || '#7f7f7f'; if(p){ drawNode(p.x, p.y, color, n.label); } });\n"
            "</script>\n"
            "</html>\n"
        )
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(output_file).write_text(html, encoding="utf-8")'''
