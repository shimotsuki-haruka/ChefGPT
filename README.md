## 🧾 项目简介｜Project Overview
📌 中文简介：知味小厨（ChefGPT）

知味小厨（ChefGPT）是一款基于 RAG（Retrieval-Augmented Generation，检索增强生成）技术构建的智能菜谱问答系统。用户可以用自然语言提问，系统会从结构化与非结构化的菜谱文档中精准检索并生成答案，实现对话式、上下文感知的美食知识问答体验。

项目目标是打造一个既能回答烹饪问题、又能理解用户语境的智能厨师助手，未来可拓展至个性化菜谱推荐、营养分析、多语言支持等功能。

🌟 功能亮点：

✅ 菜谱问答：支持自由提问菜品名称、食材用量、做法步骤、技巧说明等

✅ 多轮对话支持：自动追踪对话上下文，实现连续交流(已实现)

✅ API 接入友好：支持私有部署，适配 OpenAI / 自托管模型(自托管模型接口待实现)

✅ 文档增强式检索：支持 Markdown 菜谱文档解析、结构化增强

## 项目参考 & 致谢：

https://github.com/Anduin2017/HowToCook     


## 📁 项目结构 | Project Structure

```bash
/ChefGPT
├── config.py                   # 配置管理脚本（如 API_KEY、模型参数等）
├── main.py                     # 主程序入口，执行整体问答流程
├── data/                       # 原始/处理后菜谱数据存放目录
├── requirements.txt            # Python依赖库列表
├── rag_modules/                # 核心功能模块目录
│   ├── __init__.py
│   ├── data_preparation.py     # 数据准备模块：读取、切分、结构化文档
│   ├── index_construction.py   # 向量索引构建模块：构建嵌入并缓存索引
│   ├── retrieval_optimization.py  # 检索优化模块：改进向量/关键词混合检索
│   └── generation_integration.py  # 生成集成模块：调用大模型进行答案生成
├── vector_index/               # 存放已构建的向量索引缓存（自动生成）
└── README.md                   # 项目说明文档（即本文件）
```
## 🚀 快速开始 | Getting Started

本节将指导你如何快速启动 ChefGPT 项目，从环境配置到运行问答主程序。

### 1️⃣ 克隆项目并进入目录

```bash
git clone https://github.com/Meteor426/ChefGPT.git
cd ChefGPT
```

### 2️⃣ 创建虚拟环境（推荐使用 Conda）

```bash
conda create -n chefgpt python=3.10 -y
conda activate chefgpt
```

### 3️⃣ 安装依赖项
```bash
pip install -r requirements.txt
```

### 4️⃣ 配置 API 密钥与地址
确保你已拥有 OpenAI / 第三方兼容模型的 API Key 和 Base URL，并设置为系统环境变量

### 5️⃣ 运行主程序
```bash
python main.py
```
首次运行将执行以下流程：

加载/预处理菜谱文档数据；

构建并缓存向量索引；

启动问答引擎，支持对菜谱内容进行多轮问答。


### ! 参数修改
如果你想修改模型参数，如embedding_model，llm_model，temperature等，请修改config.py