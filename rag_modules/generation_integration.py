'''
生产集成模块
'''
import logging
import os 
from langchain_openai import ChatOpenAI
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from openai import api_key

#from langchain_core.runnables import RunnablePassthrough
logger = logging.getLogger(__name__)

class GenerationIntegrationModule:
    '''
    集成LLM与回答生成
    '''
    def __init__(self,model_name:str,temperature:float = 0,max_tokens:int = 2048,history_window_size:int = 8):
        '''
        初始化
        '''
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.history_window_size = 8
        self.llm = None
        self._setup_llm()


    def _setup_llm(self):
        '''
        加载llm
        '''
        logger.info(f"正在加载LLM : {self.model_name}")
        api_key = os.getenv("DASHSCOPY_API_KEY")

        if not api_key:
            raise ValueError("请先设置API_KEY")
        self.llm = ChatOpenAI(
            model = "deepseek-v3.2-exp",
            temperature = self.temperature,
            api_key = api_key,
            streaming = True,   #允许流式输出
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1",
            max_tokens = self.max_tokens
        )
        logger.info(f"LLM加载完成")
    
    def generate_chitchat_answer(self,query:str,context:List[Document],history = None)->str:
        '''
        闲聊类问题（不需要检索）直接用 LLM 回答
        '''
        messages = [SystemMessage(content="你是一个亲切的美食助手，可以友好地与用户闲聊。")]
        if history:
            for msg in history[-self.history_window_size:]:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=query))

        response = self.llm.invoke(messages)
        return response.content
    
    def _build_context(self,context:List[Document],max_length = 2000)->str:
        '''
        构建上下文，负责把文档列表拼成一个整体字符串
        Args:
            context: 上下文文档集合
            max_length: 最大长度
        Returns:
            格式化后的上下文字符串
        '''
        if not context:
            return "暂无上下文信息"
        #初始化一个容器，存放处理后的上下文，同时记录长度
        context_parts = []
        current_length = 0
        for i,doc in enumerate(context,1):  #从1开始计数

            #首先取出元数据
            metadata_info = f"食谱{i}"
            if 'dish_name' in doc.metadata:
                metadata_info += f"{doc.metadata['dish_name']}"
            if 'category' in doc.metadata:
                metadata_info += f"| 分类：{doc.metadata['category']}"
            if 'difficulty' in doc.metadata:
                metadata_info += f"| 难易程度：{doc.metadata['difficulty']}"

            #构建文本，拼接起来
            doc_context = f"{metadata_info} \n {doc.page_content}"

            #检查长度限制
            if current_length + len(doc_context) > max_length:
                break

            context_parts.append(doc_context)
            current_length += len(doc_context)
        return "\n" + '='*50 + '\n'+ '\n'.join(context_parts)  #换行后，每一行放一条文档数据

    def query_rewrite(self,query:str,history = None)->str:
        '''
        由大模型来判断是否需要重写质量不高的query
        '''
        prompt = PromptTemplate.from_template(
        '''
        你是一个智能查询重写助手，负责将用户输入的查询在必要时重写为更适合菜谱检索的表达。

        === 当前对话历史（用于判断上下文指代） ===
        {history}

        === 当前用户查询 ===
        {query}

        === 判断与重写规则 ===

        1. ✅ **具体明确的查询（不重写）**
        - 包含具体菜品名称：如“宫保鸡丁怎么做”、“红烧肉的做法”
        - 明确步骤/技巧提问：如“糖醋排骨用什么调料”、“如何炒菜不粘锅”

        2. ❌ **模糊或宽泛的查询（应重写）**
        - 不含菜名：如“做菜”、“推荐个菜”、“简单的”
        - 无具体目标：如“想吃点什么”、“来点素的”

        3. 🔁 **指代型查询（需上下文重写）**
        - 包含“这个”、“它”、“第一个”等指代词
        - 如果历史中出现了推荐菜品列表（如“推荐了水煮鱼、红烧肉”），则用对应菜名替换指代词
        - 若历史为空或菜名无法判断，保留原查询不做修改

        4) 💡 否定或排除型查询（必须改写为【正向表达】；禁止输出任何否定词）
        - 将否定转为正向偏好，不得出现：不、别、不要、非、无、排除、避免、不是、without、exclude、-词 等否定/排除用语
        - 参考映射（可组合）：
        • 不想吃/不要“凉拌/冷的” → “家常热菜”“爆炒/清蒸/煎/烤/炖”
        • 不辣/少辣/清淡 → “清淡口味/原味/微辣”“清蒸/清炒/水煮”
        • 不要汤/干一点 → “干炒/煎/烤/焖”
        • 不油/少油/低脂 → “低脂清淡”“清蒸/空气炸/清炒”
        • 不甜 → “咸鲜口味/蒜香/胡椒香”
        - 如含食材主题（来自当前或历史，如“鸡胸肉/鱼/牛肉”），优先生成“食材+正向做法/口味+推荐/做法”

        === 重写原则 ===
        - 增强语义清晰度，方便菜谱系统理解
        - 保留用户原意，不引入不相关信息
        - 优先推荐家常菜、易做菜，风格清晰

        === 示例 ===
        - “做菜” → “简单易做的家常菜谱”
        - “川菜” → “经典川菜菜谱”
        - “宫保鸡丁怎么做” → “宫保鸡丁怎么做”
        - “第一个怎么做” + 上文提到“水煮鱼、红烧肉” → “水煮鱼怎么做”
        - “它需要什么调料” + 上文提到“推荐了麻辣香锅” → “麻辣香锅需要什么调料”
        - “不想吃凉拌” → “推荐一些爆炒的菜”
        - “不要太油腻的” → “推荐一些清淡的家常菜”
        - 上文：推荐了“水煮鱼、红烧肉”；“第一个怎么做” → “水煮鱼怎么做”
        - 上文：鸡胸肉；“不想吃凉拌的” → “推荐一些鸡胸肉的爆炒做法”

        === 输出要求 ===
        请输出最终用于菜谱检索的查询内容（如无需改写则原样返回）：
        '''
        )
        #构建链
        chain = (prompt | self.llm | StrOutputParser())
        history = self._format_history(history)
        response = chain.invoke({'query':query,'history':history}).strip()  #得到输出并移除首尾的空格

        if response != query:
            logger.info(f"查询已改写：'{query}' -> '{response}'")
        else:
            logger.info(f"查询未改写: '{query}")

        return response

    def _format_history(self,history: List[dict]) -> str:
        '''
        将对话历史格式化为字符串
        '''
        if not history:
            return "（无）"
        formatted = ""
        for msg in history[-4:]:  # 最多保留最近 4 条
            role = "用户" if msg["role"] == "user" else "ChefGPT"
            formatted += f"{role}：{msg['content']}\n"
        return formatted.strip()

    def query_router(self,query:str)->str:
        '''
        实现多查询功能-查询路由
        根据query选择不同的处理方式
        Args:
            query:查询
        Returns:
            路由类型：('list', 'detail', 'general','chitchat)
        '''
        prompt = ChatPromptTemplate.from_template(
            '''
            根据用户的问题，将其分类为以下三种类型之一：

        1. 'list' - 用户想要获取菜品列表或推荐，只需要菜名
        例如：推荐几个素菜、有什么川菜、给我3个简单的菜

        2. 'detail' - 用户想要具体的制作方法或详细信息
        例如：宫保鸡丁怎么做、制作步骤、需要什么食材

        3. 'general' - 其他一般性知识问题
        例如：什么是川菜、制作技巧、营养价值

        4. 'chitchat' - 与菜谱无关的问题，闲聊问候类问题，（如“你好”“谢谢”“你是谁”）

        请只返回分类结果：list、detail、 general或者chitchat

        用户问题: {query}

        分类结果:'''
        )

        chain = (prompt | self.llm | StrOutputParser())

        response = chain.invoke({'query':query}).strip().lower()

        #检查分类有效性
        if response in ['list', 'detail', 'general','chitchat']:
            return response
        else:
            return 'general'
        
    def generate_list_answer(self,query:str,context:List[Document])->str:
        '''
        list 类型的问题回答器-适合推荐类查询，推荐一系列菜品
        Args:
            query:提问
            context:上下文
        Returns:
            菜品回答
        '''
        if not context:
            return "很抱歉，没有找到相关的菜谱"

        dishes = []
        for doc in context:
            dish_name = doc.metadata.get("dish_name","未知菜品")
            #去重
            if dish_name not in dishes:
                dishes.append(dish_name)
        
        #构造回答
        if len(dishes) == 1:
            return f"为你推荐此菜品：{dishes[0]} \n"
        elif len(dishes) <= 3:
            return f"为你推荐以下菜品 : \n" + "\n".join( [f"{i},{name}" for i,name in enumerate(dishes)] )
        else:
            return f"为你推荐以下菜品 : \n" + "\n".join( [f"{i},{name}" for i,name in enumerate(dishes[:3])] ) + f"\n\n 还有其他{len(dishes) - 3} 道菜品可供选择"
                
    def generate_detail_answer(self,query:str,context:List[Document],history:List[dict] = None):
        '''
        detail类型的回答器
        Args:
            query:查询
            context:上下文
            history:对话历史
        Returns:
            LLM回答
        '''
        #拼接上下文
        context = self._build_context(context) 
        
        system_prompt = '''你是一位专业的烹饪导师。请根据食谱信息，为用户提供详细的分步骤指导。

        用户问题: {question}

        相关食谱信息:
        {context}

        请灵活组织回答，建议包含以下部分（可根据实际内容调整）：

        ## 🥘 菜品介绍
        [简要介绍菜品特点和难度]

        ## 🛒 所需食材
        [列出主要食材和用量]

        ## 👨‍🍳 制作步骤
        [详细的分步骤说明，每步包含具体操作和大概所需时间]

        ## 💡 制作技巧
        [仅在有实用技巧时包含。如果原文的"附加内容"与烹饪无关或为空，可以基于制作步骤总结关键要点，或者完全省略此部分]

        注意：
        - 根据实际内容灵活调整结构
        - 不要强行填充无关内容
        - 重点突出实用性和可操作性

        回答:'''
        #构造历史消息

        #添加系统消息
        message = [SystemMessage(content = system_prompt)]
        if history:
            for msg in history[-self.history_window_size:]:
                if msg['role'] == 'user':
                    message.append(HumanMessage(content=msg['content']))
                elif msg['role'] == 'assistant':
                    message.append(AIMessage(content=msg['content']))

        #构造本轮对话
        current_input = f"用户问题：{query}\n\n相关食谱信息为:{context}"
        #添加进历史消息
        message.append(HumanMessage(content=current_input))

        #获取回答
        response = self.llm.invoke(message)
        return response.content

    def generate_general_answer(self,query:str,context:List[Document],history:List[dict] = None):
        '''
        general类型的生成器
        Args:
            query:查询
            context:上下文
        '''
        context = self._build_context(context)
        system_prompt = '''
        你是一位专业的烹饪助手。请根据以下食谱信息回答用户的问题。

        用户问题: {question}

        相关食谱信息:
        {context}

        请提供详细、实用的回答。如果信息不足，请诚实说明。

        回答:'''

        #构造历史消息

        #添加系统消息
        message = [SystemMessage(content = system_prompt)]
        if history:
            for msg in history[-self.history_window_size:]:
                if msg['role'] == 'user':
                    message.append(HumanMessage(content=msg['content']))
                elif msg['role'] == 'assistant':
                    message.append(AIMessage(content=msg['content']))

        #构造本轮对话
        current_input = f"用户问题：{query}\n\n相关食谱信息为:{context}"
        #添加进历史消息
        message.append(HumanMessage(content=current_input))

        #获取回答
        response = self.llm.invoke(message)
        return response.content
