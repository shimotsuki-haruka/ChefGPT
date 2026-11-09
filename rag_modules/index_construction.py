from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List, Dict, Any
import logging 
from langchain_community.vectorstores import FAISS
from pathlib import Path

logger = logging.getLogger(__name__)
class IndexConstructionModule:
    '''
    索引构建模块 -负责向量化和索引构建
    向量储存以及向量载入
    '''
    def __init__(self,model_name:str = "BAAI/bge-small-zh-v1.5", index_save_path:str = "./vector_index"):
        self.model_name = model_name
        #索引配置的储存路径
        self.index_save_path = index_save_path
        #先初始化 嵌入模型 与 向量储存库 为空
        self.embedding_model = None
        self.vectorstore = None
        #初始化模型
        
    def set_up_embedding(self):
        '''
        从huggingface加载模型
        '''
        self.embedding_model = HuggingFaceEmbeddings(
            model_name = self.model_name,
            model_kwargs = {'device':'cpu'},  #指定设备为cpu
            encode_kwargs = { 'normalize_embeddings':True}  #开启生成的向量归一化
        )
    
    def build_vector_index(self,chunks:List[Document]):
        '''
        构建向量储存
        '''
        if not chunks:
            raise ValueError("文档块列表不能为空")
        #读取所有chunk的文本内容
        #texts = [chunk.page_content for chunk in chunks]
        #读取元数据
        #metadatas = [chunk.metadata for chunk in chunks]
        for chunk in chunks:
            dish_name = chunk.metadata.get("dish_name", "")
            chunk.page_content = f"菜品：{dish_name}\n{chunk.page_content}"
        #构建faiss向量库索引
        self.vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embedding_model,
        )
    
        return self.vectorstore
    
    def save_index(self):
        '''
        保存向量索引到指定位置
        '''
        if not self.vectorstore:
            raise ValueError("请先构建向量索引")
        
        #确保保存的目录存在
        #这一句话是用pathlib库建立目录的用法，如果不存在父级目录也一并建立，目标目录存在也无所谓
        Path(self.index_save_path).mkdir(parents=True,exist_ok=True)

        #保存向量索引
        self.vectorstore.save_local(self.index_save_path)
    
    def load_index(self):
        '''
        从配置的路径加载向量索引
        '''
        if not self.embedding_model:
            self.set_up_embedding()
        
        #这一句判断有点问题，只能判断index文件夹存不存在,无法保证里面存不存在faiss文件
        index_path = Path(self.index_save_path)
        faiss_files = [f for f in index_path.iterdir() if f.suffix == ".faiss"]
        if not faiss_files:
            logger.warning("索引目录中没有找到 .faiss 文件")
            return None
        
        
        self.vectorstore = FAISS.load_local(
            self.index_save_path,
            self.embedding_model,
            allow_dangerous_deserialization=True  #即允许python反序列化pkl文件，即加载Metadata
        )

        return self.vectorstore








    
