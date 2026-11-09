'''
RAG系统配置文件
'''
from dataclasses import dataclass
from typing import Dict,Any

@dataclass   #使用装饰符，不用写init之类的了
class RAGConfig:
    '''
    RAG路径配置类
    '''

    #路径配置
    data_path: str = './data/cook'
    vector_index_path: str = './vector_index'

    #模型配置
    embedding_model:str = 'BAAI/bge-small-zh-v1.5'
    model_name:str = 'deepseek-v3.2-exp'

    #检索参数
    top_k:int= 3

    #生成配置
    temperature:float = 0.1
    max_tokens:int = 2048

    #模型生成上下文大小
    history_window_size = 8

    @classmethod
    def from_dict(cls,config_dict:Dict[str,Any])->'RAGConfig':
        #从字典创建配置对象
        return cls(**config_dict)

    def to_dict(self)->Dict[str,Any]:
        #从配置对象创建字典
        return{
            'data_path': self.data_path,
            'index_save_path': self.vector_index_path,
            'embedding_model': self.embedding_model,
            'llm_model': self.model_name,
            'top_k': self.top_k,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'history_window_size':self.history_window_size
        }

#配置默认配置
DEFAULT_CONFIG = RAGConfig()

