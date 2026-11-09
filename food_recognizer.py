from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# 配置路径（如果放的位置不同，修改这里）
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'my_model1.h5'
CLASSES_PATH = BASE_DIR / 'assets' / 'classes.txt'

# 加载模型和类别（延迟加载，避免一启动就占内存）
_model = None
_classes = None

def _load_model():
    global _model, _classes
    if _model is None:
        _model = load_model(MODEL_PATH, compile=False)
        with open(CLASSES_PATH, 'r', encoding='utf-8') as f:
            _classes = [line.strip() for line in f.readlines() if line.strip()]
    return _model, _classes

def recognize_food(image_source) -> str:
    """
    识别图片中的食物类别，返回标签字符串（英文，如 'Fried food'）
    """
    model, classes = _load_model()

    # 确定模型期望的输入尺寸（你训练时用的是 64x64）
    input_shape = model.input_shape[1:3]  # (height, width)
    h, w = input_shape[0], input_shape[1]

    # 预处理图片
    img = Image.open(image_source).convert('RGB')
    img = img.resize((w, h))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 预测
    preds = model.predict(img_array, verbose=0)[0]
    top_idx = np.argmax(preds)
    return classes[top_idx]
# 英文标签 -> 日常中文说法
LABEL_ZH = {
    "Bread": "面包",
    "Dairy product": "奶制品",
    "Dessert": "甜点",
    "Egg": "鸡蛋料理",
    "Fried food": "油炸小吃",
    "Meat": "肉类",
    "Noodles-Pasta": "面条/意面",
    "Rice": "米饭",
    "Seafood": "海鲜",
    "Soup": "汤",
    "Vegetable-Fruit": "蔬菜水果",
}

def make_food_question(food_label_en: str) -> str:
    """将英文食物标签转为自然的中文提问，例如 '我想做一道汤，有什么推荐做法吗？'"""
    zh_name = LABEL_ZH.get(food_label_en, food_label_en)
    return f"我想做一道{zh_name}，有什么推荐做法吗？"