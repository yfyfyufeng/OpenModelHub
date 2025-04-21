import asyncio
import pandas as pd
from io import BytesIO
from typing import List, Dict
import nest_asyncio

# 允许嵌套事件循环
nest_asyncio.apply()

# 创建全局事件循环
_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_loop)

def async_to_sync(async_func):
    """
    将异步函数转换为同步函数的装饰器。
    
    Args:
        async_func: 要转换的异步函数
        
    Returns:
        function: 转换后的同步函数，可以在同步环境中调用
    """
    def wrapper(*args, **kwargs):
        return _loop.run_until_complete(async_func(*args, **kwargs))
    return wrapper

def parse_csv_columns(file_data: bytes) -> List[Dict]:
    """
    解析CSV文件的列信息。

    Args:
        file_data (bytes): CSV文件的二进制数据

    Returns:
        List[Dict]: 包含列信息的列表，每个元素是一个字典，包含列名和数据类型
    """
    df = pd.read_csv(BytesIO(file_data), nrows=1)
    return [{"col_name": col, "col_datatype": "text"} for col in df.columns]

def format_file_size(size_bytes: int) -> str:
    """
    将字节大小格式化为人类可读的字符串。

    Args:
        size_bytes (int): 文件大小（字节）

    Returns:
        str: 格式化后的文件大小字符串，例如 "1.5MB"
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}TB"

def validate_file_upload(file, allowed_types: List[str], max_size: int) -> tuple[bool, str]:
    """
    验证上传文件的有效性。

    Args:
        file: 上传的文件对象
        allowed_types (List[str]): 允许的文件类型列表
        max_size (int): 最大文件大小（字节）

    Returns:
        tuple[bool, str]: (是否有效, 错误信息)
            - 如果文件有效，返回 (True, "")
            - 如果文件无效，返回 (False, 错误信息)
    """
    if not file:
        return False, "请选择文件"
    
    if file.name.split('.')[-1].lower() not in allowed_types:
        return False, f"不支持的文件类型，仅支持: {', '.join(allowed_types)}"
    
    if file.size > max_size:
        return False, f"文件大小超过限制 ({format_file_size(max_size)})"
    
    return True, "" 