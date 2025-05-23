import asyncio
import pandas as pd
from io import BytesIO
from typing import List, Dict

def async_to_sync(async_func):
    """将异步函数转换为同步函数"""
    def wrapper(*args, **kwargs):
        return asyncio.run(async_func(*args, **kwargs))
    return wrapper

def parse_csv_columns(file_data: bytes) -> List[Dict]:
    """解析CSV文件的列信息"""
    df = pd.read_csv(BytesIO(file_data), nrows=1)
    return [{"col_name": col, "col_datatype": "text"} for col in df.columns]

def format_file_size(size_bytes: int) -> str:
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}TB"

def validate_file_upload(file, allowed_types: List[str], max_size: int) -> tuple[bool, str]:
    """验证上传文件"""
    if not file:
        return False, "Please select your file."
    
    if file.name.split('.')[-1].lower() not in allowed_types:
        return False, f"Unsupported file type. Supported file types are: {', '.join(allowed_types)}"
    
    if file.size > max_size:
        return False, f"File size exceeds the limit ({format_file_size(max_size)})"
    
    return True, "" 