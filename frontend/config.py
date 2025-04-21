"""
配置文件模块，包含应用程序的各种配置项。

此模块定义了数据库连接、应用程序设置和文件上传限制等配置。
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 项目路径配置
current_dir = Path(__file__).parent
project_root = current_dir.parent

# 数据库配置
DB_CONFIG = {
    "username": os.getenv("DB_USERNAME"),  # 数据库用户名
    "password": os.getenv("DB_PASSWORD"),  # 数据库密码
    "host": os.getenv("DB_HOST", "localhost"),  # 数据库主机地址，默认为localhost
    "port": os.getenv("DB_PORT", 3306),  # 数据库端口，默认为3306
    "database": os.getenv("TARGET_DB")  # 目标数据库名称
}

# 应用配置
APP_CONFIG = {
    "page_title": "OpenModelHub",  # 页面标题
    "page_icon": "🧠",  # 页面图标
    "layout": "wide",  # 页面布局模式
    "initial_sidebar_state": "expanded"  # 侧边栏初始状态
}

# 文件上传配置
UPLOAD_CONFIG = {
    "allowed_types": ["csv", "txt", "zip"],  # 允许上传的文件类型
    "max_size": 100 * 1024 * 1024  # 最大文件大小限制（100MB）
} 