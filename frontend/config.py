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
    "username": os.getenv("DB_USERNAME"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", 3306),
    "database": os.getenv("TARGET_DB")
}

# 应用配置
APP_CONFIG = {
    "page_title": "OpenModelHub",
    "page_icon": "🧠",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# 文件上传配置
UPLOAD_CONFIG = {
    "allowed_types": ["csv", "txt", "zip"],
    "max_size": 100 * 1024 * 1024  # 100MB
} 

# 数据上传
DATA_CONFIG = {
    "base_dir": project_root / "database" / "data",
    "models_dir": project_root / "database" / "data" / "models",
    "datasets_dir": project_root / "database" / "data" / "datasets"
}

# 如果不存在数据
for dir_path in DATA_CONFIG.values():
    dir_path.mkdir(parents=True, exist_ok=True)