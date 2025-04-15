# db_api.py
import asyncio
from database.database_interface import *
from sqlalchemy.ext.asyncio import AsyncSession
import os
from pathlib import Path
import sys
from dotenv import load_dotenv
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.extend([str(project_root), str(project_root/"frontend")])

def async_to_sync(async_func):
    def wrapper(*args, **kwargs):
        return asyncio.run(async_func(*args, **kwargs))
    return wrapper

# 模型操作
@async_to_sync
async def db_list_models():
    async with get_db_session()() as session:
        return await list_models(session)

@async_to_sync
async def db_get_model(model_id: int):
    async with get_db_session()() as session:
        return await get_model_by_id(session, model_id)

# 数据集操作
@async_to_sync
async def db_list_datasets():
    async with get_db_session()() as session:
        return await list_datasets(session)

@async_to_sync
async def db_create_dataset(name: str, desc: str, file_path: str):
    async with get_db_session()() as session:
        dataset_data = {
            "ds_name": name,
            "ds_size": os.path.getsize(file_path),
            "media": "text",
            "task": "classification",
            "columns": [{"col_name": "text", "col_datatype": "varchar(255)"}]
        }
        return await create_dataset(session, dataset_data)

# 用户操作
@async_to_sync
async def db_list_users():
    async with get_db_session()() as session:
        return await list_users(session)

@async_to_sync
async def db_create_user(username: str, password: str):
    async with get_db_session()() as session:
        return await create_user(session, {
            "user_name": username,
            "password_hash": password,  # 应使用加密哈希
            "affiliate": "default"
        })
        
@async_to_sync
async def db_authenticate_user(username: str, password: str):
    async with get_db_session()() as session:
        result = await session.execute(select(User).filter_by(user_name=username))
        user = result.scalar_one_or_none()
        if user and user.password_hash == password:  # 实际应使用密码哈希验证
            return user
        return None
    
# 文件操作
@async_to_sync
async def db_save_file(file_data: bytes, filename: str):
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    file_path = upload_dir / filename
    with open(file_path, "wb") as f:
        f.write(file_data)
    return str(file_path)

@async_to_sync
async def db_get_file(filename: str):
    file_path = Path("uploads") / filename
    if file_path.exists():
        with open(file_path, "rb") as f:
            return f.read()
    return None

def get_db_session():
  
    load_dotenv()
    DB_USERNAME = os.getenv("DB_USERNAME")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", 3306)
    TARGET_DB = os.getenv("TARGET_DB")
    
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    
    engine = create_async_engine(
        f"mysql+aiomysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{TARGET_DB}",
        echo=True
    )
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
