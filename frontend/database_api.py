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
sys.path.extend([str(project_root), str(project_root/"database")])
sys.path.extend([str(project_root), str(project_root/"frontend")])
from database.database_interface import (
    list_models, get_model_by_id, list_datasets, get_dataset_by_id,
    list_users, get_user_by_id, list_affiliations, init_database,
    create_user, update_user, delete_user
)
from database.database_interface import User
from sqlalchemy import select

def async_to_sync(async_func):
    def wrapper(*args, **kwargs):
        return asyncio.run(async_func(*args, **kwargs))
    return wrapper

# 模型操作
@async_to_sync
async def db_list_models():
    """获取所有模型列表"""
    async with get_db_session()() as session:
        try:
            # 首先获取所有模型
            stmt = select(Model)
            result = await session.execute(stmt)
            models = result.scalars().all()
            
            # 为每个模型加载关联数据
            for model in models:
                await session.refresh(model, ['tasks', 'authors', 'datasets', 'cnn', 'rnn', 'transformer'])
            
            return models
        except Exception as e:
            print(f"获取模型列表时出错: {str(e)}")
            return []

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
async def db_create_dataset(name: str, dataset_data: dict):
    async with get_db_session()() as session:
        return await create_dataset(session, dataset_data)

# 用户操作
@async_to_sync
async def db_list_users():
    async with get_db_session()() as session:
        return await list_users(session)

@async_to_sync
async def db_create_user(username: str, password: str, affiliate: str = None, is_admin: bool = False):
    async with get_db_session()() as session:
        user = User(
            user_name=username,
            password_hash=password,
            affiliate=affiliate,
            is_admin=is_admin
        )
        session.add(user)
        await session.commit()
        return user

@async_to_sync
async def db_authenticate_user(username: str, password: str):
    async with get_db_session()() as session:
        stmt = select(User).where(User.user_name == username)
        result = await session.execute(stmt)
        user = result.scalar_one_or_none()
        if user and user.password_hash == password:
            return user
        return None

@async_to_sync
async def db_get_user_by_username(username: str):
    async with get_db_session()() as session:
        stmt = select(User).where(User.user_name == username)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

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

