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
    """
    将异步函数转换为同步函数的装饰器。

    Args:
        async_func: 要转换的异步函数

    Returns:
        function: 转换后的同步函数
    """
    def wrapper(*args, **kwargs):
        return asyncio.run(async_func(*args, **kwargs))
    return wrapper

# 模型操作
@async_to_sync
async def db_list_models():
    """
    获取所有模型列表。

    Returns:
        List[Model]: 包含所有模型对象的列表，每个模型对象都加载了关联数据
    """
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
    """
    根据ID获取特定模型。

    Args:
        model_id (int): 模型ID

    Returns:
        Model: 模型对象，如果未找到则返回None
    """
    async with get_db_session()() as session:
        return await get_model_by_id(session, model_id)

# 数据集操作
@async_to_sync
async def db_list_datasets():
    """
    获取所有数据集列表。

    Returns:
        List[Dataset]: 包含所有数据集对象的列表
    """
    async with get_db_session()() as session:
        return await list_datasets(session)

@async_to_sync
async def db_create_dataset(name: str, desc: str, file_path: str) -> Dataset:
    """
    创建新的数据集。

    Args:
        name (str): 数据集名称
        desc (str): 数据集描述
        file_path (str): 数据集文件路径

    Returns:
        Dataset: 新创建的数据集对象
    """
    session = get_db_session()
    try:
        # 获取文件大小
        file_size = os.path.getsize(file_path)
        
        # 创建数据集
        dataset_data = {
            "ds_name": name,
            "ds_size": file_size,
            "media": "text",  # 默认媒体类型
            "task": "classification",  # 默认任务类型
            "columns": []  # 空列列表，后续可以添加
        }
        
        dataset = await create_dataset(session, dataset_data)
        return dataset
    finally:
        await session.close()

# 用户操作
@async_to_sync
async def db_list_users():
    """
    获取所有用户列表。

    Returns:
        List[User]: 包含所有用户对象的列表
    """
    async with get_db_session()() as session:
        return await list_users(session)

@async_to_sync
async def db_create_user(username: str, password: str, affiliate: str = None, is_admin: bool = False):
    """
    创建新用户。

    Args:
        username (str): 用户名
        password (str): 密码
        affiliate (str, optional): 所属机构
        is_admin (bool, optional): 是否为管理员

    Returns:
        User: 新创建的用户对象
    """
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
    """
    验证用户身份。

    Args:
        username (str): 用户名
        password (str): 密码

    Returns:
        User: 验证成功的用户对象，验证失败则返回None
    """
    async with get_db_session()() as session:
        stmt = select(User).where(User.user_name == username)
        result = await session.execute(stmt)
        user = result.scalar_one_or_none()
        if user and user.password_hash == password:
            return user
        return None

@async_to_sync
async def db_get_user_by_username(username: str):
    """
    根据用户名获取用户。

    Args:
        username (str): 用户名

    Returns:
        User: 用户对象，如果未找到则返回None
    """
    async with get_db_session()() as session:
        stmt = select(User).where(User.user_name == username)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

# 文件操作
@async_to_sync
async def db_save_file(file_data: bytes, filename: str):
    """
    保存文件到服务器。

    Args:
        file_data (bytes): 文件二进制数据
        filename (str): 文件名

    Returns:
        str: 保存后的文件路径
    """
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    file_path = upload_dir / filename
    with open(file_path, "wb") as f:
        f.write(file_data)
    return str(file_path)

@async_to_sync
async def db_get_file(filename: str):
    """
    获取文件内容。

    Args:
        filename (str): 文件名

    Returns:
        bytes: 文件二进制数据，如果文件不存在则返回None
    """
    file_path = Path("uploads") / filename
    if file_path.exists():
        with open(file_path, "rb") as f:
            return f.read()
    return None

def get_db_session():
    """
    获取数据库会话工厂函数。

    Returns:
        function: 返回一个创建数据库会话的函数
    """
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

