# db_api.py
import asyncio
from database.database_interface import *
from sqlalchemy.ext.asyncio import AsyncSession
import os
from pathlib import Path
import sys
from dotenv import load_dotenv
import socket
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.extend([str(project_root), str(project_root/"database")])
sys.path.extend([str(project_root), str(project_root/"frontend")])
sys.path.extend([str(project_root), str(project_root/"security")])
sys.path.extend([str(project_root), str(project_root/"agent")])
from database.database_interface import (
    list_models, get_model_by_id, list_datasets, get_dataset_by_id,
    list_users, get_user_by_id, list_affiliations, init_database,
    create_user, update_user, delete_user
)
from database.database_interface import User
from sqlalchemy import select, text
from sqlalchemy.orm import selectinload
from agent.agent_main import query_agent

try:
    from security.conn import InitUser, GetUser, StoreFile, LoadFile, CreateInvitation, AcceptInvitation, RevokeAccess
    from security.enc import encrypt, decrypt
    SECURITY_AVAILABLE = True
except (ImportError, ConnectionRefusedError):
    SECURITY_AVAILABLE = False
    print("Security module not available, running in non-encrypted mode")

curr_username = None
curr_password = None

def is_port_in_use(port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    except:
        return False

def async_to_sync(async_func):
    def wrapper(*args, **kwargs):
        return asyncio.run(async_func(*args, **kwargs))
    return wrapper

# Agent查询
@async_to_sync
async def db_agent_query(query: str):
    """使用自然语言查询数据库"""
    async with get_db_session()() as session:
        try:
            # 使用 agent 的 query_agent 函数
            result = await query_agent(query, verbose=False, session=session)
            
            # 返回与 agent_main.py 一致的格式
            return (result['sql_res'] if result['err'] == 0 and result['sql_res'] else [], 
                   {
                       'natural_language_query': query,
                       'generated_sql': result['sql'],
                       'error_code': result['err'],
                       'sql_res': result['sql_res'],
                       'has_results': bool(result['sql_res']),
                       'error': None if result['err'] == 0 else 'SQL执行失败'
                   })
        except Exception as e:
            print(f"执行查询时出错: {str(e)}")
            return [], {
                'natural_language_query': query,
                'generated_sql': '',
                'error_code': 1,
                'has_results': False,
                'error': str(e),
                'sql_res': []
            }

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
        stmt = select(Dataset).options(
            selectinload(Dataset.columns),
            selectinload(Dataset.Dataset_TASK)
        )
        result = await session.execute(stmt)
        return result.scalars().all()

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
    global curr_username, curr_password
    if is_port_in_use(8080) and SECURITY_AVAILABLE:
        try:
            InitUser(username, password)
        except Exception as e:
            print("Error in security: InitUser:", str(e))
    curr_username = username
    curr_password = password
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
    global curr_username, curr_password
    if is_port_in_use(8080) and SECURITY_AVAILABLE:
        try:
            GetUser(username, password)
        except Exception as e:
            print("Error in security: GetUser:", str(e))
            return None
    curr_username = username
    curr_password = password
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
    global curr_username, curr_password
    if is_port_in_use(8080) and SECURITY_AVAILABLE:
        key = os.urandom(32)
        try:
            StoreFile(curr_username, curr_password, filename, key)
        except Exception as e:
            print("Error in security: StoreFile:", str(e))
        file_data = encrypt(key, file_data)

    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    file_path = upload_dir / filename
    with open(file_path, "wb") as f:
        f.write(file_data)
    return str(file_path)

@async_to_sync
async def db_get_file(filename: str):
    global curr_username, curr_password
    if is_port_in_use(8080) and SECURITY_AVAILABLE:
        try:
            LoadFile(curr_username, curr_password, filename)
        except Exception as e:
            print("Error in security: LoadFile:", str(e))
        key = os.urandom(32)

    file_path = Path("uploads") / filename
    if file_path.exists():
        with open(file_path, "rb") as f:
            if is_port_in_use(8080) and SECURITY_AVAILABLE:
                return decrypt(key, f.read())
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
@async_to_sync
async def db_create_model(model_data: dict):
    """创建新模型"""
    async with get_db_session()() as session:
        try:
            # 转换枚举类型
            from database.database_schema import ArchType, Media_type, Trainname, Task_name
            
            # 创建模型
            model = Model(
                model_name=model_data["model_name"],
                param_num=model_data["param_num"],
                media_type=Media_type[model_data["media_type"]],
                arch_name=ArchType[model_data["arch_name"]],
                trainname=Trainname[model_data["trainname"]],
                param=model_data["param"]
            )
            session.add(model)
            await session.flush()
            
            # 添加任务
            for task_name in model_data["tasks"]:
                task = ModelTask(
                    model_id=model.model_id,
                    task_name=Task_name[task_name]
                )
                session.add(task)
            
            await session.commit()
            await session.refresh(model)
            return model
        except Exception as e:
            await session.rollback()
            raise Exception(f"创建模型失败: {str(e)}")

