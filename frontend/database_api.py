# db_api.py
import asyncio
from database.database_interface import *
from sqlalchemy.ext.asyncio import AsyncSession
import os
from pathlib import Path
import sys
from dotenv import load_dotenv
import socket
from datetime import datetime

current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.extend([str(project_root), str(project_root / "database")])
sys.path.extend([str(project_root), str(project_root / "frontend")])
sys.path.extend([str(project_root), str(project_root / "security")])
sys.path.extend([str(project_root), str(project_root / "agent")])
from database.database_interface import (
    list_models, get_model_by_id, list_datasets, get_dataset_by_id,
    list_users, get_user_by_id, list_affiliations, init_database,
    create_user, update_user, delete_user
)
from database.database_interface import User
from sqlalchemy import select, text
from sqlalchemy.orm import selectinload
from agent.agent_main import query_agent
from frontend.config import DATA_CONFIG

try:
    from security.conn import InitUser, GetUser, StoreFile, LoadFile, CreateInvitation, AcceptInvitation, RevokeAccess
    from security.enc import encrypt, decrypt

    SECURITY_AVAILABLE = True
except (ImportError, ConnectionRefusedError):
    SECURITY_AVAILABLE = False
    print("Security module not available, running in non-encrypted mode")

curr_username = None
curr_password = None


def safe_get_value(obj, attr_name):
    if hasattr(obj, attr_name):
        attr = getattr(obj, attr_name)
        if hasattr(attr, 'value'):  # Check if it's an enum
            return attr.value
        else:
            return attr
    return "未知"


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


# Agent query
@async_to_sync
async def db_agent_query(query: str, instance_type: int):
    """Query database using natural language"""
    async with get_db_session()() as session:
        try:
            # Use query_agent function from agent module
            result = await query_agent(query, instance_type=instance_type, verbose=False, session=session)

            # Return format consistent with agent_main.py
            return (result['sql_res'] if result['err'] == 0 and result['sql_res'] else [],
                    {
                        'natural_language_query': query,
                        'generated_sql': result['sql'],
                        'error_code': result['err'],
                        'sql_res': result['sql_res'],
                        'has_results': bool(result['sql_res']),
                        'error': None if result['err'] == 0 else 'SQL execution failed'
                    })
        except Exception as e:
            print(f"Error executing query: {str(e)}")
            return [], {
                'natural_language_query': query,
                'generated_sql': '',
                'error_code': 1,
                'has_results': False,
                'error': str(e),
                'sql_res': []
            }


# Model operations
@async_to_sync
async def db_list_models():
    """Get all models list"""
    async with get_db_session()() as session:
        try:
            # First get all models
            stmt = select(Model)
            result = await session.execute(stmt)
            models = result.scalars().all()

            # Load related data for each model
            for model in models:
                await session.refresh(model, ['tasks', 'authors', 'datasets', 'cnn', 'rnn', 'transformer'])

            return models
        except Exception as e:
            print(f"Error getting model list: {str(e)}")
            return []


@async_to_sync
async def db_get_model(model_id: int):
    """获取指定ID的模型"""
    async with get_db_session()() as session:
        stmt = select(Model).options(
            selectinload(Model.tasks),
            selectinload(Model.authors),
            selectinload(Model.datasets),
            selectinload(Model.cnn),
            selectinload(Model.rnn),
            selectinload(Model.transformer)
        ).filter(Model.model_id == model_id)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()


# Dataset operations
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
        dataset = Dataset(
            ds_name=dataset_data["ds_name"],
            ds_size=dataset_data["ds_size"],
            media=dataset_data["media"],
            description=dataset_data["description"],
            created_at=dataset_data.get("created_at", datetime.now())  # 使用传入的时间或当前时间
        )
        return await create_dataset(session, dataset_data)


# User operations
@async_to_sync
async def db_get_dataset(dataset_id: int):
    """获取指定ID的数据集"""
    async with get_db_session()() as session:
        stmt = select(Dataset).options(
            selectinload(Dataset.columns),
            selectinload(Dataset.Dataset_TASK)
        ).filter(Dataset.ds_id == dataset_id)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()


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


async def db_save_file(file_data: bytes, filename: str):
    global curr_username, curr_password
    if is_port_in_use(8080) and SECURITY_AVAILABLE:
        key = os.urandom(32)


@async_to_sync
async def db_get_user_by_id(user_id: int):
    """Get user by ID
    Args:
        user_id: User ID to get
    Returns:
        User object or None if not found
    """
    async with get_db_session()() as session:
        try:
            stmt = select(User).filter(User.user_id == user_id)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            print(f"Error getting user by ID: {str(e)}")
            return None


@async_to_sync
async def db_get_file(filename: str):
    """从数据目录获取文件"""
    file_path = Path(__file__).parent.parent / "database" / "data" / filename
    if file_path.exists():
        with open(file_path, "rb") as f:
            return f.read()
    return None


# File operations
@async_to_sync
async def db_save_file(file_data: bytes, filename: str, file_type: str = "datasets"):
    """Save file to specified directory"""
    # Select save directory
    if file_type == "models":
        save_dir = DATA_CONFIG["models_dir"]
    else:
        save_dir = DATA_CONFIG["datasets_dir"]

    # Create timestamped filename to avoid conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = save_dir / f"{timestamp}_{filename}"

    # Save file
    with open(file_path, "wb") as f:
        f.write(file_data)

    return str(file_path)


@async_to_sync
async def db_get_file(filename: str, file_type: str = "datasets"):
    """Get file from specified directory"""
    try:
        # Select directory
        if file_type == "models":
            search_dir = DATA_CONFIG["models_dir"]
        else:
            search_dir = DATA_CONFIG["datasets_dir"]

        # Ensure directory exists
        if not search_dir.exists():
            print(f"Directory does not exist: {search_dir}")
            return None

        # Find latest matching file
        matching_files = list(search_dir.glob(f"*_{filename}"))
        if not matching_files:
            # Try direct filename match
            direct_match = search_dir / filename
            if direct_match.exists():
                with open(direct_match, "rb") as f:
                    return f.read()
            print(f"File not found: {filename}")
            return None

        # Get latest file
        latest_file = max(matching_files, key=lambda x: x.stat().st_mtime)
        print(f"Found file: {latest_file}")

        with open(latest_file, "rb") as f:
            return f.read()

    except Exception as e:
        print(f"Error getting file: {str(e)}")
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
    """Create new model"""
    async with get_db_session()() as session:
        try:
            # Convert enum types
            from database.database_schema import ArchType, Media_type, Trainname, Task_name

            # Create model
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

            # Add tasks
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
            raise Exception(f"Failed to create model: {str(e)}")


@async_to_sync
async def db_export_all_data():
    """导出所有数据到JSON格式"""
    async with get_db_session()() as session:
        try:
            # 获取所有数据
            users = await list_users(session)
            models = await list_models(session)
            datasets = await list_datasets(session)
            affiliations = await list_affiliations(session)

            # 构建JSON数据
            json_data = {
                "affiliation": [
                    {
                        "affil_name": affil.affil_name
                    } for affil in affiliations
                ],
                "user": [
                    {
                        "user_id": user.user_id,
                        "user_name": user.user_name,
                        "password_hash": user.password_hash,
                        "affiliate": user.affiliate,
                        "is_admin": user.is_admin
                    } for user in users
                ],
                "dataset": [
                    {
                        "ds_name": dataset.ds_name,
                        "ds_size": dataset.ds_size,
                        "media": dataset.media.value if hasattr(dataset.media, 'value') else dataset.media,
                        "task": [task.task.value if hasattr(task.task, 'value') else task.task for task in
                                 dataset.Dataset_TASK],
                        "columns": [
                            {
                                "col_name": col.col_name,
                                "col_datatype": col.col_datatype
                            } for col in dataset.columns
                        ]
                    } for dataset in datasets
                ],
                "model": [
                    {
                        "model_name": model.model_name,
                        "param_num": model.param_num,
                        "media_type": model.media_type.value if hasattr(model.media_type,'value') else model.media_type,
                        "arch_name": model.arch_name.value if hasattr(model.arch_name, 'value') else model.arch_name,
                        "trainname": model.trainname.value if hasattr(model.trainname, 'value') else model.trainname,
                        "task": [task.task_name.value if hasattr(task.task_name, 'value') else task.task_name for task
                                 in model.tasks],
                        "param": 10  # 示例值
                    } for model in models
                ]
            }

            return json_data
        except Exception as e:
            print(f"导出数据时出错: {str(e)}")
            return None


@async_to_sync
async def db_update_user(user_id: int, is_admin: bool = None, affiliate: str = None) -> bool:
    """Update user information
    Args:
        user_id: User ID to update
        is_admin: New admin status (optional)
        affiliate: New affiliate (optional)
    Returns:
        bool: True if update successful, False otherwise
    """
    try:
        async with get_db_session()() as session:
            # Get user by ID
            stmt = select(User).filter(User.user_id == user_id)
            result = await session.execute(stmt)
            user = result.scalar_one_or_none()

            if not user:
                return False

            # Update fields if provided
            if is_admin is not None:
                user.is_admin = is_admin
            if affiliate is not None:
                user.affiliate = affiliate

            # Commit changes
            await session.commit()
            return True
    except Exception as e:
        print(f"Error updating user: {str(e)}")
        return False