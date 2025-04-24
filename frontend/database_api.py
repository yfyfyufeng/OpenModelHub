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
    async with get_db_session()() as session:
        return await get_model_by_id(session, model_id)

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

# File operations
'''
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
'''
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

