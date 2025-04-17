from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
import streamlit as st

from pathlib import Path
import sys
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.extend([str(project_root), str(project_root/"database")])
sys.path.extend([str(project_root), str(project_root/"frontend")])
import frontend.config as config

@st.cache_resource
def get_db_session():
    """创建并返回数据库会话工厂"""
    engine = create_async_engine(
        f"mysql+aiomysql://{config.DB_CONFIG['username']}:{config.DB_CONFIG['password']}@{config.DB_CONFIG['host']}:{config.DB_CONFIG['port']}/{config.DB_CONFIG['database']}",
        echo=True
    )
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def get_db():
    """获取数据库会话"""
    Session = get_db_session()
    async with Session() as session:
        try:
            yield session
        finally:
            await session.close() 