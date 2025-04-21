"""
数据库连接管理模块。

此模块提供了创建和管理数据库会话的功能。
"""

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
    """
    创建并返回数据库会话工厂。

    Returns:
        async_sessionmaker: 异步会话工厂，用于创建数据库会话
    """
    engine = create_async_engine(
        f"mysql+aiomysql://{config.DB_CONFIG['username']}:{config.DB_CONFIG['password']}@{config.DB_CONFIG['host']}:{config.DB_CONFIG['port']}/{config.DB_CONFIG['database']}",
        echo=True
    )
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def get_db():
    """
    获取数据库会话的异步生成器。

    Yields:
        AsyncSession: 数据库会话对象

    Note:
        使用此函数时应该使用 async with 语句，以确保会话正确关闭。
    """
    Session = get_db_session()
    async with Session() as session:
        try:
            yield session
        finally:
            await session.close() 