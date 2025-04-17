import streamlit as st
import hashlib
from typing import Optional, Dict
from pathlib import Path
import sys
import asyncio
import nest_asyncio
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.extend([str(project_root), str(project_root/"database")])
sys.path.extend([str(project_root), str(project_root/"frontend")])
import frontend.database_api as db_api
from frontend.db import get_db_session

# 允许嵌套事件循环
nest_asyncio.apply()

class AuthManager:
    def __init__(self):
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'current_user' not in st.session_state:
            st.session_state.current_user = None
        # 使用当前事件循环
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self._ensure_admin_exists())

    async def _ensure_admin_exists(self):
        """确保管理员账户存在"""
        try:
            # 检查管理员账户是否存在
            admin = db_api.db_get_user_by_username("admin")
            if not admin:
                # 如果不存在，创建管理员账户
                db_api.db_create_user("admin", "admin", "系统管理员", is_admin=True)
                st.info("已创建默认管理员账户：admin/admin")
        except Exception as e:
            st.error(f"初始化管理员账户失败：{str(e)}")

    async def login(self, username: str, password: str) -> bool:
        """用户登录"""
        try:
            user = db_api.db_authenticate_user(username, password)
            if user:
                st.session_state.authenticated = True
                st.session_state.current_user = {
                    "user_id": user.user_id,
                    "username": user.user_name,
                    "role": "admin" if user.is_admin else "user"
                }
                return True
            return False
        except Exception as e:
            st.error(f"登录失败：{str(e)}")
            return False

    def logout(self):
        """用户登出"""
        st.session_state.authenticated = False
        st.session_state.current_user = None

    def get_current_user(self) -> Optional[Dict]:
        """获取当前用户信息"""
        return st.session_state.current_user

    def is_authenticated(self) -> bool:
        """检查用户是否已认证"""
        return st.session_state.authenticated

    def is_admin(self) -> bool:
        """检查当前用户是否为管理员"""
        return st.session_state.current_user and st.session_state.current_user["role"] == "admin" 