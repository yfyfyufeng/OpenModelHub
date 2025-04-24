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

# Allow nested event loops
nest_asyncio.apply()

class AuthManager:
    def __init__(self):
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'current_user' not in st.session_state:
            st.session_state.current_user = None
        # Use current event loop
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self._ensure_admin_exists())

    async def _ensure_admin_exists(self):
        """Ensure admin account exists"""
        try:
            # Check if admin account exists
            admin = db_api.db_get_user_by_username("admin")
            if not admin:
                # If not exists, create admin account
                db_api.db_create_user("admin", "admin", "System Administrator", is_admin=True)
                st.info("Default admin account created: admin/admin")
        except Exception as e:
            st.error(f"Failed to initialize admin account: {str(e)}")

    async def login(self, username: str, password: str) -> bool:
        """User login"""
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
            st.error(f"Login failed: {str(e)}")
            return False

    def logout(self):
        """User logout"""
        st.session_state.authenticated = False
        st.session_state.current_user = None

    def get_current_user(self) -> Optional[Dict]:
        """Get current user information"""
        return st.session_state.current_user

    def is_authenticated(self) -> bool:
        """Check if user is authenticated"""
        return st.session_state.authenticated

    def is_admin(self) -> bool:
        """Check if current user is admin"""
        return st.session_state.current_user and st.session_state.current_user["role"] == "admin"
