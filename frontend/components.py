import streamlit as st
import pandas as pd
import asyncio
import nest_asyncio
from typing import List, Dict

from pathlib import Path
import sys
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.extend([str(project_root), str(project_root/"database")])
sys.path.extend([str(project_root), str(project_root/"frontend")])
import frontend.database_api as db_api
from frontend.utils import parse_csv_columns, validate_file_upload
from frontend.config import UPLOAD_CONFIG

# 允许嵌套事件循环
nest_asyncio.apply()

class Sidebar:
    """
    侧边栏组件类，负责渲染和管理侧边栏的显示内容。
    
    Attributes:
        auth_manager: 认证管理器实例，用于处理用户认证相关功能
    """
    def __init__(self, auth_manager):
        """
        初始化侧边栏组件。

        Args:
            auth_manager: 认证管理器实例
        """
        self.auth_manager = auth_manager

    def render(self):
        """
        渲染侧边栏的主要方法。
        根据用户认证状态显示不同的内容。

        Returns:
            str: 当前选中的页面名称
        """
        with st.sidebar:
            st.title("OpenModelHub")
            if not self.auth_manager.is_authenticated():
                self._render_login_form()
            else:
                self._render_user_info()
            return self._render_navigation()

    def _render_login_form(self):
        """
        渲染登录表单。
        处理用户登录请求并显示登录结果。
        """
        with st.form("登录", clear_on_submit=True):
            username = st.text_input("用户名")
            password = st.text_input("密码", type="password")
            if st.form_submit_button("登录"):
                # 使用当前事件循环
                loop = asyncio.get_event_loop()
                if loop.run_until_complete(self.auth_manager.login(username, password)):
                    st.rerun()
                else:
                    st.error("用户名或密码错误")

    def _render_user_info(self):
        """
        渲染当前登录用户的信息。
        显示用户名和欢迎信息。
        """
        user = self.auth_manager.get_current_user()
        st.success(f"欢迎，{user['username']}！")
        if st.button("退出登录"):
            self.auth_manager.logout()
            st.rerun()

    def _render_navigation(self):
        """
        渲染导航菜单。
        根据用户权限显示可访问的页面选项。

        Returns:
            str: 用户选择的页面名称
        """
        pages = ["首页", "模型仓库", "数据集管理", "智能查询"]
        if self.auth_manager.is_admin():
            pages.append("用户管理")
        return st.radio("导航", pages)

class UserManager:
    """
    用户管理组件类，负责处理用户相关的管理功能。
    """
    def __init__(self):
        """初始化用户管理组件。"""
        self.users = db_api.db_list_users()

    def render(self):
        """
        渲染用户管理界面。
        显示用户列表和管理功能。
        """
        st.header("👥 用户管理")
        
        # 创建用户表单
        with st.expander("➕ 添加新用户", expanded=False):
            with st.form("new_user", clear_on_submit=True):
                username = st.text_input("用户名*")
                password = st.text_input("密码*", type="password")
                is_admin = st.checkbox("管理员权限")
                affiliate = st.text_input("所属机构")
                
                if st.form_submit_button("创建用户"):
                    if not username or not password:
                        st.error("带*的字段为必填项")
                    else:
                        try:
                            # 检查用户名是否已存在
                            existing_user = db_api.db_get_user_by_username(username)
                            if existing_user:
                                st.error(f"用户名 '{username}' 已存在")
                            else:
                                # 创建新用户
                                db_api.db_create_user(username, password, affiliate, is_admin=is_admin)
                                st.success("用户创建成功")
                                st.rerun()
                        except Exception as e:
                            st.error(f"创建失败：{str(e)}")
        
        # 用户列表
        df = pd.DataFrame([{
            "ID": user.user_id,
            "用户名": user.user_name,
            "所属机构": user.affiliate,
            "管理员": "✅" if user.is_admin else "❌"
        } for user in self.users])
        
        st.dataframe(
            df,
            column_config={
                "ID": "用户ID",
                "管理员": st.column_config.CheckboxColumn("管理员状态")
            },
            use_container_width=True,
            hide_index=True
        )

class DatasetUploader:
    """
    数据集上传组件类，负责处理数据集的上传和管理。
    """
    def __init__(self):
        """初始化数据集上传组件。"""
        self.allowed_types = UPLOAD_CONFIG["allowed_types"]
        self.max_size = UPLOAD_CONFIG["max_size"]

    def render(self):
        """
        渲染数据集上传界面。
        显示上传表单和处理上传逻辑。
        """
        with st.expander("📤 上传新数据集", expanded=False):
            with st.form("dataset_upload", clear_on_submit=True):
                name = st.text_input("数据集名称*")
                desc = st.text_area("描述")
                media_type = st.selectbox("媒体类型", ["text", "image", "audio", "video"])
                task_type = st.selectbox("任务类型", ["classification", "detection", "generation"])
                file = st.file_uploader("选择数据文件*", type=self.allowed_types)
                
                if st.form_submit_button("提交"):
                    return self._handle_submit(name, desc, media_type, task_type, file)
        return False

    def _handle_submit(self, name: str, desc: str, media_type: str, task_type: str, file):
        """
        处理数据集上传提交。

        Args:
            name (str): 数据集名称
            desc (str): 数据集描述
            media_type (str): 媒体类型
            task_type (str): 任务类型
            file: 上传的文件对象
        """
        if not name or not file:
            st.error("带*的字段为必填项")
            return False

        is_valid, error_msg = validate_file_upload(file, self.allowed_types, self.max_size)
        if not is_valid:
            st.error(error_msg)
            return False

        try:
            file_path = db_api.db_save_file(file.getvalue(), file.name)
            columns = []
            if file.name.endswith(".csv"):
                columns = parse_csv_columns(file.getvalue())
            
            dataset_data = {
                "ds_name": name,
                "ds_size": file.size,
                "media": media_type,
                "task": task_type,
                "columns": columns
            }
            
            db_api.db_create_dataset(name, dataset_data)
            st.success("数据集上传成功！")
            return True
        except Exception as e:
            st.error(f"上传失败：{str(e)}")
            return False