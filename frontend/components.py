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
from database.database_schema import ArchType, Media_type, Task_name, Trainname

# 允许嵌套事件循环
nest_asyncio.apply()

class Sidebar:
    def __init__(self, auth_manager):
        self.auth_manager = auth_manager

    def render(self):
        """渲染侧边栏"""
        with st.sidebar:
            st.title("OpenModelHub")
            if not self.auth_manager.is_authenticated():
                self._render_login_form()
            else:
                self._render_user_info()
            return self._render_navigation()

    def _render_login_form(self):
        """渲染登录表单"""
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
        """渲染用户信息"""
        user = self.auth_manager.get_current_user()
        st.success(f"欢迎，{user['username']}！")
        if st.button("退出登录"):
            self.auth_manager.logout()
            st.rerun()

    def _render_navigation(self):
        """渲染导航菜单"""
        # 主页始终显示
        menu_items = ["主页"]
        
        # 其他菜单项需要登录
        if self.auth_manager.is_authenticated():
            menu_items.extend(["模型仓库", "数据集", "用户管理"])
            if self.auth_manager.is_admin():
                menu_items.append("系统管理")
                
        return st.radio("导航菜单", menu_items)

class UserManager:
    def __init__(self):
        self.users = db_api.db_list_users()

    def render(self):
        """渲染用户管理界面"""
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
    def __init__(self):
        self.allowed_types = UPLOAD_CONFIG["allowed_types"]
        self.max_size = UPLOAD_CONFIG["max_size"]

    def render(self):
        """渲染数据集上传组件"""
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
        """处理表单提交"""
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
        
class ModelUploader:
    def __init__(self):
        self.allowed_types = ["pt", "pth", "ckpt", "bin","txt"]  # Model file types
        self.max_size = UPLOAD_CONFIG["max_size"]

    def render(self):
        """渲染模型上传组件"""
        with st.expander("📤 上传新模型", expanded=False):
            with st.form("model_upload", clear_on_submit=True):
                # Basic Information
                name = st.text_input("模型名称*")
                param_num = st.number_input("参数量", min_value=1000, value=1000000)
                
                # Model Architecture
                arch_type = st.selectbox(
                    "架构类型*", 
                    options=[arch.value for arch in ArchType]
                )
                
                # Media and Task Types
                media_type = st.selectbox(
                    "媒体类型*",
                    options=[media.value for media in Media_type]
                )
                
                tasks = st.multiselect(
                    "任务类型*",
                    options=[task.value for task in Task_name]
                )
                
                train_type = st.selectbox(
                    "训练类型*",
                    options=[train.value for train in Trainname]
                )
                
                # File Upload
                model_file = st.file_uploader("选择模型文件*", type=self.allowed_types)
                
                if st.form_submit_button("提交"):
                    return self._handle_submit(
                        name=name,
                        param_num=param_num,
                        arch_type=arch_type,
                        media_type=media_type,
                        tasks=tasks,
                        train_type=train_type,
                        file=model_file
                    )
        return False

    def _handle_submit(self, name, param_num, arch_type, media_type, tasks, train_type, file):
        """处理表单提交"""
        if not all([name, arch_type, media_type, tasks, file]):
            st.error("带*的字段为必填项")
            return False

        is_valid, error_msg = validate_file_upload(file, self.allowed_types, self.max_size)
        if not is_valid:
            st.error(error_msg)
            return False

        try:
            file_path = db_api.db_save_file(file.getvalue(), file.name)
            
            model_data = {
                "model_name": name,
                "param_num": param_num,
                "arch_name": arch_type,
                "media_type": media_type,
                "tasks": tasks,
                "trainname": train_type,
                "param": file_path
            }
            
            db_api.db_create_model(model_data)
            st.success("模型上传成功！")
            return True
            
        except Exception as e:
            st.error(f"上传失败：{str(e)}")
            return False

def create_search_section(type: str = None):
    """创建搜索区域
    Args:
        type: 搜索类型，可以是 "models" 或 "datasets"
    """
    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input("搜索", placeholder="输入关键词搜索...")
        with col2:
            if type:
                search_type = type
            else:
                search_type = st.selectbox("类型", ["全部", "模型", "数据集"])
        
        if search_query:
            # 根据搜索类型和关键词进行搜索
            if search_type == "全部" or search_type == "models" or search_type == "模型":
                models = db_api.db_search_models(search_query)
                if models:
                    st.subheader("模型搜索结果")
                    for model in models:
                        with st.container(border=True):
                            st.write(f"### {model.model_name}")
                            st.caption(f"架构: {model.arch_name.value} | 媒体类型: {model.media_type.value}")
            
            if search_type == "全部" or search_type == "datasets" or search_type == "数据集":
                datasets = db_api.db_search_datasets(search_query)
                if datasets:
                    st.subheader("数据集搜索结果")
                    for dataset in datasets:
                        with st.container(border=True):
                            st.write(f"### {dataset.ds_name}")
                            st.caption(f"类型: {dataset.media} | 大小: {dataset.ds_size/1024:.1f}KB")