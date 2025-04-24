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

# 创建全局搜索框和类型查询下拉框
def create_search_section(search_key: str):
    entity_types = ["全部", "模型", "数据集", "用户", "机构"]
    
    entity_dict = {
        "全部": 0,
        "模型": 1,
        "数据集": 2,
        "用户": 3,
        "机构": 4
    }
    
    st.markdown("""
        <style>
        .stButton > button {
            margin-top: 25px;  /* Adjust this value to match your input height */
        }
        div.row-widget.stSelectbox {
            margin-top: 25px;  /* Match the button margin */
        }
        </style>
    """, unsafe_allow_html=True)
    
    with st.container():
        col1, col2, col3 = st.columns([4.2, 0.5, 0.3])
        with col1:
            query = st.text_input("搜索", placeholder="输入自然语言查询", key=f"search_input_{search_key}")
        with col2:
            search_type = st.selectbox(
                "搜索类型",
                entity_types,
                key=f"search_type_{search_key}"
            )
        with col3:
            search_clicked = st.button(
                "搜索", 
                key=f"search_button_{search_key}",
                use_container_width=True
            )
    
    if search_clicked and query:
        # 添加类型信息到查询
        if search_type != "全部":
            query = f"搜索{search_type}：{query}"
        instance_type = entity_dict[search_type]
        print(instance_type)
        results, query_info = db_api.db_agent_query(query, instance_type)
        # 显示查询详情
        with st.expander("查询详情"):
            st.json({
                'natural_language_query': query_info['natural_language_query'],
                'generated_sql': query_info['generated_sql'],
                'error_code': query_info['error_code'],
                'has_results': query_info['has_results'],
                'error': query_info.get('error', None),
                'sql_res': results
            })
        if results:
            df = pd.DataFrame(results)
            st.dataframe(df)
            return True
    return False

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
        menu_items = ["主页", "模型仓库", "数据集", "用户管理"]
        if self.auth_manager.is_admin():
            menu_items += ["系统管理"]
        return st.radio("导航菜单", menu_items)

class UserManager:
    def __init__(self):
        self.users = db_api.db_list_users()

    def render(self):
        """渲染用户管理界面"""
        st.header("👥 用户管理")
        
        # 使用统一的搜索部分
        if create_search_section("users"):
            return
        
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
            file_path = db_api.db_save_file(file.getvalue(), file.name, file_type="datasets")
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
            file_path = db_api.db_save_file(file.getvalue(), file.name, file_type="models")
            
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