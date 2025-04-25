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

# Allow nested event loops
nest_asyncio.apply()

# Create global search bar and type query dropdown
def create_search_section(search_key: str, search_type = 0):
    entity_types = ["All", "Model", "Dataset", "User", "Organization"]
    
    entity_dict = {
        "All": 0,
        "Model": 1,
        "Dataset": 2,
        "User": 3,
        "Organization": 4
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
        col1, col2, col3 = st.columns([3.9, 0.7, 0.5])
        with col1:
            query = st.text_input("Search", placeholder="Enter natural language query", key=f"search_input_{search_key}")
        with col2:
            search_type = st.selectbox(
                "Search Type",
                entity_types,
                index = search_type,
            )
        with col3:
            search_clicked = st.button(
                "Search", 
                key=f"search_button_{search_key}",
                use_container_width=True
            )
    
    if search_clicked and query:
        # Add type information to query
        if search_type != "All":
            query = f"Search {search_type}: {query}"
        instance_type = entity_dict[search_type]
        print(instance_type)
        results, query_info = db_api.db_agent_query(query, instance_type)
        # Display query details
        with st.expander("Query Details"):
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
        """Render sidebar"""
        with st.sidebar:
            st.title("Open Model Hub")
            if not self.auth_manager.is_authenticated():
                self._render_login_form()
            else:
                self._render_user_info()
            return self._render_navigation()

    def _render_login_form(self):
        """Render login form"""
        with st.form("Login", clear_on_submit=True):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                # Use current event loop
                loop = asyncio.get_event_loop()
                if loop.run_until_complete(self.auth_manager.login(username, password)):
                    st.rerun()
                else:
                    st.error("Incorrect username or password")

    def _render_user_info(self):
        """Render user information"""
        user = self.auth_manager.get_current_user()
        st.success(f"Welcome, {user['username']}!")
        if st.button("Log Out"):
            self.auth_manager.logout()
            st.rerun()

    def _render_navigation(self):
        """Render navigation menu"""
        menu_items = ["Home", "Model Repository", "Datasets", "User Management"]
        if self.auth_manager.is_admin():
            menu_items += ["System Management"]
        return st.radio("Navigation Menu", menu_items)

class UserManager:
    def __init__(self):
        self.users = db_api.db_list_users()

    def render(self):
        """Render user management interface"""
        st.header("👥 User Management")
        
        # Use unified search section
        if create_search_section("users", 3):
            return
        
        # Create user form
        with st.expander("➕ Add New User", expanded=False):
            with st.form("new_user", clear_on_submit=True):
                username = st.text_input("Username*")
                password = st.text_input("Password*", type="password")
                is_admin = st.checkbox("Admin Privileges")
                affiliate = st.text_input("Organization")
                
                if st.form_submit_button("Create User"):
                    if not username or not password:
                        st.error("Fields marked with * are required")
                    else:
                        try:
                            # Check if username already exists
                            existing_user = db_api.db_get_user_by_username(username)
                            if existing_user:
                                st.error(f"Username '{username}' already exists")
                            else:
                                # Create new user
                                db_api.db_create_user(username, password, affiliate, is_admin=is_admin)
                                st.success("User created successfully")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Creation failed: {str(e)}")
        
        # User list
        df = pd.DataFrame([{
            "ID": user.user_id,
            "Username": user.user_name,
            "Organization": user.affiliate,
            "Admin": "✅" if user.is_admin else "❌"
        } for user in self.users])
        
        st.dataframe(
            df,
            column_config={
                "ID": "User ID",
                "Admin": st.column_config.CheckboxColumn("Admin Status")
            },
            use_container_width=True,
            hide_index=True
        )

class DatasetUploader:
    def __init__(self):
        self.allowed_types = UPLOAD_CONFIG["allowed_types"]
        self.max_size = UPLOAD_CONFIG["max_size"]

    def render(self):
        """Render dataset upload component"""
        with st.expander("📤 Upload New Dataset", expanded=False):
            with st.form("dataset_upload", clear_on_submit=True):
                name = st.text_input("Dataset Name*")
                desc = st.text_area("Description")
                media_type = st.selectbox("Media Type", ["text", "image", "audio", "video"])
                task_type = st.selectbox("Task Type", ["classification", "detection", "generation"])
                file = st.file_uploader("Select Data File*", type=self.allowed_types)
                
                if st.form_submit_button("Submit"):
                    return self._handle_submit(name, desc, media_type, task_type, file)
        return False

    def _handle_submit(self, name: str, desc: str, media_type: str, task_type: str, file):
        """Handle form submission"""
        if not name or not file:
            st.error("Fields marked with * are required")
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
            st.success("Dataset uploaded successfully!")
            return True
        except Exception as e:
            st.error(f"Upload failed: {str(e)}")
            return False 
        
class ModelUploader:
    def __init__(self):
        self.allowed_types = ["pt", "pth", "ckpt", "bin","txt"]  # Model file types
        self.max_size = UPLOAD_CONFIG["max_size"]

    def render(self):
        """Render model upload component"""
        with st.expander("📤 Upload New Model", expanded=False):
            with st.form("model_upload", clear_on_submit=True):
                # Basic Information
                name = st.text_input("Model Name*")
                param_num = st.number_input("Parameter Count", min_value=1000, value=1000000)
                
                # Model Architecture
                arch_type = st.selectbox(
                    "Architecture Type*", 
                    options=[arch.value for arch in ArchType]
                )
                
                # Media and Task Types
                media_type = st.selectbox(
                    "Media Type*",
                    options=[media.value for media in Media_type]
                )
                
                tasks = st.multiselect(
                    "Task Types*",
                    options=[task.value for task in Task_name]
                )
                
                train_type = st.selectbox(
                    "Training Type*",
                    options=[train.value for train in Trainname]
                )
                
                # File Upload
                model_file = st.file_uploader("Select Model File*", type=self.allowed_types)
                
                if st.form_submit_button("Submit"):
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
        """Handle form submission"""
        if not all([name, arch_type, media_type, tasks, file]):
            st.error("Fields marked with * are required")
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
            st.success("Model uploaded successfully!")
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
