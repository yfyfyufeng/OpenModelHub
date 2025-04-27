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
from frontend.db import get_db_session
from database.database_schema import ArchType, Media_type, Task_name, Trainname
from database.database_interface import get_model_ids_by_attribute, get_dataset_ids_by_attribute, get_user_ids_by_attribute

# Allow nested event loops
nest_asyncio.apply()

def create_search_section(page_type: str = "all"):
    """Create unified search section for different pages, supporting Enter key search"""
    entity_types = ["all", "models", "datasets", "users"]
    entity_dict = {
        "all": 0,
        "models": 1,
        "datasets": 2,
        "users": 3
    }

    st.markdown("""
        <style>
        .stButton > button {
            margin-top: 25px;
        }
        div.row-widget.stSelectbox {
            margin-top: 25px;
        }
        </style>
    """, unsafe_allow_html=True)

    default_index = entity_dict.get(page_type, 0)

    with st.container():
        with st.form(f"search_form_{page_type}"):
            # 第一行
            col1, col2, col3 = st.columns([3.9, 0.7, 0.5])
            with col1:
                search_query = st.text_input(
                    f"Search {page_type.capitalize() if page_type else ''}", 
                    placeholder="Enter natural language query",
                    key=f"search_input_{page_type}"
                )
            with col2:
                selected_type = st.selectbox(
                    "Select Type",
                    entity_types,
                    key=f"type_select_{page_type}",
                    index=default_index,
                )
            with col3:
                pass  # 空着，按钮放最后统一处理

            # 第二行
            if selected_type == "models":
                field_options = ["model_id", "model_name", "param_num", "media_type", "arch_name", "trainname"]
            elif selected_type == "datasets":
                field_options = ["ds_id", "ds_name", "ds_size", "media", "created_at"]
            elif selected_type == "users":
                field_options = ["user_id", "user_name", "email", "role"]
            else:
                field_options = ["id", "name", "created_at"]

            field_col1, field_col2, field_col3 = st.columns([0.7, 0.7, 0.5])
            with field_col1:
                field_attr = st.selectbox(
                    "Select Field",
                    options=field_options,
                    key=f"field_select_{selected_type}",
                )
            with field_col2:
                field_val = st.text_input(
                    "Enter Query Value",
                    key=f"field_input_{selected_type}",
                )
            with field_col3:
                pass

            # ✅ 最后统一的提交按钮
            submit_clicked = st.form_submit_button("Search", use_container_width=True)

    if submit_clicked:
        try:
            if field_val:  # 字段搜索
                if selected_type == "models":
                    ids = asyncio.run(get_model_ids_by_attribute(get_db_session()(), field_attr, field_val))
                elif selected_type == "datasets":
                    ids = asyncio.run(get_dataset_ids_by_attribute(get_db_session()(), field_attr, field_val))
                elif selected_type == "users":
                    ids = asyncio.run(get_user_ids_by_attribute(get_db_session()(), field_attr, field_val))
                else:
                    model_ids = asyncio.run(get_model_ids_by_attribute(get_db_session()(), field_attr, field_val))
                    dataset_ids = asyncio.run(get_dataset_ids_by_attribute(get_db_session()(), field_attr, field_val))
                    user_ids = asyncio.run(get_user_ids_by_attribute(get_db_session()(), field_attr, field_val))
                    ids = model_ids + dataset_ids + user_ids

                if not ids:
                    st.info(f"No {page_type} found matching the criteria")
                else:
                    st.session_state.filtered_ids = ids
                return True

            elif search_query:  # 自然语言搜索
                results, query_info = db_api.db_agent_query(search_query, instance_type=entity_dict.get(selected_type, 0))
                if results:
                    df = pd.DataFrame(results)
                    st.dataframe(df)

                    st.markdown("---")
                    st.subheader("Query Details")
                    st.json({
                        'Natural Language Query': query_info['natural_language_query'],
                        'Generated SQL': query_info['generated_sql'],
                        'Error Code': query_info['error_code'],
                        'Has Results': query_info['has_results'],
                        'Error Message': query_info.get('error', None),
                        'Query Results': results
                    })
                    return True
                else:
                    st.info("No results found")
                    return True

            else:
                st.warning("Please enter a search query or field value")
                return False

        except Exception as e:
            st.error(f"Search failed: {str(e)}")
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
            # menu_items.append("System Management")
            menu_items.append("data insight")
        return st.radio("Navigation Menu", menu_items)

class UserManager:
    def __init__(self):
        self.users = db_api.db_list_users()

    def render(self):
        """Render user management interface"""
        st.header("👥 User Management")
        
        # Use unified search section and handle search results
        if create_search_section("users"):
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
            st.error(f"Upload failed: {str(e)}")
            return False
