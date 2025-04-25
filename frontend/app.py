# main.py
#streamlit run ./frontend/app.py [ARGUMENTS]
#(replace by your absolute path)
import streamlit as st
from pathlib import Path
import pandas as pd
import sys
import asyncio
from datetime import datetime
import nest_asyncio
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.extend([str(project_root), str(project_root/"database")])
sys.path.extend([str(project_root), str(project_root/"frontend")])
import frontend.database_api as db_api
from database.database_interface import (
     get_model_by_id, list_datasets, get_dataset_by_id,
    list_users, get_user_by_id, list_affiliations, init_database,
    create_user, update_user, delete_user, get_dataset_info, get_model_info,
    get_model_ids_by_attribute, get_dataset_ids_by_attribute, get_user_ids_by_attribute
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select
import os
from dotenv import load_dotenv
import hashlib
from typing import List, Dict
from io import BytesIO
from frontend.config import APP_CONFIG
from frontend.db import get_db_session
from frontend.auth import AuthManager
from frontend.components import Sidebar, DatasetUploader, UserManager, ModelUploader, create_search_section


# Allow nested event loops
nest_asyncio.apply()

# Initialize page configuration (must be at the beginning)
st.set_page_config(**APP_CONFIG)

def create_pagination(items, type, page_size=10, page_key="default"):
    """创建分页
    Args:
        items: 数据项列表，可以是对象或字典
        type: 数据类型，如 "models" 或 "datasets"
        page_size: 每页显示的数量
        page_key: 分页状态的唯一标识
    """
    # 获取当前页码
    page_state_key = f'current_page_num_{page_key}'
    if page_state_key not in st.session_state:
        st.session_state[page_state_key] = 1
        
    # 计算总页数
    total_pages = (len(items) + page_size - 1) // page_size
    
    # 获取当前页的项目
    start_idx = (st.session_state[page_state_key] - 1) * page_size
    end_idx = min(start_idx + page_size, len(items))
    current_items = items[start_idx:end_idx]
    
    # 显示当前页的项目
    for idx, item in enumerate(current_items):
        with st.container(border=True):
            col1, col2 = st.columns([5, 0.7])
            with col1:
                # 处理对象和字典两种情况
                if isinstance(item, dict):
                    name = item.get('ds_name' if type == 'datasets' else 'model_name', '')
                    if type == "models":
                        st.write(f"### {name}")
                        st.caption(f"架构: {item.get('arch_name', '')} | 媒体类型: {item.get('media_type', '')} | 参数量: {item.get('param_num', 0):,}")
                    else:
                        st.write(f"### {name}")
                        tasks = item.get('task', [])
                        st.caption(f"类型: {item.get('media', '')} | 任务: {', '.join(tasks)} | 大小: {item.get('ds_size', 0)/1024:.1f}KB")
                else:
                    st.write(f"### {item.model_name if type=='models' else item.ds_name}")
                    if type == "models":
                        st.caption(f"架构: {item.arch_name.value} | 媒体类型: {item.media_type.value} | 参数量: {item.param_num:,}")
                    else:
                        tasks = [task.task.value for task in item.Dataset_TASK] if hasattr(item, 'Dataset_TASK') else []
                        st.caption(f"类型: {item.media} | 任务: {', '.join(tasks)} | 大小: {item.ds_size/1024:.1f}KB")
            
            with col2:
                item_id = item.get('ds_id' if type == 'datasets' else 'model_id', 0) if isinstance(item, dict) else (item.model_id if type == 'models' else item.ds_id)
                if st.button("查看详情", key=f"{type[:-1]}_{item_id}_{idx}", use_container_width=True):
                    st.session_state[f"selected_{type[:-1]}"] = item
                    st.session_state.current_page = f"{type[:-1]}_detail"
                    st.rerun()
    
    # 分页控制
    _, _, col3 = st.columns([10, 10, 3.5])
    with col3:
        col1, col2, col3 = st.columns([1.8, 1, 1])
        with col1:
            st.button(f'{st.session_state[page_state_key]}/{total_pages}', disabled=True, key=f"page_num_{page_key}")
        with col2:
            if st.button("←", key=f"prev_{page_key}"):
                st.session_state[page_state_key] = max(1, st.session_state[page_state_key] - 1)
                st.rerun()
        with col3:
            if st.button("→", key=f"next_{page_key}"):
                st.session_state[page_state_key] = min(total_pages, st.session_state[page_state_key] + 1)
                st.rerun()

def parse_csv_columns(file_data: bytes) -> List[Dict]:
    df = pd.read_csv(BytesIO(file_data), nrows=1)
    return [{"col_name": col, "col_datatype": "text"} for col in df.columns]

# Async execution decorator
def async_to_sync(async_func):
    def wrapper(*args, **kwargs):
        return asyncio.run(async_func(*args, **kwargs))
    return wrapper

# Database session management
@st.cache_resource
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

# User authentication state management
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None

# File upload handling
def handle_file_upload():
    with st.expander("Upload New Dataset"):
        with st.form("dataset_upload"):
            name = st.text_input("Dataset Name")
            desc = st.text_area("Description")
            file = st.file_uploader("Select Data File", type=["csv", "txt"])
            if st.form_submit_button("Submit"):
                if file:
                    file_path = db_api.db_save_file(file.getvalue(), file.name)
                    db_api.db_create_dataset(name, desc, file_path)
                    st.success("Dataset uploaded successfully!")
                else:
                    st.error("Please select a file")

# File download handling
def handle_file_download(dataset):
    file_data = db_api._file(dataset.ds_name + ".zip")
    if file_data:
        st.download_button(
            label="Download",
            data=file_data,
            file_name=f"{dataset.ds_name}.zip",
            key=f"download_{dataset.ds_id}"
        )
    else:
        st.error("File does not exist")
        
def download_model(model):
    """Handle model download"""
    file_data = db_api.db_get_file(f"{model.model_name}.pt", file_type="models")
    if file_data:
        st.download_button(
            label="Download Model",
            data=file_data,
            file_name=f"{model.model_name}.pt",
            mime="application/octet-stream",
            key=f"download_model_{model.model_id}"
        )
    else:
        st.error("Model file does not exist")

def download_dataset(dataset):
    """Handle dataset download"""
    # Try different possible file extensions
    for ext in ['.txt', '.csv', '.zip']:
        file_data = db_api.db_get_file(f"{dataset.ds_name}{ext}", file_type="datasets")
        if file_data:
            st.download_button(
                label="Download Dataset",
                data=file_data,
                file_name=f"{dataset.ds_name}{ext}",
                mime="text/plain" if ext in ['.txt', '.csv'] else "application/zip",
                key=f"download_dataset_{dataset.ds_id}"
            )
            return True
    
    st.error(f"Dataset file {dataset.ds_name}.* does not exist")
    return False
        
# Login form
def login_form():
    with st.form("Login", clear_on_submit=True):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        use_encryption = st.checkbox("Use Encrypted Login", value=True)
        if st.form_submit_button("Login"):
            # Use hashed password for verification (example uses sha256, production should use bcrypt)
            #      hashed_pwd = hashlib.sha256(password.encode()).hexdigest()
            hashed_pwd = password  # Use plaintext password directly
            if use_encryption:
                user = db_api.db_authenticate_user(username, hashed_pwd)
            else:
                # Non-encrypted login
                user = db_api.db_get_user_by_username(username)
                if user and user.password_hash == password:
                    user = user
                else:
                    user = None

            if user:
                st.session_state.authenticated = True
                st.session_state.current_user = {
                    "user_id": user.user_id,
                    "username": user.user_name,
                    "role": "admin" if user.is_admin else "user"
                }
                st.rerun()
            else:
                st.error("Username or password incorrect")

# Sidebar navigation
def sidebar():
    with st.sidebar:
        st.title("OpenModelHub")
        if not st.session_state.authenticated:
            login_form()
        else:
            st.success(f"Welcome, {st.session_state.current_user['username']}!")
            if st.button("Logout"):
                st.session_state.authenticated = False
                st.session_state.current_user = None
                st.rerun()
            
        menu_items = ["Home", "Model Repository", "Datasets", "User Management"]
        if st.session_state.current_user and st.session_state.current_user["role"] == "admin":
            menu_items += ["System Management"]
            
        return st.radio("Navigation Menu", menu_items)

# Home page
def render_home():
    """Render home page"""
    st.header("Platform Overview")
    
    # Direct call to database API
    models = db_api.db_list_models()
    datasets = db_api.db_list_datasets()
    users = db_api.db_list_users()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Models", len(models))
    with col2:
        st.metric("Total Datasets", len(datasets))
    with col3:
        st.metric("Registered Users", len(users))
    with col4:
        st.metric("Today's Downloads", 2543)

# Pages
def create_pagination(items, type, page_size=10, page_key="default"):
    """Create pagination for items"""
    # Get current page from session state
    page_state_key = f'current_page_num_{page_key}'
    if page_state_key not in st.session_state:
        st.session_state[page_state_key] = 1
        
    # Calculate total pages
    total_pages = (len(items) + page_size - 1) // page_size
    
    # Get items for current page
    start_idx = (st.session_state[page_state_key] - 1) * page_size
    end_idx = min(start_idx + page_size, len(items))
    current_items = items[start_idx:end_idx]
    
    if (type == "models"):
        # Display model information (sorted by creation time in descending order)
        for model in current_items:
            with st.container(border=True):
                col1, col2 = st.columns([5, 0.7])
                with col1:
                    # Put title and button in the same line
                    st.write(
                        f"### {model.model_name} ",
                        unsafe_allow_html=True
                    )
                    # Get model tasks
                    tasks = [task.task_name.value for task in model.tasks] if hasattr(model, 'tasks') else []
                    task_str = ", ".join(tasks) if tasks else "No tasks"
                    st.caption(f"Architecture: {model.arch_name.value} | Media Type: {model.media_type.value} | Parameters: {model.param_num:,}")
                
                with col2:
                    # Hide the actual button but keep functionality
                    if st.button("View Details", key=f"model_{model.model_id}", use_container_width=True):
                        st.session_state.selected_model = model
                        st.session_state.current_page = "model_detail"
                        
    elif (type == "datasets"):
        # Display dataset information (sorted by creation time in descending order)
        for dataset in current_items:
            with st.container(border=True):
                col1, col2 = st.columns([5, 0.7])
                with col1:
                    # Put title and button in the same line
                    st.write(
                        f"### {dataset.ds_name}",
                        unsafe_allow_html=True
                    )
                    # Get dataset tasks
                    tasks = [task.task.value for task in dataset.Dataset_TASK]
                    task_str = ", ".join(tasks) if tasks else "No tasks"
                    st.caption(
                        f"Type: {dataset.media} | "
                        f"Tasks: {task_str} | "
                        f"Size: {dataset.ds_size/1024:.1f}KB"
                    )
                with col2:
                    if st.button("View Details", key=f"dataset_{dataset.ds_id}", use_container_width=True):
                        st.session_state.selected_dataset = dataset
                        st.session_state.current_page = "dataset_detail"
    
    # Create pagination controls on the right
    _, _, col3 = st.columns([10, 10, 3.5])
    
    with col3:
        # Use columns for layout
        col1, col2, col3 = st.columns([1.8, 1, 1])
        
        # Display page numbers
        with col1:
            st.button(f'{st.session_state[page_state_key]}/{total_pages}', disabled=True, key=f"page_num_{page_key}")
            
        # Previous page button
        with col2:
            if st.button("←", key=f"prev_{page_key}") and st.session_state[page_state_key] > 1:
                st.session_state[page_state_key] -= 1
                st.rerun()
        
        # Next page button
        with col3:
            if st.button("→", key=f"next_{page_key}") and st.session_state[page_state_key] < total_pages:
                st.session_state[page_state_key] += 1
                st.rerun()
    
    st.write("")

# Model repository
def create_unified_search_section(page_type: str = None):
    """Create unified search section for different pages
    Args:
        page_type: Type of page (models, datasets, users)
    Returns:
        bool: True if search was performed, False otherwise
    """
    # Search section
    with st.container():
        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            search_query = st.text_input(f"Search {page_type.capitalize() if page_type else ''}", 
                                       placeholder="Enter natural language query")
        with col2:
            # Add type selection dropdown
            selected_type = st.selectbox(
                "Select Type",
                ["Models", "Datasets", "Users"],
                key=f"type_select_{page_type}",
                index=1 if page_type == "datasets" else 0
            )
            if selected_type == "Models" and page_type != "models":
                st.session_state.current_page = "models"
                st.rerun()
            elif selected_type == "Datasets" and page_type != "datasets":
                st.session_state.current_page = "datasets"
                st.rerun()
            elif selected_type == "Users" and page_type != "users":
                st.session_state.current_page = "users"
                st.rerun()
            
            # Get appropriate field options based on page type
            if page_type == "models":
                field_options = ["model_id", "model_name", "param_num", "media_type", "arch_name", "trainname"]
            elif page_type == "datasets":
                field_options = ["ds_id", "ds_name", "ds_size", "media", "created_at"]
            else:
                field_options = ["user_id", "user_name", "email", "role"]
            
            field_attr = st.selectbox(
                "Select Field",
                field_options
            )
            field_val = st.text_input("Enter Query Value")
        with col3:
            search_clicked = st.button("Search", key=f"{page_type}_search", use_container_width=True)
    
    # Handle search logic
    if search_clicked:
        if search_query:  # Natural language search
            instance_type = 1 if page_type == "models" else (2 if page_type == "datasets" else 3)
            results, query_info = db_api.db_agent_query(search_query, instance_type=instance_type)
            with st.expander("Query Details"):
                st.json({
                    'Natural Language Query': query_info['natural_language_query'],
                    'Generated SQL': query_info['generated_sql'],
                    'Error Code': query_info['error_code'],
                    'Has Results': query_info['has_results'],
                    'Error Message': query_info.get('error', None),
                    'Query Results': results
                })
            if results:
                df = pd.DataFrame(results)
                st.dataframe(df)
                return True
        else:  # Attribute search
            try:
                # Use asyncio.run to create new event loop
                if page_type == "models":
                    ids = asyncio.run(get_model_ids_by_attribute(get_db_session()(), field_attr, field_val))
                elif page_type == "datasets":
                    ids = asyncio.run(get_dataset_ids_by_attribute(get_db_session()(), field_attr, field_val))
                else:
                    ids = asyncio.run(get_user_ids_by_attribute(get_db_session()(), field_attr, field_val))
                
                if not ids:
                    st.info(f"No {page_type} found matching the criteria")
                else:
                    st.session_state.filtered_ids = ids
                return True
            except Exception as e:
                st.error(f"Search failed: {str(e)}")
                return True
    
    return False

def render_models():
    """Render model repository page"""
    st.title("Model Repository")
    
    # Use unified search section
    if create_unified_search_section("models"):
        return
    
    # Model upload
    with st.expander("Upload New Model", expanded=False):
        with st.form("model_upload", clear_on_submit=True):
            name = st.text_input("Model Name*")
            param_num = st.number_input("Parameter Count", min_value=1000, value=1000000)
            
            col1, col2 = st.columns(2)
            with col1:
                arch_type = st.selectbox(
                    "Architecture Type*",
                    ["CNN", "RNN", "TRANSFORMER"],
                    help="Select model architecture type"
                )
                media_type = st.selectbox(
                    "Media Type*",
                    ["TEXT", "IMAGE", "AUDIO", "VIDEO"],
                    help="Select applicable media type"
                )
            with col2:
                train_type = st.selectbox(
                    "Training Type*",
                    ["PRETRAIN", "FINETUNE", "RL"],
                    help="Select training type"
                )
                selected_tasks = st.multiselect(
                    "Task Types*",
                    ["CLASSIFICATION", "DETECTION", "GENERATION", "SEGMENTATION"],
                    default=["CLASSIFICATION"]
                )
            
            file = st.file_uploader("Select Model File*", type=["pt", "pth", "ckpt"])
            
            if st.form_submit_button("Submit"):
                if not name or not file:
                    st.error("Fields marked with * are required")
                else:
                    try:
                        file_path = db_api.db_save_file(file.getvalue(), file.name)
                        model_data = {
                            "model_name": name,
                            "param_num": param_num,
                            "arch_name": arch_type,
                            "media_type": media_type,
                            "tasks": selected_tasks,
                            "trainname": train_type,
                            "param": str(file_path)
                        }
                        db_api.db_create_model(model_data)
                        st.success("Model uploaded successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Upload failed: {str(e)}")
    
    # Get all models
    models = db_api.db_list_models()
    
    if not models:
        st.info("No models available")
        return
    
    # Display model list
    create_pagination(models, "models")
    
    # Display model details
    if st.session_state.get("current_page") == "model_detail":
        model = st.session_state.get("selected_model")
        if model:
            st.markdown("---")
            st.subheader(f"Model Details - {model.model_name if hasattr(model, 'model_name') else model.get('model_name', '')}")
            
            # Display basic information
            st.write("**Basic Information:**")
            info_data = {
                "Model ID": model.model_id if hasattr(model, 'model_id') else model.get('model_id', ''),
                "Architecture Type": model.arch_name.value if hasattr(model, 'arch_name') else model.get('arch_name', ''),
                "Parameter Count": f"{model.param_num:,}" if hasattr(model, 'param_num') else f"{model.get('param_num', 0):,}",
                "Training Type": model.trainname.value if hasattr(model, 'trainname') else model.get('trainname', '')
            }
            st.table(pd.DataFrame(list(info_data.items()), columns=["Property", "Value"]))
            
            # Display related information
            st.write("**Related Information:**")
            if hasattr(model, 'tasks'):
                tasks = [task.task_name.value for task in model.tasks]
            else:
                tasks = model.get('tasks', [])
            
            if hasattr(model, 'authors'):
                authors = [author.user_name for author in model.authors]
            else:
                authors = model.get('authors', [])
            
            if hasattr(model, 'datasets'):
                datasets = [dataset.ds_name for dataset in model.datasets]
            else:
                datasets = model.get('datasets', [])
            
            rel_data = {
                "Supported Tasks": ", ".join(tasks),
                "Authors": ", ".join(authors),
                "Related Datasets": ", ".join(datasets)
            }
            st.table(pd.DataFrame(list(rel_data.items()), columns=["Type", "Name"]))
            
            # Download button
            if st.button("Download Model", key=f"download_{model.model_id if hasattr(model, 'model_id') else model.get('model_id', '')}"):
                file_data = db_api.db_get_file(model.model_name if hasattr(model, 'model_name') else model.get('model_name', '') + ".pt")
                if file_data:
                    st.download_button(
                        label="Click to Download",
                        data=file_data,
                        file_name=f"{model.model_name if hasattr(model, 'model_name') else model.get('model_name', '')}.pt",
                        mime="application/octet-stream"
                    )
                else:
                    st.error("File does not exist")
            
            # Return button
            if st.button("Back to List", key="back_to_list"):
                st.session_state.current_page = "models"
                st.rerun()

def render_datasets():
    """Render dataset management page"""
    st.title("Dataset Management")
    
    # Use unified search section
    if create_unified_search_section("datasets"):
        return
    
    # Dataset upload
    with st.expander("Upload New Dataset", expanded=False):
        with st.form("dataset_upload", clear_on_submit=True):
            name = st.text_input("Dataset Name*")
            desc = st.text_area("Description")
            media_type = st.selectbox("Media Type", ["text", "image", "audio", "video"])
            task_type = st.selectbox("Task Type", ["classification", "detection", "generation"])
            file = st.file_uploader("Select Data File*", type=["txt", "csv"])
            
            if st.form_submit_button("Submit"):
                if not name or not file:
                    st.error("Fields marked with * are required")
                else:
                    try:
                        # Save file
                        file_path = db_api.db_save_file(file.getvalue(), file.name)
                        
                        # Create dataset
                        dataset_data = {
                            "ds_name": name,
                            "ds_size": len(file.getvalue()),
                            "media": media_type,
                            "task": [task_type],
                            "columns": [
                                {"col_name": "content", "col_datatype": "text"}
                            ],
                            "description": desc
                        }
                        db_api.db_create_dataset(name, dataset_data)
                        st.success("Dataset uploaded successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Upload failed: {str(e)}")
    
    # Get all datasets
    datasets = db_api.db_list_datasets()
    
    if not datasets:
        st.info("No datasets available")
        return
    
    # Display dataset list
    create_pagination(datasets, "datasets")
    
    # Display dataset details
    if st.session_state.get("current_page") == "dataset_detail":
        dataset = st.session_state.get("selected_dataset")
        if dataset:
            st.markdown("---")
            st.subheader(f"Dataset Details - {dataset.ds_name if hasattr(dataset, 'ds_name') else dataset.get('ds_name', '')}")
            
            # Display description
            st.write("**Description:**")
            st.write(dataset.description if hasattr(dataset, 'description') else dataset.get('description', 'No description available'))
            
            # Display task information
            st.write("**Task Types:**")
            if hasattr(dataset, 'Dataset_TASK'):
                tasks = [task.task.value for task in dataset.Dataset_TASK]
            else:
                tasks = dataset.get('task', [])
            st.write(", ".join(tasks) if tasks else "No tasks")
            
            # Display dataset size
            st.write("**Dataset Size:**")
            size = dataset.ds_size if hasattr(dataset, 'ds_size') else dataset.get('ds_size', 0)
            st.write(f"{size/1024:.1f}KB")
            
            # Download button
            if st.button("Download Dataset", key=f"download_{dataset.ds_id if hasattr(dataset, 'ds_id') else dataset.get('ds_id', '')}"):
                file_data = db_api.db_get_file(dataset.ds_name if hasattr(dataset, 'ds_name') else dataset.get('ds_name', '') + ".txt")
                if file_data:
                    st.download_button(
                        label="Click to Download",
                        data=file_data,
                        file_name=f"{dataset.ds_name if hasattr(dataset, 'ds_name') else dataset.get('ds_name', '')}.txt",
                        mime="text/plain"
                    )
                else:
                    st.error("File does not exist")
            
            # Return button
            if st.button("Back to List", key="back_to_list"):
                st.session_state.current_page = "datasets"
                st.rerun()

# User management (admin function)
def render_users():
    """Render user management page"""
    st.title("User Management")
    
    # Use unified search section
    if create_unified_search_section("users"):
        return
    
    user_manager = UserManager()
    user_manager.render()

# Default login function
def default_login():
    """Use default account to login, for development testing"""
    if not st.session_state.get('authenticated'):
        st.session_state.authenticated = True
        st.session_state.current_user = {
            "user_id": 1,
            "username": "admin",
            "role": "admin"
        }

def main():
    """Main program entry"""
    # Development mode: use default login
    # default_login()  # Uncomment to enable default login
    
    # Normal mode: use authentication manager
    auth_manager = AuthManager()
    sidebar = Sidebar(auth_manager)
    
    # Get current page
    page = sidebar.render()
    
    # Home page doesn't require login
    if page == "Home":
        render_home()
        return
        
    # Other pages require login
    if not auth_manager.is_authenticated():
        st.warning("Please login to access this page")
        return
    
    # Route to corresponding page
    if page == "Model Repository":
        render_models()
    elif page == "Datasets":
        render_datasets()
    elif page == "User Management" and auth_manager.is_admin():
        render_users()
    elif page == "System Management":
        st.write("System management functionality under development...")

if __name__ == "__main__":
    try:
        # Start application directly
        main()
    except Exception as e:
        st.error(f"Application startup failed: {str(e)}")
        print(f"Error details: {str(e)}")
