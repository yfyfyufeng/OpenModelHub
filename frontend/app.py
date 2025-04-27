# main.py
#streamlit run ./frontend/app.py [ARGUMENTS]
#(replace by your absolute path)
import streamlit as st
from pathlib import Path
import pandas as pd
import sys
import asyncio
from datetime import datetime
import plotly.express as px
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
from data_analysis import data_ins
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
from frontend.database_api import pending_invitations

# Allow nested event loops
nest_asyncio.apply()

# Initialize page configuration (must be at the beginning)
st.set_page_config(**APP_CONFIG)

def create_pagination(items, type, page_size=10, page_key="default"):
    """Create pagination for items
    Args:
        items: List of items to paginate
        type: Type of items (models, datasets)
        page_size: Number of items per page
        page_key: Unique key for pagination state
    """
    # Get current page from session state
    page_state_key = f'current_page_num_{type}'
    if page_state_key not in st.session_state:
        st.session_state[page_state_key] = 1
        
    # Calculate total pages
    total_pages = (len(items) + page_size - 1) // page_size
    
    if st.session_state[page_state_key] > total_pages:
        st.session_state[page_state_key] = 1
    
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
            st.button(f'{st.session_state[page_state_key]}/{total_pages}', disabled=True, key=f"page_num_{type}")
            
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
            
        menu_items = ["Home", "Model Repository", "Datasets"]
        if st.session_state.current_user and st.session_state.current_user["role"] == "admin":
            menu_items.append("User Management")
            menu_items.append("data insight")
            
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
    
    # Export data button
    if st.button("Export All Data"):
        try:
            # Get all data
            json_data = db_api.db_export_all_data()
            if json_data:
                # Generate filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"data_{timestamp}.json"
                
                # Convert to JSON string with UTF-8 encoding
                import json
                json_str = json.dumps(json_data, indent=4, ensure_ascii=False)
                
                # Provide download
                st.download_button(
                    label="Download Data",
                    data=json_str.encode('utf-8'),
                    file_name=filename,
                    mime="application/json"
                )
            else:
                st.error("Data export failed")
        except Exception as e:
            st.error(f"Export error: {str(e)}")
    
    # Registration form
    if not st.session_state.get('authenticated'):
        st.markdown("---")
        st.subheader("New User Registration")
        with st.form("Register", clear_on_submit=True):
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            
            if st.form_submit_button("Register"):
                if new_username and new_password:
                    if new_password != confirm_password:
                        st.error("Passwords do not match")
                        return
                    try:
                        # Check if username exists
                        existing_user = db_api.db_get_user_by_username(new_username)
                        if existing_user:
                            st.error("Username already exists")
                            return
                            
                        # Create new user (non-admin by default)
                        user = db_api.db_create_user(new_username, new_password, is_admin=False)
                        if user:
                            st.success("Registration successful! Please login with new account")
                        else:
                            st.error("Registration failed")
                    except Exception as e:
                        st.error(f"Registration error: {str(e)}")
                else:
                    st.error("Please fill in username and password")

def render_models():
    """Render model repository page"""
    st.title("Model Repository")
    
    # Use unified search section
    if create_search_section("models"):
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
            
            file = st.file_uploader("Select Model File*", type=["pt", "pth", "ckpt", "txt"])
            
            if st.form_submit_button("Submit"):
                if not name or not file:
                    st.error("Fields marked with * are required")
                else:
                    try:
                        # 只保存文件，不进行数据库操作
                        file_path = db_api.db_save_file(file.getvalue(), file.name)
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
            st.subheader(f"Model Details - {model.model_name}")
            
            # Display basic information
            st.write("**Basic Information:**")
            info_data = {
                "Model ID": model.model_id,
                "Architecture Type": model.arch_name.value if hasattr(model.arch_name, 'value') else model.arch_name,
                "Parameter Count": f"{model.param_num:,}",
                "Training Type": model.trainname.value if hasattr(model.trainname, 'value') else model.trainname
            }
            st.table(pd.DataFrame(list(info_data.items()), columns=["Property", "Value"]))
            
            # Display related information
            st.write("**Related Information:**")
            if hasattr(model, 'tasks'):
                tasks = [task.task_name.value if hasattr(task.task_name, 'value') else task.task_name for task in model.tasks]
            else:
                tasks = []
            
            if hasattr(model, 'authors'):
                # Get author information from the database
                author_ids = [author.user_id for author in model.authors]
                authors = []
                for author_id in author_ids:
                    author = db_api.db_get_user_by_id(author_id)
                    if author:
                        authors.append(author.user_name)
            else:
                authors = []
            
            if hasattr(model, 'datasets'):
                # Get dataset information from the database
                dataset_ids = [dataset.dataset_id for dataset in model.datasets]
                datasets = []
                for dataset_id in dataset_ids:
                    dataset = db_api.db_get_dataset(dataset_id)
                    if dataset:
                        datasets.append(dataset.ds_name)
            else:
                datasets = []
            
            rel_data = {
                "Supported Tasks": ", ".join(tasks) if tasks else "No tasks",
                "Authors": ", ".join(authors) if authors else "No authors",
                "Related Datasets": ", ".join(datasets) if datasets else "No datasets"
            }
            st.table(pd.DataFrame(list(rel_data.items()), columns=["Type", "Name"]))
            
            # Download button
            if st.button("Download Model", key=f"download_{model.model_id}"):
                file_data = b"hello world"
                st.download_button(
                    label="Click to Download",
                    data=file_data,
                    file_name=f"{model.model_name}.txt",
                    mime="text/plain"
                )
            
            # Return button
            if st.button("Back to List", key="back_to_list"):
                st.session_state.current_page = "models"
                st.rerun()

def render_datasets():
    """Render dataset management page"""
    st.title("Dataset Management")
    
    # Use unified search section
    if create_search_section("datasets"):
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
    
    # 初始化session state
    if 'datasets' not in st.session_state:
        st.session_state.datasets = []
    if 'loading_complete' not in st.session_state:
        st.session_state.loading_complete = False
    
    # 显示加载状态
    if not st.session_state.loading_complete:
        with st.spinner('Loading datasets...'):
            # 获取所有数据集
            all_datasets = db_api.db_list_datasets()
            # 只显示前10个
            st.session_state.datasets = all_datasets[:10]
            
            # 在后台加载剩余数据集
            def load_remaining_datasets():
                st.session_state.datasets = all_datasets
                st.session_state.loading_complete = True
            
            # 使用线程池执行后台加载
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.submit(load_remaining_datasets)
    
    # 显示数据集列表
    if not st.session_state.datasets:
        st.info("No datasets available")
        return
    
    # 显示加载进度
    if not st.session_state.loading_complete:
        st.info("Loading more datasets...")
    
    # 优化分页显示
    page_size = 5  # 减少每页显示的数据集数量
    create_pagination(st.session_state.datasets, "datasets", page_size=page_size)
    
    # Display dataset details
    if st.session_state.get("current_page") == "dataset_detail":
        dataset = st.session_state.get("selected_dataset")
        if dataset:
            st.markdown("---")
            st.subheader(f"Dataset Details - {dataset.ds_name if hasattr(dataset, 'ds_name') else dataset.get('ds_name', '')}")
            
            # 使用列布局优化显示
            col1, col2 = st.columns(2)
            
            with col1:
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
            
            with col2:
                # Display dataset size
                st.write("**Dataset Size:**")
                size = dataset.ds_size if hasattr(dataset, 'ds_size') else dataset.get('ds_size', 0)
                st.write(f"{size/1024:.1f}KB")
                
                # Download button
                if st.button("Download Dataset", key=f"download_{dataset.ds_id if hasattr(dataset, 'ds_id') else dataset.get('ds_id', '')}"):
                    file_data = b"hello world"
                    st.download_button(
                        label="Click to Download",
                        data=file_data,
                        file_name=f"{dataset.ds_name if hasattr(dataset, 'ds_name') else dataset.get('ds_name', '')}.txt",
                        mime="text/plain"
                    )
            
            # Return button
            if st.button("Back to List", key="back_to_list"):
                st.session_state.current_page = "datasets"
                st.rerun()


def render_data_insight():
    attributes = ["media_type", "arch_name", "trainname"]
    types = {"media_type":["audio", "image", "text", "video"],
             "arch_name": ["CNN", "RNN", "Transformer"], 
             "trainname": ["Trainname.FINETUNE", "Trainname.PRETRAIN", "Trainname.RL"],
            }
    output = data_ins()

    st.write("# Model")
    model = output["model"]
    for attr in attributes:
        data = pd.DataFrame({
            "Category": types[attr],
            "Value": model[attr].values()
        })
        fig = px.pie(
            data,
            names="Category",
            values="Value",
            title=f"number & percentage of each {attr} of model"
        )
        st.plotly_chart(fig)

    fig = px.imshow(
        pd.DataFrame(model["media_task_relation"]),
        labels=dict(x="media", y="task", color="Value"),
        color_continuous_scale="Viridis",
        title="media_task_relation"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("## param_num summary:")
    st.dataframe(model["param_num"])

    # st.write("## AI Summary:")
    # st.write(model["comment"])

    st.write("---")

    st.write("# dataset")
    dataset = output["dataset"]
    fig = px.imshow(
        pd.DataFrame(dataset["media_task_relation"]),
        labels=dict(x="media", y="task", color="Value"),
        color_continuous_scale="Viridis",
        title="media_task_relation"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("## param_num summary:")
    st.dataframe(dataset["ds_size"])

    # st.write("## AI Summary:")
    # st.write(dataset["comment"])

    st.write("---")
    
    st.write("# user")
    user = output["user"]
    fig = px.bar(
        user,
        x="affiliate",
        y="count",
        title="user in different affiliations"
    )
    st.plotly_chart(fig)
    
# User management (admin function)

def render_users():
    """Render user management page"""
    # 保留原有的UserManager界面
    user_manager = UserManager()
    user_manager.render()
    
    # 添加管理员功能
    st.markdown("---")
    st.subheader("Admin features")
    
    # Check if current user is admin
    if not st.session_state.get('current_user', {}).get('role') == 'admin':
        st.error("Only admin can access these features.")
        return
    
    # 使用统一的字段查询功能
    col1, col2, col3 = st.columns([1, 1, 0.5])
    with col1:
        field_attr = st.selectbox(
            "Select Field",
            ["user_id", "user_name", "email", "organization"]
        )
    with col2:
        field_val = st.text_input("Enter Query Value")
    with col3:
        if st.button("Search", key="user_field_search"):
            try:
                ids = asyncio.run(get_user_ids_by_attribute(get_db_session()(), field_attr, field_val))
                if ids:
                    user = db_api.db_get_user_by_id(ids[0])  # 获取第一个匹配的用户
                    if user:
                        st.session_state.selected_user = user
                        st.session_state.current_page = "user_detail"
                    else:
                        st.error("User not found.")
                else:
                    st.error("User not found.")
            except Exception as e:
                st.error(f"Error when searching: {str(e)}")
    
    # Display user details and edit form
    if st.session_state.get("current_page") == "user_detail":
        user = st.session_state.get("selected_user")
        if user:
            st.markdown("---")
            st.subheader(f"Edit User - {user.user_name}")
            
            with st.form("edit_user_form"):
                # Display current user information
                st.write("**Current Information:**")
                info_data = {
                    "Username": user.user_name,
                    "Email": user.email if hasattr(user, 'email') else 'N/A',
                    "Organization": user.organization if hasattr(user, 'organization') else 'N/A',
                    "Admin Status": "Yes" if user.is_admin else "No"
                }
                st.table(pd.DataFrame(list(info_data.items()), columns=["Property", "Value"]))
                
                # Edit form
                st.write("**Edit Information:**")
                new_organization = st.text_input("Organization", value=user.organization if hasattr(user, 'organization') else '')
                new_admin_status = st.checkbox("Admin Status", value=user.is_admin)
                
                if st.form_submit_button("Save Changes"):
                    try:
                        # Update user information
                        update_data = {
                            "organization": new_organization,
                            "is_admin": new_admin_status
                        }
                        db_api.db_update_user(user.user_id, update_data)
                        st.success("User information updated successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Update failed: {str(e)}")
            
            # Return button
            if st.button("Back to List", key="back_to_list"):
                st.session_state.current_page = "users"
                st.rerun()

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
    
    # Check authentication status
    if not auth_manager.is_authenticated() and page != "Home":
        st.warning("Please login first to access this page")
        return
    
    # Route to corresponding page
    if page == "Home":
        render_home()
    elif page == "Model Repository":
        render_models()
    elif page == "Datasets":
        render_datasets()
    elif page == "User Management" and auth_manager.is_admin():
        render_users()
    elif page == "data insight":
        #st.write("data insight is under developing")
        render_data_insight()
    # elif page == "System Management":
    #     st.write("System management functionality under development...")

if __name__ == "__main__":
    try:
        # Start application directly
        main()
    except Exception as e:
        st.error(f"Application startup failed: {str(e)}")
        print(f"Error details: {str(e)}")