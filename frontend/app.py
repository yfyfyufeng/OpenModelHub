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
    create_user, update_user, delete_user
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
    """创建分页"""
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
    for item in current_items:
        with st.container(border=True):
            col1, col2 = st.columns([5, 0.7])
            with col1:
                st.write(f"### {item.model_name if type=='models' else item.ds_name}")
                if type == "models":
                    st.caption(f"架构: {item.arch_name.value} | 媒体类型: {item.media_type.value} | 参数量: {item.param_num:,}")
                else:
                    tasks = [task.task.value for task in item.Dataset_TASK] if hasattr(item, 'Dataset_TASK') else []
                    st.caption(f"类型: {item.media} | 任务: {', '.join(tasks)} | 大小: {item.ds_size/1024:.1f}KB")
            
            with col2:
                if st.button("查看详情", key=f"{type[:-1]}_{item.model_id if type=='models' else item.ds_id}", use_container_width=True):
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
        st.metric("今日下载量", 2543)
    
    # 添加导出数据按钮
    if st.button("导出所有数据"):
        try:
            # 获取所有数据
            json_data = db_api.db_export_all_data()
            if json_data:
                # 生成文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"data_{timestamp}.json"
                
                # 转换为JSON字符串，确保使用UTF-8编码
                import json
                json_str = json.dumps(json_data, indent=4, ensure_ascii=False)
                
                # 提供下载
                st.download_button(
                    label="点击下载数据",
                    data=json_str.encode('utf-8'),  # 确保使用UTF-8编码
                    file_name=filename,
                    mime="application/json"
                )
            else:
                st.error("导出数据失败")
        except Exception as e:
            st.error(f"导出数据时出错：{str(e)}")
    
    # 添加注册功能
    if not st.session_state.get('authenticated'):
        st.markdown("---")
        st.subheader("新用户注册")
        with st.form("注册", clear_on_submit=True):
            new_username = st.text_input("用户名")
            new_password = st.text_input("密码", type="password")
            confirm_password = st.text_input("确认密码", type="password")
            
            if st.form_submit_button("注册"):
                if new_username and new_password:
                    if new_password != confirm_password:
                        st.error("两次输入的密码不一致")
                        return
                    try:
                        # 检查用户名是否已存在
                        existing_user = db_api.db_get_user_by_username(new_username)
                        if existing_user:
                            st.error("用户名已存在")
                            return
                            
                        # 创建新用户，默认非管理员
                        user = db_api.db_create_user(new_username, new_password, is_admin=False)
                        if user:
                            st.success("注册成功！请使用新账号登录")
                        else:
                            st.error("注册失败，请重试")
                    except Exception as e:
                        st.error(f"注册失败：{str(e)}")
                else:
                    st.error("请填写用户名和密码")

# Model repository
def render_models():
    """Render model repository page"""
    st.title("Model Repository")
    
     # Use unified search section
    if create_search_section("models"):
        return
    
    # Add search input box
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
        col1, col2 = st.columns([4.6, 0.4])
        
        with col1:
            search_query = st.text_input("Search Models", placeholder="Enter natural language query")
        with col2:
            search_clicked = st.button("Search", key="model_search", use_container_width=True)
    
    if search_query and search_clicked:
        results, query_info = db_api.db_agent_query(search_query)
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
            return
    
    # Model upload
    with st.expander("Upload New Model"):
        with st.form("model_upload"):
            name = st.text_input("Model Name")
            param_num = st.number_input("Parameter Count", min_value=1000, value=1000000)
            
            # Architecture type selection
            arch_types = ["CNN", "RNN", "TRANSFORMER"]
            arch_type = st.selectbox(
                "Architecture Type",
                arch_types,
                help="Select model architecture type"
            )
            
            # Media type selection
            media_types = ["TEXT", "IMAGE", "AUDIO", "VIDEO"]
            media_type = st.selectbox(
                "Media Type",
                media_types,
                help="Select media type the model is suitable for"
            )
            
            # Task selection
            task_types = ["CLASSIFICATION", "DETECTION", "GENERATION", "SEGMENTATION"]
            selected_tasks = st.multiselect(
                "Task Types",
                task_types,
                default=["CLASSIFICATION"],
                help="You can select multiple task types"
            )
            
            # Training type selection
            train_types = ["PRETRAIN", "FINETUNE", "RL"]
            train_type = st.selectbox(
                "Training Type",
                train_types,
                help="Select model training type"
            )
            
            # File upload
            file = st.file_uploader("Select Model File", type=["pt", "pth", "ckpt", "bin", "txt"])
            
            if st.form_submit_button("Submit"):
                if file and name:
                    try:
                        # Save file
                        file_path = db_api.db_save_file(file.getvalue(), file.name)
                        
                        # Create model
                        model_data = {
                            "model_name": name,
                            "param_num": param_num,
                            "arch_name": arch_type,
                            "media_type": media_type,  # 直接使用大写值
                            "tasks": selected_tasks,
                            "trainname": train_type,
                            "param": str(file_path),
                            "creator_id": st.session_state.current_user["user_id"] if st.session_state.get("current_user") else 1
                        }
                        db_api.db_create_model(model_data)
                        st.success("Model uploaded successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Upload failed: {str(e)}")
                else:
                    st.error("Please enter model name and select a file")
                    
    st.markdown("""
        <style>
        /* Reduce model card padding */
        .stContainer {
            padding: 1rem !important;
        }
        
        /* Make subheader smaller */
        .stMarkdown h3 {
            font-size: 1.4rem !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        
        /* Reduce caption size and spacing */
        .stMarkdown p {
            font-size: 0.9rem !important;
            margin: 0.2rem 0 !important;
            padding: 0 !important;
        }
        
        /* Make buttons smaller */
        .stButton button {
            padding: 0.2rem 1rem !important;
            height: auto !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Display all models
    models = db_api.db_list_models()
    
    if not models:
        st.info("No models available")
        return
    
    # Sort models by creation time
    sorted_models = sorted(models, 
                           key=lambda x: x.created_at if hasattr(x, 'created_at') else datetime.now(), 
                           reverse=True)
    
    # Get current page items
    create_pagination(sorted_models, "models")
    
    # Display model details
    if st.session_state.get("current_page") == "model_detail":
        model = st.session_state.get("selected_model")
        if model:
            st.markdown("---")
            st.subheader(f"Model Details - {model.model_name}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Architecture Type:**")
                st.write(model.arch_name.value)
                
                st.write("**Media Type:**")
                st.write(model.media_type.value)
                
                st.write("**Parameter Count:**")
                st.write(f"{model.param_num:,}")
            
            with col2:
                st.write("**Training Type:**")
                st.write(model.trainname.value)
                
                st.write("**Supported Tasks:**")
                tasks = [task.task_name.value for task in model.tasks] if hasattr(model, 'tasks') else []
                st.write(", ".join(tasks) if tasks else "No tasks")
            
            # Download button
            if st.button("Download Model", key=f"download_{model.model_id}"):
                file_data = db_api.db_get_file(f"{model.model_name}.pt")
                if file_data:
                    st.download_button(
                        label="Click to Download",
                        data=file_data,
                        file_name=f"{model.model_name}.pt",
                        mime="application/octet-stream"
                    )
                else:
                    st.error("File does not exist")
            
            # Return button
            if st.button("Back to List", key="back_to_list"):
                st.session_state.current_page = "models"
                st.rerun()
    
    # If no search or search has no results, display all models
    models = db_api.db_list_models()
    
    # Show model list
    df = pd.DataFrame([{
        "ID": model.model_id,
        "Name": model.model_name,
        "Type": model.arch_name.value,
        "Parameter Count": f"{model.param_num:,}" if hasattr(model, 'param_num') else "Unknown"
    } for model in models])
    
    st.dataframe(
        df,
        column_config={
            "ID": "Model ID",
            "Name": "Model Name",
            "Type": "Architecture Type",
            "Parameter Count": "Parameter Count"
        },
        hide_index=True,
        use_container_width=True
    )
    
    # Model details sidebar
    selected_id = st.number_input("Enter model ID to view details", min_value=1)
    if selected_id:
        model = db_api.db_get_model(selected_id)
        if model:
            with st.expander(f"Model Details - {model.model_name}"):
                st.write(f"**Architecture Type**: {model.arch_name.value}")
                st.write(f"**Applicable Media Type**: {model.media_type}")
                
                if model.tasks:
                    st.write("**Supported Tasks**:")
                    for task in model.tasks:
                        st.code(task.task_name)
                
                # Download button
                if st.button("Download Model"):
                    st.success("Download started... (Demo)")

# Modified dataset management
def render_datasets():
    """Render dataset management page"""
    st.title("Dataset Management")
    
    # Use unified search section
    if create_search_section("datasets"):
        return
    
    # Add search input box
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
        col1, col2 = st.columns([4.6, 0.4])
        
        with col1:
            search_query = st.text_input("Search Datasets", placeholder="Enter natural language query")
        with col2:
            search_clicked = st.button("Search", key="dataset_search", use_container_width=True)
    if search_query and search_clicked:
        results, query_info = db_api.db_agent_query(search_query)
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
            return
    
    # Dataset upload
    with st.expander("Upload New Dataset"):
        with st.form("dataset_upload"):
            name = st.text_input("Dataset Name")
            desc = st.text_area("Description")
            file = st.file_uploader("Select Data File", type=["txt"])
            
            # Task selection
            st.write("Select Task Type:")
            # Predefined task types
            predefined_tasks = ["classification", "detection", "generation", "segmentation"]
            selected_tasks = st.multiselect(
                "Select Task Types",
                predefined_tasks,
                default=["classification"],
                help="You can select multiple task types"
            )
            
            if st.form_submit_button("Submit"):
                if file:
                    try:
                        # Save file
                        file_path = db_api.db_save_file(file.getvalue(), file.name)
                        
                        # Create dataset
                        dataset_data = {
                            "ds_name": name,
                            "ds_size": len(file.getvalue()),
                            "media": "text",  # Default type
                            "task": selected_tasks,  # Use selected tasks
                            "columns": [
                                {"col_name": "content", "col_datatype": "text"}
                            ],
                            "description": desc,  # Add description field
                            "created_at": datetime.now()  # Add creation time
                        }
                        db_api.db_create_dataset(name, dataset_data)
                        st.success("Dataset uploaded successfully!")
                        st.rerun()  # Refresh page to show new dataset
                    except Exception as e:
                        st.error(f"Upload failed: {str(e)}")
                else:
                    st.error("Please select a file")
    
    # If no search or search has no results, display all datasets
    datasets = db_api.db_list_datasets()
    
    st.markdown("""
        <style>
        /* Reduce model card padding */
        .stContainer {
            padding: 1rem !important;
        }
        
        /* Make subheader smaller */
        .stMarkdown h3 {
            font-size: 1.5rem !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        
        /* Reduce caption size and spacing */
        .stMarkdown p {
            font-size: 0.9rem !important;
            margin: 0.2rem 0 !important;
            padding: 0 !important;
        }
        
        /* Make buttons smaller */
        .stButton button {
            padding: 0.2rem 1rem !important;
            height: auto !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    if not datasets:
        st.info("No datasets available")
        return
    
    # Sort datasets by creation time
    sorted_datasets = sorted(datasets, key=lambda x: x.created_at, reverse=True)
    
    # Get current page items
    create_pagination(sorted_datasets, "datasets")
    
    # Display dataset details
    if st.session_state.get("current_page") == "dataset_detail":
        dataset = st.session_state.get("selected_dataset")
        if dataset:
            st.markdown("---")
            st.subheader(f"Dataset Details - {dataset.ds_name}")
            
            # Display description
            st.write("**Description:**")
            st.write(dataset.description if hasattr(dataset, 'description') else "No description available")
            
            # Display task information
            st.write("**Task Types:**")
            tasks = [task.task.value for task in dataset.Dataset_TASK]
            st.write(", ".join(tasks) if tasks else "No tasks")
            
            # Display dataset size
            st.write("**Dataset Size:**")
            st.write(f"{dataset.ds_size/1024:.1f}KB")
            
            # Download button
            if st.button("Download Dataset", key=f"download_{dataset.ds_id}"):
                file_data = db_api.db_get_file(dataset.ds_name + ".txt")
                if file_data:
                    st.download_button(
                        label="Click to Download",
                        data=file_data,
                        file_name=f"{dataset.ds_name}.txt",
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
    elif page == "System Management":
        st.write("System management functionality under development...")

if __name__ == "__main__":
    try:
        # Start application directly
        main()
    except Exception as e:
        st.error(f"Application startup failed: {str(e)}")
        print(f"Error details: {str(e)}")
