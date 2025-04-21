# main.py
#streamlit run ~/OpenModelHub/frontend/app.py [ARGUMENTS]
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
sys.path.extend([str(project_root), str(project_root/"agent")])
import frontend.database_api as db_api
import agent.agent_main as agent
from frontend.utils import async_to_sync
from database.database_interface import (
    list_models, get_model_by_id, list_datasets, get_dataset_by_id,
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
from frontend.components import Sidebar, DatasetUploader, UserManager

# 允许嵌套事件循环
nest_asyncio.apply()

# 初始化页面配置（必须在最前面）
st.set_page_config(**APP_CONFIG)

def parse_csv_columns(file_data: bytes) -> List[Dict]:
    """
    解析CSV文件的列信息。

    Args:
        file_data (bytes): CSV文件的二进制数据

    Returns:
        List[Dict]: 包含列信息的列表，每个元素是一个字典，包含列名和数据类型
    """
    df = pd.read_csv(BytesIO(file_data), nrows=1)
    return [{"col_name": col, "col_datatype": "text"} for col in df.columns]

# 数据库会话管理
@st.cache_resource
def get_db_session():
    """
    获取数据库会话。

    Returns:
        AsyncSession: 异步数据库会话对象
    """
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

# 用户认证状态管理
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None

# 文件上传处理
def handle_file_upload():
    """
    处理文件上传功能。
    显示文件上传界面，处理用户上传的文件，并将文件信息保存到数据库。
    """
    with st.expander("上传新数据集"):
        with st.form("dataset_upload"):
            name = st.text_input("数据集名称")
            desc = st.text_area("描述")
            file = st.file_uploader("选择数据文件", type=["csv", "txt"])
            if st.form_submit_button("提交"):
                if file:
                    try:
                        # 保存文件
                        file_path = db_api.db_save_file(file.getvalue(), file.name)
                        # 创建数据集
                        dataset = db_api.db_create_dataset(name, desc, file_path)
                        st.success("数据集上传成功！")
                    except Exception as e:
                        st.error(f"上传失败：{str(e)}")
                else:
                    st.error("请选择文件")

# 文件下载处理
def handle_file_download(dataset):
    """
    处理文件下载功能。

    Args:
        dataset: 要下载的数据集对象
    """
    file_data = db_api._file(dataset.ds_name + ".zip")
    if file_data:
        st.download_button(
            label="下载",
            data=file_data,
            file_name=f"{dataset.ds_name}.zip",
            key=f"download_{dataset.ds_id}"
        )
    else:
        st.error("文件不存在")
        
# 登录表单
def login_form():
    """
    显示登录表单。
    处理用户登录验证，并在成功登录后设置会话状态。
    """
    with st.form("登录", clear_on_submit=True):
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        if st.form_submit_button("登录"):
            # 使用哈希后的密码进行验证（示例使用sha256，生产环境应使用bcrypt）
      #      hashed_pwd = hashlib.sha256(password.encode()).hexdigest()
            hashed_pwd = password  # 直接使用明文密码
            user = db_api.db_authenticate_user(username, hashed_pwd)
            if user  :
                st.session_state.authenticated = True
                st.session_state.current_user = {
                    "user_id": user.user_id,
                    "username": user.user_name,
                    "role": "admin" if user.is_admin else "user"
                }
                st.rerun()
            else:
                st.error("用户名或密码错误")

# 侧边栏导航
def sidebar():
    """
    显示侧边栏导航。
    根据用户登录状态显示不同的导航选项。
    """
    with st.sidebar:
        st.title("OpenModelHub")
        if not st.session_state.authenticated:
            login_form()
        else:
            st.success(f"欢迎，{st.session_state.current_user['username']}！")
            if st.button("退出登录"):
                st.session_state.authenticated = False
                st.session_state.current_user = None
                st.rerun()
            
        menu_items = ["主页", "模型仓库", "数据集", "用户管理"]
        if st.session_state.current_user and st.session_state.current_user["role"] == "admin":
            menu_items += ["系统管理"]
            
        return st.radio("导航菜单", menu_items)

# 主页
def render_home():
    """
    渲染首页内容。
    显示欢迎信息和系统概述。
    """
    st.header("平台概览")
    
    # 直接调用数据库API
    models = db_api.db_list_models()
    datasets = db_api.db_list_datasets()
    users = db_api.db_list_users()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总模型数", len(models))
    with col2:
        st.metric("总数据集", len(datasets))
    with col3:
        st.metric("注册用户", len(users))
    with col4:
        st.metric("今日下载量", 2543)

# 模型仓库
def render_models():
    """
    渲染模型管理页面。
    显示模型列表，支持模型的查看、编辑和删除操作。
    """
    st.header("模型仓库")
    
    # 直接调用数据库API
    models = db_api.db_list_models()
    
    # 搜索和过滤
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("🔍 搜索模型（支持自然语言）", key="model_search")
    with col2:
        filter_arch = st.selectbox("架构类型", ["全部", "CNN", "RNN", "Transformer"])
    
    # 如果用户输入了搜索查询，使用agent进行搜索
    if search_query:
        with st.spinner("正在处理查询..."):
            try:
                # 使用异步执行装饰器调用agent
                result = async_to_sync(agent.query_agent)(f"在模型仓库中搜索：{search_query}")
                st.info("智能搜索结果：")
                st.write(result)
            except Exception as e:
                st.error(f"搜索失败：{str(e)}")
    
    # 展示模型列表
    df = pd.DataFrame([{
        "ID": model.model_id,
        "名称": model.model_name,
        "类型": model.arch_name.value,
        "参数数量": f"{model.param_num:,}"
    } for model in models])
    
    if filter_arch != "全部":
        df = df[df["类型"] == filter_arch]
    
    st.dataframe(
        df,
        column_config={
            "ID": "模型ID",
            "名称": "模型名称",
            "类型": "架构类型",
            "参数数量": "参数量"
        },
        hide_index=True,
        use_container_width=True
    )
    
    # 模型详情侧边栏
    selected_id = st.number_input("输入模型ID查看详情", min_value=1)
    if selected_id:
        model = db_api.db_get_model(selected_id)
        if model:
            with st.expander(f"模型详情 - {model.model_name}"):
                st.write(f"**架构类型**: {model.arch_name.value}")
                st.write(f"**适用媒体类型**: {model.media_type}")
                
                if model.tasks:
                    st.write("**支持任务**:")
                    for task in model.tasks:
                        st.code(task.task_name)
                
                # 下载按钮
                if st.button("下载模型"):
                    st.success("下载开始...（演示用）")

# 修改后的数据集管理
def render_datasets():
    """
    渲染数据集管理页面。
    显示数据集列表，支持数据集的查看、上传、下载和删除操作。
    """
    st.header("📁 数据集管理")
    
    # 数据集上传
    uploader = DatasetUploader()
    if uploader.render():
        st.rerun()
    
    # 直接调用数据库API
    datasets = db_api.db_list_datasets()
    
    if not datasets:
        st.info("暂无数据集")
        return
    
    # 搜索框
    search_term = st.text_input("🔍 搜索数据集（支持自然语言）")
    
    # 如果用户输入了搜索查询，使用agent进行搜索
    if search_term:
        with st.spinner("正在处理查询..."):
            try:
                # 使用异步执行装饰器调用agent
                result = async_to_sync(agent.query_agent)(f"在数据集中搜索：{search_term}")
                st.info("智能搜索结果：")
                st.write(result)
            except Exception as e:
                st.error(f"搜索失败：{str(e)}")
    
    filtered_datasets = [d for d in datasets if search_term.lower() in d.ds_name.lower()]
    
    for dataset in filtered_datasets:
        with st.container(border=True):
            cols = st.columns([1, 4, 1])
            cols[0].markdown(f"**ID**: {dataset.ds_id}")
            cols[1].markdown(f"### {dataset.ds_name}")
            cols[1].caption(f"类型：{dataset.media} | 任务：{dataset.task} | 大小：{dataset.ds_size/1024:.1f}KB")
            
            # 下载按钮
            with cols[2]:
                file_data = db_api.db_get_file(dataset.ds_name + ".zip")
                if file_data:
                    st.download_button(
                        label="下载",
                        data=file_data,
                        file_name=f"{dataset.ds_name}.zip",
                        key=f"download_{dataset.ds_id}"
                    )
                else:
                    st.error("文件缺失")

            # 元数据显示
            with st.expander("详细信息"):
                cols = st.columns(2)
                cols[0].write(f"**创建时间**: {dataset.created_at}")
                cols[1].write(f"**数据列**: {len(dataset.columns)}")
                
                if dataset.columns:
                    st.write("### 数据结构")
                    for col in dataset.columns:
                        st.code(f"{col.col_name}: {col.col_datatype}")

# 用户管理（管理员功能）
def render_users():
    """
    渲染用户管理页面。
    显示用户列表，支持用户的查看、编辑和删除操作。
    """
    user_manager = UserManager()
    user_manager.render()

# 查询页面
def render_query():
    """
    渲染查询页面。
    提供模型查询和数据集查询功能。
    """
    st.header("🔍 智能查询")
    
    # 查询输入
    query = st.text_area("输入您的查询", height=100)
    
    if st.button("查询"):
        if query:
            with st.spinner("正在处理查询..."):
                try:
                    # 使用异步执行装饰器调用agent
                    result = async_to_sync(agent.query_agent)(query)
                    st.success("查询完成！")
                    st.write("查询结果：")
                    st.write(result)
                except Exception as e:
                    st.error(f"查询失败：{str(e)}")
        else:
            st.warning("请输入查询内容")

# 主程序逻辑
def main():
    """
    应用程序的主入口函数。
    初始化应用程序，设置页面配置，并处理用户会话。
    """
    auth_manager = AuthManager()
    sidebar = Sidebar(auth_manager)
    
    # 获取当前页面
    page = sidebar.render()
    
    # 检查认证状态
    if not auth_manager.is_authenticated() and page != "主页":
        st.warning("请先登录以访问该页面")
        return
    
    # 路由到对应页面
    if page == "主页":
        render_home()
    elif page == "模型仓库":
        render_models()
    elif page == "数据集":
        render_datasets()
    elif page == "用户管理" and auth_manager.is_admin():
        render_users()
    elif page == "系统管理":
        st.write("系统管理功能开发中...")
    elif page == "智能查询":
        render_query()

if __name__ == "__main__":
    try:
        # 直接启动应用
        main()
    except Exception as e:
        st.error(f"应用启动失败：{str(e)}")
        print(f"错误详情：{str(e)}")