# main.py
#streamlit run e:/Code/python/OpenModelHub/frontend/app.py [ARGUMENTS]
#(replace by your absolute path)
import streamlit as st
from pathlib import Path
import pandas as pd
import sys
import asyncio
from datetime import datetime
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.extend([str(project_root), str(project_root/"database")])
sys.path.extend([str(project_root), str(project_root/"frontend")])
import frontend.database_api as db_api
from database.database_interface import (
    list_models, get_model_by_id, list_datasets, get_dataset_by_id,
    list_users, get_user_by_id, list_affiliations, init_database,
    create_user, update_user, delete_user
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import os
from dotenv import load_dotenv
import hashlib
from typing import List, Dict
from io import BytesIO
def parse_csv_columns(file_data: bytes) -> List[Dict]:
    df = pd.read_csv(BytesIO(file_data), nrows=1)
    return [{"col_name": col, "col_datatype": "text"} for col in df.columns]

# 异步执行装饰器
def async_to_sync(async_func):
    def wrapper(*args, **kwargs):
        return asyncio.run(async_func(*args, **kwargs))
    return wrapper

# 数据库会话管理
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



# 初始化页面配置
st.set_page_config(
    page_title="OpenModelHub",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 用户认证状态管理
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
# 文件上传处理
def handle_file_upload():
    with st.expander("上传新数据集"):
        with st.form("dataset_upload"):
            name = st.text_input("数据集名称")
            desc = st.text_area("描述")
            file = st.file_uploader("选择数据文件", type=["csv", "txt"])
            if st.form_submit_button("提交"):
                if file:
                    file_path = db_api.db_save_file(file.getvalue(), file.name)
                    db_api.db_create_dataset(name, desc, file_path)
                    st.success("数据集上传成功！")
                else:
                    st.error("请选择文件")

# 文件下载处理
def handle_file_download(dataset):
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
    st.header("🏠 平台概览")
    
    # 获取实时统计信息
    Session = get_db_session()
    
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
    
    # 模型类型分布
    st.subheader("📊 模型类型分布")
    if models:
        df = pd.DataFrame([{
            "类型": model.arch_name.value,
            "参数数量": model.param_num,
            "创建时间": model.created_at
        } for model in models])
        st.bar_chart(df["类型"].value_counts())

# 模型仓库
def render_models():
    st.header("🤖 模型仓库")
    
    Session = get_db_session()
    models = db_api.db_list_models()
    
    # 搜索和过滤
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("🔍 搜索模型（支持自然语言）", key="model_search")
    with col2:
        filter_arch = st.selectbox("架构类型", ["全部", "CNN", "RNN", "Transformer"])
    
    # 展示模型列表
    df = pd.DataFrame([{
        "ID": model.model_id,
        "名称": model.model_name,
        "类型": model.arch_name.value,
        "参数数量": f"{model.param_num:,}",
        "创建时间": model.created_at.strftime("%Y-%m-%d"),
        "下载量": model.download_count
    } for model in models])
    
    if filter_arch != "全部":
        df = df[df["类型"] == filter_arch]
    
    st.dataframe(
        df,
        column_config={
            "ID": "模型ID",
            "名称": "模型名称",
            "类型": "架构类型",
            "参数数量": "参数量",
            "下载量": st.column_config.NumberColumn(
                "下载次数",
                format="%d次"
            )
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
                st.write(f"**创建时间**: {model.created_at}")
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
    st.header("📁 数据集管理")
    
    # 数据集上传
    with st.expander("📤 上传新数据集", expanded=False):
        with st.form("dataset_upload", clear_on_submit=True):
            name = st.text_input("数据集名称*")
            desc = st.text_area("描述")
            media_type = st.selectbox("媒体类型", ["text", "image", "audio", "video"])
            task_type = st.selectbox("任务类型", ["classification", "detection", "generation"])
            file = st.file_uploader("选择数据文件*", type=["csv", "zip"])
            
            if st.form_submit_button("提交"):
                if not name or not file:
                    st.error("带*的字段为必填项")
                else:
                    try:
                        # 保存文件并获取元数据
                        file_path = db_api.db_save_file(file.getvalue(), file.name)
                        
                        # 解析列信息
                        columns = []
                        if file.name.endswith(".csv"):
                            columns =parse_csv_columns(file.getvalue())
                        
                        # 创建数据集记录
                        dataset_data = {
                            "ds_name": name,
                            "ds_size": os.path.getsize(file_path),
                            "media": media_type,
                            "task": task_type,
                            "columns": columns
                        }
                        
                        async def create_dataset_wrapper():
                            async with get_db_session()() as session:
                                return await db_api.create_dataset(session, dataset_data)
                        
                        asyncio.run(create_dataset_wrapper())
                        st.success("数据集上传成功！")
                    except Exception as e:
                        st.error(f"上传失败：{str(e)}")

    # 数据集列表展示
    datasets = db_api.db_list_datasets()
    if not datasets:
        st.info("暂无数据集")
        return
    
    search_term = st.text_input("🔍 搜索数据集")
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
                        # 密码哈希处理
                  #      hashed_pwd = hashlib.sha256(password.encode()).hexdigest()
                        hashed_pwd = password  # 直接使用明文密码
                        db_api.db_create_user(username, hashed_pwd, affiliate)
                        st.success("用户创建成功")
                        st.rerun()
                    except Exception as e:
                        st.error(f"创建失败：{str(e)}")
    
    # 用户列表
    users = db_api.db_list_users()
    df = pd.DataFrame([{
        "ID": user.user_id,
        "用户名": user.user_name,
        "所属机构": user.affiliate,
        "管理员": "✅" if user.is_admin else "❌",
        "注册时间": user.created_at.strftime("%Y-%m-%d")
    } for user in users])
    
    st.dataframe(
        df,
        column_config={
            "ID": "用户ID",
            "管理员": st.column_config.CheckboxColumn("管理员状态")
        },
        use_container_width=True,
        hide_index=True
    )

# 主程序逻辑
def main():
    page = sidebar()
 #   st.session_state.authenticated = True
    db_api.db_create_user("admin", "admin")
    if not st.session_state.authenticated and page != "主页":
        st.warning("请先登录以访问该页面")
        return
    
    if page == "主页":
        render_home()
    elif page == "模型仓库":
        render_models()
    elif page == "数据集":
        render_datasets()
    elif page == "用户管理" and st.session_state.current_user["role"] == "admin":
        render_users()
    elif page == "系统管理":
        st.write("系统管理功能开发中...")

if __name__ == "__main__":
    # 初始化数据库连接
    try:
        asyncio.run(init_database())
    except Exception as e:
        st.error(f"数据库连接失败：{str(e)}")
    
    main()