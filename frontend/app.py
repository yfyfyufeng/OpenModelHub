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
from frontend.components import Sidebar, DatasetUploader, UserManager, ModelUploader

# 允许嵌套事件循环
nest_asyncio.apply()

# 初始化页面配置（必须在最前面）
st.set_page_config(**APP_CONFIG)

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
        use_encryption = st.checkbox("使用加密登录", value=True)
        
        if st.form_submit_button("登录"):
            # 使用哈希后的密码进行验证（示例使用sha256，生产环境应使用bcrypt）
            #      hashed_pwd = hashlib.sha256(password.encode()).hexdigest()
            hashed_pwd = password  # 直接使用明文密码
            if use_encryption:
                user = db_api.db_authenticate_user(username, hashed_pwd)
            else:
                # 非加密登录
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
    """渲染主页"""
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
                json_str = json.dumps(json_data, indent=4, ensure_ascii=False, encoding='utf-8')
                
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

# 模型仓库
def render_models():
    """渲染模型仓库页面"""
    st.title("模型仓库")
    
    # 添加搜索输入框
    search_query = st.text_input("搜索模型", placeholder="输入自然语言查询")
    
    # 添加搜索按钮
    if st.button("搜索", key="model_search"):
        if search_query:
            results, query_info = db_api.db_agent_query(search_query)
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
                return
    
    # 模型上传
    with st.expander("上传新模型"):
        with st.form("model_upload"):
            name = st.text_input("模型名称")
            param_num = st.number_input("参数量", min_value=1000, value=1000000)
            
            # 架构类型选择
            arch_types = ["CNN", "RNN", "TRANSFORMER"]
            arch_type = st.selectbox(
                "架构类型",
                arch_types,
                help="选择模型架构类型"
            )
            
            # 媒体类型选择
            media_types = ["TEXT", "IMAGE", "AUDIO", "VIDEO"]
            media_type = st.selectbox(
                "媒体类型",
                media_types,
                help="选择模型适用的媒体类型"
            )
            
            # 任务选择
            task_types = ["CLASSIFICATION", "DETECTION", "GENERATION", "SEGMENTATION"]
            selected_tasks = st.multiselect(
                "任务类型",
                task_types,
                default=["CLASSIFICATION"],
                help="可以选择多个任务类型"
            )
            
            # 训练类型选择
            train_types = ["PRETRAIN", "FINETUNE", "RL"]
            train_type = st.selectbox(
                "训练类型",
                train_types,
                help="选择模型的训练类型"
            )
            
            # 文件上传
            file = st.file_uploader("选择模型文件", type=["pt", "pth", "ckpt", "bin", "txt"])
            
            if st.form_submit_button("提交"):
                if file and name:
                    try:
                        # 保存文件
                        file_path = db_api.db_save_file(file.getvalue(), file.name)
                        
                        # 创建模型
                        model_data = {
                            "model_name": name,
                            "param_num": param_num,
                            "arch_name": arch_type,
                            "media_type": media_type,
                            "tasks": selected_tasks,
                            "trainname": train_type,
                            "param": file_path,
                            "creator_id": st.session_state.current_user["user_id"] if st.session_state.get("current_user") else 1
                        }
                        db_api.db_create_model(model_data)
                        st.success("模型上传成功！")
                        st.rerun()
                    except Exception as e:
                        st.error(f"上传失败：{str(e)}")
                else:
                    st.error("请填写模型名称并选择文件")
    
    # 显示所有模型
    models = db_api.db_list_models()
    
    if not models:
        st.info("暂无模型")
        return
    
    # 添加筛选选项
    show_my_models = st.checkbox("只显示我创建的模型", value=False)
    
    # 根据筛选条件过滤模型
    if show_my_models and st.session_state.get("current_user"):
        current_user_id = st.session_state.current_user["user_id"]
        models = [model for model in models if model.creator_id == current_user_id]
    
    # 展示模型列表
    df = pd.DataFrame([{
        "ID": model.model_id,
        "名称": model.model_name,
        "类型": model.arch_name.value,
        "参数数量": f"{model.param_num:,}" if hasattr(model, 'param_num') else "未知",
        "创建者": model.creator.user_name if hasattr(model, 'creator') else "未知",
        "创建时间": model.created_at.strftime("%Y-%m-%d %H:%M:%S") if hasattr(model, 'created_at') else "未知"
    } for model in models])
    
    st.dataframe(
        df,
        column_config={
            "ID": "模型ID",
            "名称": "模型名称",
            "类型": "架构类型",
            "参数数量": "参数量",
            "创建者": "创建者",
            "创建时间": "创建时间"
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
                st.write(f"**创建者**: {model.creator.user_name if hasattr(model, 'creator') else '未知'}")
                st.write(f"**创建时间**: {model.created_at.strftime('%Y-%m-%d %H:%M:%S') if hasattr(model, 'created_at') else '未知'}")
                
                if model.tasks:
                    st.write("**支持任务**:")
                    for task in model.tasks:
                        st.code(task.task_name)
                
                # 下载按钮
                if st.button("下载模型"):
                    file_data = db_api.db_get_file(f"{model.model_name}.pt")
                    if file_data:
                        st.download_button(
                            label="点击下载",
                            data=file_data,
                            file_name=f"{model.model_name}.pt",
                            mime="application/octet-stream"
                        )
                    else:
                        st.error("文件不存在")

# 修改后的数据集管理
def render_datasets():
    """渲染数据集管理页面"""
    st.title("数据集管理")
    
    # 添加搜索输入框
    search_query = st.text_input("搜索数据集", placeholder="输入自然语言查询")
    
    # 添加搜索按钮
    if st.button("搜索", key="dataset_search"):
        if search_query:
            results, query_info = db_api.db_agent_query(search_query)
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
                return
    
    # 数据集上传
    with st.expander("上传新数据集"):
        with st.form("dataset_upload"):
            name = st.text_input("数据集名称")
            desc = st.text_area("描述")
            file = st.file_uploader("选择数据文件", type=["txt"])
            
            # 任务选择
            st.write("选择任务类型：")
            # 预定义的任务类型
            predefined_tasks = ["classification", "detection", "generation", "segmentation"]
            selected_tasks = st.multiselect(
                "选择任务类型",
                predefined_tasks,
                default=["classification"],
                help="可以选择多个任务类型"
            )
            
            if st.form_submit_button("提交"):
                if file:
                    try:
                        # 保存文件
                        file_path = db_api.db_save_file(file.getvalue(), file.name)
                        
                        # 创建数据集
                        dataset_data = {
                            "ds_name": name,
                            "ds_size": len(file.getvalue()),
                            "media": "text",  # 默认类型
                            "task": selected_tasks,  # 使用选择的任务
                            "columns": [
                                {"col_name": "content", "col_datatype": "text"}
                            ],
                            "description": desc,  # 添加描述字段
                            "creator_id": st.session_state.current_user["user_id"] if st.session_state.get("current_user") else 1
                        }
                        db_api.db_create_dataset(name, dataset_data)
                        st.success("数据集上传成功！")
                        st.rerun()  # 刷新页面以显示新数据集
                    except Exception as e:
                        st.error(f"上传失败：{str(e)}")
                else:
                    st.error("请选择文件")
    
    # 如果没有搜索或搜索无结果，显示所有数据集
    datasets = db_api.db_list_datasets()
    
    if not datasets:
        st.info("暂无数据集")
        return
    
    # 添加筛选选项
    show_my_datasets = st.checkbox("只显示我创建的数据集", value=False)
    
    # 根据筛选条件过滤数据集
    if show_my_datasets and st.session_state.get("current_user"):
        current_user_id = st.session_state.current_user["user_id"]
        datasets = [dataset for dataset in datasets if dataset.creator_id == current_user_id]
    
    # 展示数据集列表
    df = pd.DataFrame([{
        "ID": dataset.ds_id,
        "名称": dataset.ds_name,
        "类型": dataset.media.value if hasattr(dataset.media, 'value') else dataset.media,
        "大小": f"{dataset.ds_size/1024:.1f}KB",
        "创建者": dataset.creator.user_name if hasattr(dataset, 'creator') else "未知",
        "创建时间": dataset.created_at.strftime("%Y-%m-%d %H:%M:%S") if hasattr(dataset, 'created_at') else "未知"
    } for dataset in datasets])
    
    st.dataframe(
        df,
        column_config={
            "ID": "数据集ID",
            "名称": "数据集名称",
            "类型": "媒体类型",
            "大小": "大小",
            "创建者": "创建者",
            "创建时间": "创建时间"
        },
        hide_index=True,
        use_container_width=True
    )
    
    # 数据集详情侧边栏
    selected_id = st.number_input("输入数据集ID查看详情", min_value=1)
    if selected_id:
        dataset = db_api.db_get_dataset(selected_id)
        if dataset:
            with st.expander(f"数据集详情 - {dataset.ds_name}"):
                st.write(f"**描述**: {dataset.description if hasattr(dataset, 'description') else '暂无描述'}")
                st.write(f"**类型**: {dataset.media.value if hasattr(dataset.media, 'value') else dataset.media}")
                st.write(f"**创建者**: {dataset.creator.user_name if hasattr(dataset, 'creator') else '未知'}")
                st.write(f"**创建时间**: {dataset.created_at.strftime('%Y-%m-%d %H:%M:%S') if hasattr(dataset, 'created_at') else '未知'}")
                
                if dataset.Dataset_TASK:
                    st.write("**任务类型**:")
                    for task in dataset.Dataset_TASK:
                        st.code(task.task.value if hasattr(task.task, 'value') else task.task)
                
                # 下载按钮
                if st.button("下载数据集"):
                    file_data = db_api.db_get_file(dataset.ds_name + ".txt")
                    if file_data:
                        st.download_button(
                            label="点击下载",
                            data=file_data,
                            file_name=f"{dataset.ds_name}.txt",
                            mime="text/plain"
                        )
                    else:
                        st.error("文件不存在")

# 用户管理（管理员功能）
def render_users():
    """渲染用户管理页面"""
    st.title("用户管理")
    
    # 获取所有用户
    users = db_api.db_list_users()
    
    if not users:
        st.info("暂无用户")
        return
    
    # 展示用户列表
    df = pd.DataFrame([{
        "ID": user.user_id,
        "用户名": user.user_name,
        "管理员": "是" if user.is_admin else "否",
        "创建时间": user.created_at.strftime("%Y-%m-%d %H:%M:%S") if hasattr(user, 'created_at') else "未知"
    } for user in users])
    
    st.dataframe(
        df,
        column_config={
            "ID": "用户ID",
            "用户名": "用户名",
            "管理员": "管理员权限",
            "创建时间": "创建时间"
        },
        hide_index=True,
        use_container_width=True
    )
    
    # 用户权限管理
    st.subheader("修改用户权限")
    selected_user_id = st.number_input("输入用户ID", min_value=1)
    if selected_user_id:
        user = db_api.db_get_user_by_id(selected_user_id)
        if user:
            is_admin = st.checkbox("设为管理员", value=user.is_admin)
            if st.button("更新权限"):
                try:
                    # 更新用户权限
                    db_api.db_update_user(selected_user_id, is_admin=is_admin)
                    st.success("权限更新成功！")
                    st.rerun()
                except Exception as e:
                    st.error(f"更新失败：{str(e)}")
        else:
            st.error("用户不存在")

# 默认登录函数
def default_login():
    """使用默认账号登录，用于开发测试"""
    if not st.session_state.get('authenticated'):
        st.session_state.authenticated = True
        st.session_state.current_user = {
            "user_id": 1,
            "username": "admin",
            "role": "admin"
        }

def main():
    """主程序入口"""
    # 开发模式：使用默认登录
    # default_login()  # 取消注释以启用默认登录
    
    # 正常模式：使用认证管理器
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

if __name__ == "__main__":
    try:
        # 直接启动应用
        main()
    except Exception as e:
        st.error(f"应用启动失败：{str(e)}")
        print(f"错误详情：{str(e)}")