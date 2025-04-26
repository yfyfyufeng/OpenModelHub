import os
from pathlib import Path
import streamlit as st
from datetime import datetime
import frontend.database_api as db_api
import pandas as pd

def handle_file_upload(file, file_type: str) -> str:
    """Handle file upload and save to appropriate directory
    Args:
        file: Uploaded file object
        file_type: Type of file ('model' or 'dataset')
    Returns:
        str: Path to saved file
    """
    try:
        # Create data directory if it doesn't exist
        data_dir = Path("database/data")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.name}"
        file_path = data_dir / filename
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(file.getvalue())
        
        return str(file_path)
    except Exception as e:
        st.error(f"File upload failed: {str(e)}")
        return None

def handle_file_download(file_path: str, original_filename: str) -> bool:
    """Handle file download
    Args:
        file_path: Path to the file (not used in this implementation)
        original_filename: Original filename to use for download
    Returns:
        bool: True if download successful, False otherwise
    """
    try:
        # Create a simple text file with "hello world"
        file_data = b"hello world"
        st.download_button(
            label="Download File",
            data=file_data,
            file_name=original_filename,
            mime="text/plain"
        )
        return True
    except Exception as e:
        st.error(f"File download failed: {str(e)}")
        return False

def render_upload_section(file_type: str):
    """Render file upload section
    Args:
        file_type: Type of file ('model' or 'dataset')
    """
    with st.expander(f"Upload New {file_type.capitalize()}", expanded=False):
        with st.form(f"{file_type}_upload", clear_on_submit=True):
            if file_type == "model":
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
                            file_path = handle_file_upload(file, "model")
                            if file_path:
                                model_data = {
                                    "model_name": name,
                                    "param_num": param_num,
                                    "arch_name": arch_type,
                                    "media_type": media_type.lower(),
                                    "tasks": [task.lower() for task in selected_tasks],
                                    "trainname": train_type.lower(),
                                    "param": file_path
                                }
                                db_api.db_create_model(model_data)
                                st.success("Model uploaded successfully!")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Upload failed: {str(e)}")
            
            else:  # dataset
                name = st.text_input("Dataset Name*")
                desc = st.text_area("Description")
                media_type = st.selectbox("Media Type", ["TEXT", "IMAGE", "AUDIO", "VIDEO"])
                task_type = st.selectbox("Task Type", ["CLASSIFICATION", "DETECTION", "GENERATION"])
                file = st.file_uploader("Select Data File*", type=["txt", "csv"])
                
                if st.form_submit_button("Submit"):
                    if not name or not file:
                        st.error("Fields marked with * are required")
                    else:
                        try:
                            file_path = handle_file_upload(file, "dataset")
                            if file_path:
                                dataset_data = {
                                    "ds_name": name,
                                    "ds_size": len(file.getvalue()),
                                    "media": media_type.lower(),
                                    "task": [task_type.lower()],
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

def render_detail_view(item, item_type: str):
    """Render detail view for model or dataset
    Args:
        item: Model or dataset object
        item_type: Type of item ('model' or 'dataset')
    """
    st.markdown("---")
    st.subheader(f"{item_type.capitalize()} Details - {item.model_name if item_type == 'model' else item.ds_name}")
    
    # Display basic information
    st.write("**Basic Information:**")
    if item_type == "model":
        info_data = {
            "ID": item.model_id,
            "Name": item.model_name,
            "Architecture Type": item.arch_name.value if hasattr(item.arch_name, 'value') else item.arch_name,
            "Parameter Count": f"{item.param_num:,}",
            "Media Type": item.media_type.value if hasattr(item.media_type, 'value') else item.media_type,
            "Training Type": item.trainname.value if hasattr(item.trainname, 'value') else item.trainname
        }
    else:
        info_data = {
            "ID": item.ds_id,
            "Name": item.ds_name,
            "Size": f"{item.ds_size/1024:.1f}KB",
            "Media Type": item.media,
            "Description": item.description
        }
    
    st.table(pd.DataFrame(list(info_data.items()), columns=["Property", "Value"]))
    
    # Display related information
    st.write("**Related Information:**")
    if item_type == "model":
        tasks = [task.task_name.value if hasattr(task.task_name, 'value') else task.task_name for task in item.tasks]
        authors = []
        for author in item.authors:
            author_name = getattr(author, 'name', None) or getattr(author, 'author_name', None) or getattr(author, 'user_name', None)
            if author_name:
                authors.append(author_name)
        
        datasets = []
        for dataset in item.datasets:
            dataset_name = getattr(dataset, 'name', None) or getattr(dataset, 'ds_name', None) or getattr(dataset, 'dataset_name', None)
            if dataset_name:
                datasets.append(dataset_name)
        
        rel_data = {
            "Supported Tasks": ", ".join(tasks) if tasks else "No tasks",
            "Authors": ", ".join(authors) if authors else "No authors",
            "Related Datasets": ", ".join(datasets) if datasets else "No datasets"
        }
    else:
        tasks = [task.task.value if hasattr(task.task, 'value') else task.task for task in item.Dataset_TASK]
        rel_data = {
            "Task Types": ", ".join(tasks) if tasks else "No tasks"
        }
    
    st.table(pd.DataFrame(list(rel_data.items()), columns=["Type", "Name"]))
    
    # Download button
    if item_type == "model":
        file_path = item.param
        original_filename = f"{item.model_name}.pt"
    else:
        file_path = item.ds_name + ".txt"
        original_filename = f"{item.ds_name}.txt"
    
    handle_file_download(file_path, original_filename)
    
    # Return button
    if st.button("Back to List", key="back_to_list"):
        st.session_state.current_page = f"{item_type}s"
        st.rerun() 