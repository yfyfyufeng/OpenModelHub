import asyncio
import subprocess
import sys
from pathlib import Path
import os

# 添加项目根目录到系统路径
current_dir = Path(__file__).parent
project_root = current_dir
sys.path.extend([str(project_root), str(project_root/"database")])
sys.path.extend([str(project_root), str(project_root/"frontend")])

from database.load_data import main as load_data
from database.database_interface import *
async def main():
    """主函数：加载数据并启动应用"""
    print("=== Starting OpenModelHub ===")

    # 1. 清空数据
    print("\n1. Clearing old data...")
    try:
        await drop_database()
    except Exception as e:
        print(f"❌ Error when clearing old data: {str(e)}")
        return
    # 2. 加载数据
    print("\n2. Loading data...")
    try:
        await load_data()
        print("✅ finish loading")
    except Exception as e:
        print(f"❌ Error when loading data: {str(e)}")
        return
    
    # 3. 启动Streamlit应用
    print("\n3. Starting the app...")
    try:
        # 获取app.py的绝对路径
        app_path = str(project_root / "frontend" / "app.py")
        
        # 启动Streamlit
        process = subprocess.Popen(
            ["streamlit", "run", app_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 等待应用启动
        print("✅ 应用启动成功")
        print("\n=== OpenModelHub 已就绪 ===")
        
        # 保持进程运行
        process.wait()
        
    except Exception as e:
        print(f"❌ 应用启动失败: {str(e)}")
        return

if __name__ == "__main__":
    # 设置工作目录为项目根目录
    os.chdir(str(project_root))
    
    # 运行主函数
    asyncio.run(main()) 