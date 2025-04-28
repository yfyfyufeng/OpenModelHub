import subprocess
import time
import shutil
import socket


def run_command(cmd, cwd=None, input_text=None, wait=True):
    print(f"开始执行: {cmd} 🚀")
    try:
        if wait:
            subprocess.run(
                cmd,
                shell=True,
                check=True,
                text=True,
                cwd=cwd,
                input=input_text
            )
        else:
            subprocess.Popen(
                cmd,
                shell=True,
                cwd=cwd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        print(f"完成: {cmd} ✅\n")
    except subprocess.CalledProcessError as e:
        print(f"指令失败: {cmd} ❌")
        print(f"错误信息: {e}")
        exit(1)


def get_local_ip():
    """获取本机局域网IP"""
    try:
        # 建立一个UDP连接，只是为了拿到本地IP（不发送数据）
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))  # 连接外网，不发送数据
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        print(f"获取本机IP失败: {e} ❌")
        return "127.0.0.1"  # 退回本地环回地址


if __name__ == "__main__":
    # 检测python3或者python
    python_cmd = None
    if shutil.which("python3"):
        python_cmd = "python3"
    elif shutil.which("python"):
        python_cmd = "python"
    else:
        print("错误：找不到 python3 或 python 命令 ❌")
        exit(1)

    # 获取本机局域网IP
    local_ip = get_local_ip()
    print(f"本机IP地址: {local_ip} 🌟")

    # 第一步：运行 load_data.py，并输入"1"
    run_command(f"{python_cmd} database/load_data.py", input_text="1\n")

    # 第二步：进入 security 目录，后台运行 Go 程序
    run_command("go run main.go", cwd="security", wait=False)

    # 给 go 程序一点时间启动
    time.sleep(5)

    # 第三步：运行 Streamlit 应用，绑定到本机IP
    run_command(f"streamlit run ./frontend/app.py --server.address {local_ip}")
