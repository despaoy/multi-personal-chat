"""检查服务器环境：conda环境、GPU、vLLM进程、模型路径。

凭据通过 remote_config.connect_ssh() 从 LAB_SSH_KEY 或 LAB_PASS 读取，
避免在源码中硬编码密码。运行前请设置 LAB_HOST、LAB_USER 以及
LAB_SSH_KEY 或 LAB_PASS 环境变量。
"""
from remote_config import connect_ssh

# 1. 建立 SSH 连接（凭据来自环境变量）
ssh = connect_ssh()

# 1. lhm2 目录结构
print("=== lhm2 目录结构 ===")
stdin, stdout, stderr = ssh.exec_command("ls -la /home/szw/lhm2/")
print(stdout.read().decode().strip())

# 2. conda 环境列表
print("\n=== conda 环境 ===")
stdin, stdout, stderr = ssh.exec_command("source /home/szw/miniconda3/etc/profile.d/conda.sh 2>/dev/null; conda env list 2>/dev/null || ls /home/szw/lhm2/envs/ 2>/dev/null")
print(stdout.read().decode().strip())

# 3. lhm2/envs 下的环境
print("\n=== lhm2/envs 目录 ===")
stdin, stdout, stderr = ssh.exec_command("ls -la /home/szw/lhm2/envs/ 2>/dev/null")
print(stdout.read().decode().strip())

# 4. 检查 qqchat-gpu-qwen3 环境 Python 版本和关键包
print("\n=== qqchat-gpu-qwen3 环境关键包 ===")
cmd = (
    "source /home/szw/lhm2/envs/qqchat-gpu-qwen3/bin/activate && "
    "python --version && "
    "pip list 2>/dev/null | grep -iE '^(torch|vllm|transformers|peft|trl|datasets|bitsandbytes|fastapi|nonebot) '"
)
stdin, stdout, stderr = ssh.exec_command(cmd)
print(stdout.read().decode().strip())

# 5. GPU 状态
print("\n=== GPU 状态 ===")
stdin, stdout, stderr = ssh.exec_command("nvidia-smi --query-gpu=index,name,memory.used,memory.total,memory.free,temperature.gpu,utilization.gpu --format=csv,noheader,nounits")
print(stdout.read().decode().strip())

# 6. GPU 进程
print("\n=== GPU 进程 ===")
stdin, stdout, stderr = ssh.exec_command("nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv,noheader")
print(stdout.read().decode().strip())

# 7. vLLM 进程
print("\n=== vLLM 进程 ===")
stdin, stdout, stderr = ssh.exec_command("ps aux | grep -E 'vllm|api_server' | grep -v grep")
print(stdout.read().decode().strip()[:500])

# 8. 模型路径
print("\n=== 模型路径 ===")
stdin, stdout, stderr = ssh.exec_command("ls -la /home/szw/lhm2/runtime/models/ 2>/dev/null")
print(stdout.read().decode().strip())

# 9. LoRA 适配器
print("\n=== LoRA 适配器 ===")
stdin, stdout, stderr = ssh.exec_command("ls -la /home/szw/lhm2/runtime/loras/kisaki/ 2>/dev/null")
print(stdout.read().decode().strip())

# 10. 检查 gcc（之前 triton 编译失败）
print("\n=== gcc 版本 ===")
stdin, stdout, stderr = ssh.exec_command("gcc --version 2>&1 | head -1")
print(stdout.read().decode().strip())

# 11. 端口占用
print("\n=== 端口 8000/8001/8002 占用 ===")
stdin, stdout, stderr = ssh.exec_command("ss -tlnp 2>/dev/null | grep -E ':(8000|8001|8002)' || netstat -tlnp 2>/dev/null | grep -E ':(8000|8001|8002)'")
print(stdout.read().decode().strip())

ssh.close()
print("\n=== 检查完成 ===")
