#!/usr/bin/env python3
"""
LSTM价格预测系统启动器
解决PyTorch与Streamlit的兼容性问题
"""

import os
import sys
import subprocess
import importlib.util
import time

def setup_environment():
    """设置环境变量和配置"""
    print("🔧 正在配置环境...")
    
    # 设置环境变量解决兼容性问题
    os.environ['STREAMLIT_CONFIG_WATCHER_TYPE'] = 'polling'
    os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '0'
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # 添加当前目录到Python路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

def check_python_version():
    """检查Python版本"""
    print("🐍 检查Python版本...")
    if sys.version_info < (3, 8):
        print("❌ 错误: 需要Python 3.8或更高版本")
        print(f"当前版本: {sys.version}")
        return False
    print(f"✅ Python版本正常: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """检查并安装依赖"""
    print("📦 检查依赖包...")
    
    required_packages = [
        'streamlit',
        'torch',
        'pandas',
        'numpy',
        'scikit-learn',
        'plotly'
    ]
    
    missing_packages = []
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"⚠️  缺失依赖包: {', '.join(missing_packages)}")
        print("💾 正在安装依赖包...")
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'

        def _pip_install(extra_args: list[str]):
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt', '--upgrade', *extra_args
            ], env=env)

        try:
            _pip_install([])
            print("✅ 依赖包安装完成")
        except subprocess.CalledProcessError:
            # Windows 常见权限问题，尝试 --user 方案
            print("⚠️  安装失败，尝试使用 --user 选项重新安装 …")
            try:
                _pip_install(['--user'])
                print("✅ 依赖包安装完成 (--user)")
            except subprocess.CalledProcessError:
                print("❌ 依赖包安装仍然失败")
                print("�� 请手动运行: pip install -r requirements.txt --user --force-reinstall")
                return False
    else:
        print("✅ 所有依赖包已安装")
    
    return True

def fix_torch_compatibility():
    """修复PyTorch兼容性问题"""
    print("🔧 修复PyTorch兼容性...")
    
    try:
        import torch
        
        # 修复torch.classes的__path__问题
        if hasattr(torch, 'classes'):
            if not hasattr(torch.classes, '__path__'):
                torch.classes.__path__ = []
                print("✅ 已修复torch.classes.__path__问题")
        
        # 设置torch相关环境
        if hasattr(torch, 'set_num_threads'):
            torch.set_num_threads(1)  # 避免多线程冲突
            
        print("✅ PyTorch兼容性修复完成")
        return True
        
    except ImportError:
        print("❌ 无法导入PyTorch")
        return False
    except Exception as e:
        print(f"⚠️  PyTorch兼容性修复警告: {e}")
        return True  # 继续运行，可能不影响

def create_streamlit_config():
    """创建Streamlit配置文件"""
    config_dir = ".streamlit"
    config_file = os.path.join(config_dir, "config.toml")
    
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    
    if not os.path.exists(config_file):
        print("📝 创建Streamlit配置文件...")
        config_content = """[server]
fileWatcherType = "polling"
headless = false
port = 8502

[browser]
gatherUsageStats = false

[runner]
magicEnabled = false

[global]
developmentMode = false
"""
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
        print("✅ 配置文件创建完成")

def check_port_available(port):
    """检查端口是否可用"""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex(('localhost', port))
            return result != 0  # 端口不可用则返回True
    except:
        return True

def kill_existing_streamlit():
    """终止现有的Streamlit进程"""
    try:
        import psutil
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'python' in proc.info['name'].lower():
                    cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                    if 'streamlit' in cmdline and 'price_prediction_system.py' in cmdline:
                        print(f"🔄 终止现有进程: PID {proc.info['pid']}")
                        proc.terminate()
                        proc.wait(timeout=3)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except ImportError:
        # 如果没有psutil，使用系统命令
        try:
            if os.name == 'nt':  # Windows
                subprocess.run(['taskkill', '/F', '/IM', 'python.exe'], 
                             capture_output=True, text=True)
        except:
            pass

def start_streamlit():
    """启动Streamlit应用"""
    print("\n📱 正在启动LSTM价格预测系统...")
    
    # 检查端口是否已被占用
    if not check_port_available(8502):
        print("⚠️  端口8502已被占用，正在尝试终止现有进程...")
        kill_existing_streamlit()
        time.sleep(2)
        
        if not check_port_available(8502):
            print("❌ 无法释放端口8502，请手动关闭其他Streamlit进程")
            print("💡 或者等待几秒钟后重试")
            return False
    
    print("🌐 浏览器将自动打开页面...")
    print("🔗 手动访问: http://localhost:8502")
    print("=" * 50)
    
    # 启动Streamlit
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'price_prediction_system.py',
            '--server.port', '8502',
            '--server.headless', 'false',
            '--browser.gatherUsageStats', 'false',
            '--server.fileWatcherType', 'polling'
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Streamlit启动失败: {e}")
        return False
    except KeyboardInterrupt:
        print("\n🔄 用户中断，正在退出...")
        return True
    
    return True

def main():
    """主函数"""
    print("=" * 50)
    print("    🔮 LSTM价格预测系统 - Python启动器")
    print("=" * 50)
    print()
    
    # 环境设置
    setup_environment()
    
    # 检查Python版本
    if not check_python_version():
        input("按回车键退出...")
        return 1
    
    # 检查依赖
    if not check_dependencies():
        input("按回车键退出...")
        return 1
    
    # 修复PyTorch兼容性
    if not fix_torch_compatibility():
        print("⚠️  PyTorch兼容性问题，但将尝试继续运行...")
    
    # 创建配置文件
    create_streamlit_config()
    
    print("✅ 所有检查完成！")
    print()
    
    # 启动应用
    success = start_streamlit()
    
    print("\n🔄 应用已退出")
    if success:
        print("💡 如需重新启动，请再次运行此脚本")
    
    input("按回车键退出...")
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 