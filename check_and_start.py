#!/usr/bin/env python3
"""
启动检查脚本 - 确保只有一个LSTM预测系统实例运行
"""
import os
import sys
import socket
import subprocess
import time

def check_port_available(port):
    """检查端口是否可用"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex(('localhost', port))
            return result != 0
    except:
        return True

def is_streamlit_running():
    """检查是否有Streamlit在运行"""
    try:
        import psutil
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'python' in proc.info['name'].lower():
                    cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                    if 'streamlit' in cmdline and 'price_prediction_system.py' in cmdline:
                        return True, proc.info['pid']
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return False, None
    except ImportError:
        # 简单的端口检查
        return not check_port_available(8502), None

def main():
    print("🔍 检查LSTM价格预测系统状态...")
    
    running, pid = is_streamlit_running()
    
    if running:
        print(f"✅ 系统已在运行 (PID: {pid if pid else '未知'})")
        print("🌐 访问地址: http://localhost:8502")
        print("\n❓ 选择操作:")
        print("1. 打开浏览器")
        print("2. 重启系统")
        print("3. 退出")
        
        choice = input("\n请输入选择 (1-3): ").strip()
        
        if choice == "1":
            try:
                import webbrowser
                webbrowser.open('http://localhost:8502')
                print("🌐 浏览器已打开")
            except:
                print("❌ 无法打开浏览器，请手动访问: http://localhost:8502")
        elif choice == "2":
            print("🔄 正在重启系统...")
            if pid:
                try:
                    import psutil
                    proc = psutil.Process(pid)
                    proc.terminate()
                    proc.wait(timeout=5)
                    print("✅ 已终止旧进程")
                except:
                    print("⚠️ 终止进程失败，将强制启动")
            
            time.sleep(2)
            subprocess.run([sys.executable, 'start_lstm_predictor.py'])
        else:
            print("👋 退出")
    else:
        print("💡 系统未运行，正在启动...")
        subprocess.run([sys.executable, 'start_lstm_predictor.py'])

if __name__ == "__main__":
    main() 