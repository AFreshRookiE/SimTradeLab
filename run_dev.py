#!/usr/bin/env python3
"""
开发模式运行脚本，监控代码变化并自动重启服务
"""

import os
import sys
import time
import subprocess
import signal
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class CodeChangeHandler(FileSystemEventHandler):
    def __init__(self, restart_callback):
        self.restart_callback = restart_callback
        self.last_modified = 0
        self.cooldown = 0.5  # 冷却时间，避免频繁重启

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.py'):
            current_time = time.time()
            if current_time - self.last_modified > self.cooldown:
                self.last_modified = current_time
                print(f"文件变化: {event.src_path}")
                self.restart_callback()

class DevServer:
    def __init__(self):
        self.process = None
        self.observer = None

    def start_server(self):
        """启动run.py服务"""
        if self.process:
            self.stop_server()
        
        print("启动服务...")
        self.process = subprocess.Popen(
            [sys.executable, "run.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # 显示服务启动输出
        def read_output():
            while self.process and self.process.poll() is None:
                line = self.process.stdout.readline()
                if line:
                    print(line.rstrip())
        
        import threading
        threading.Thread(target=read_output, daemon=True).start()

    def stop_server(self):
        """停止服务"""
        if self.process:
            print("停止服务...")
            try:
                # 在Windows上使用Terminate
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("强制终止服务...")
                self.process.kill()
            finally:
                self.process = None

    def restart_server(self):
        """重启服务"""
        self.stop_server()
        self.start_server()

    def run(self):
        """运行开发服务器"""
        # 启动服务
        self.start_server()
        
        # 监控代码变化
        event_handler = CodeChangeHandler(self.restart_server)
        self.observer = Observer()
        self.observer.schedule(event_handler, "src", recursive=True)
        self.observer.start()
        
        print("开发服务器已启动，监控代码变化...")
        print("按 Ctrl+C 停止服务")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("正在停止开发服务器...")
        finally:
            self.stop_server()
            if self.observer:
                self.observer.stop()
                self.observer.join()

if __name__ == "__main__":
    server = DevServer()
    server.run()
