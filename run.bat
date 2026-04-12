@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONPATH=src
python run.py
