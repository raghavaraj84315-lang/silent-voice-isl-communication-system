@echo off
chcp 65001 >nul
title Silent Voice - Web Dashboard
echo ========================================================
echo   Starting Silent Voice Web Application on http://127.0.0.1:8000
echo ========================================================
cd /d "%~dp0"
"%~dp0venv\Scripts\python.exe" "%~dp0web_app.py"
pause
