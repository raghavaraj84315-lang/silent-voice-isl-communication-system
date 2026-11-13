@echo off
chcp 65001 >nul
title Silent Voice - GUI Application
echo ========================================================
echo   Starting Silent Voice Desktop Application
echo ========================================================
cd /d "%~dp0"
"%~dp0venv\Scripts\python.exe" "%~dp0app_gui.py"
pause
