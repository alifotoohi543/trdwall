@echo off
rem Live dashboard at http://localhost:8501 (read-only view of the engine state).
title Wallex Agentic Trader - Dashboard
chcp 65001 >nul
cd /d "%~dp0"
.venv\Scripts\streamlit.exe run dashboard\app.py --server.address localhost --server.port 8501 --browser.gatherUsageStats false
pause
