@echo off
rem Wallex Agentic Trader - runs the engine in its own window, independent of any IDE/app.
rem Restarts automatically after a crash; stops when the challenge is over (exit 0)
rem or when another engine is already running (exit 3).
title Wallex Agentic Trader (TRADING_MODE from .env)
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

:loop
echo [%date% %time%] starting engine...
.venv\Scripts\python.exe -m src.engine
set CODE=%errorlevel%
if %CODE%==0 goto done
if %CODE%==3 goto done
echo [%date% %time%] engine exited with code %CODE% - restarting in 15 seconds (close this window to stop)
timeout /t 15 /nobreak >nul
goto loop

:done
echo [%date% %time%] engine stopped (code %CODE%).
pause
