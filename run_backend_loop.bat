@echo off
REM ============================================================
REM  YouDub backend supervisor
REM  Keeps the API on :8000 running. The worker exits with code 3
REM  when it can no longer spawn ffmpeg (a state this host
REM  occasionally gets into); this loop starts a fresh process and
REM  the task is resumed automatically on startup.
REM ============================================================
cd /d "%~dp0"

:loop
echo.
echo [supervisor] starting YouDub backend ... %DATE% %TIME%
set PYTHONPATH=
set PYTHONDONTWRITEBYTECODE=
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
echo [supervisor] backend exited with code %ERRORLEVEL%
timeout /t 3 /nobreak >nul
goto loop
