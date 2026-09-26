@echo off
REM ============================================================
REM  YouDub WebUI launcher - debug-friendly
REM  Use DEBUG=1 to pause at each step.
REM ============================================================

setlocal ENABLEDELAYEDEXPANSION
set "DEBUG=0"
if "%DEBUG%"=="1" set "PAUSE=pause >nul" else set "PAUSE=rem"

REM Do NOT change chcp at top - causes flash-close on some Windows.

cd /d "%~dp0"

echo.
echo ============================================================
echo  YouDub WebUI  D:\index\YouDub
echo ============================================================
echo.
%PAUSE%

REM ---------- 1. .env ----------
echo [1/6] Checking .env ...
if not exist ".env" (
    echo [ERR] .env missing. Run setup first.
    pause
    exit /b 1
)
echo OK .env found
%PAUSE%

REM ---------- 2. Python ----------
echo [2/6] Finding Python 3.12 ...
set "PY_EXE="
where py >nul 2>nul
if not errorlevel 1 (
    for /f "delims=" %%P in ('py -3.12 -c "import sys; print(sys.executable)" 2^>nul') do (
        set "PY_EXE=%%P"
    )
)
if "!PY_EXE!"=="" (
    where python >nul 2>nul
    if not errorlevel 1 (
        for /f "delims=" %%P in ('python -c "import sys; print(sys.executable)" 2^>nul') do (
            set "PY_EXE=%%P"
        )
    )
)
if "!PY_EXE!"=="" (
    echo [ERR] No Python. Install Python 3.12.
    pause
    exit /b 1
)
echo OK Python = !PY_EXE!
%PAUSE%

REM ---------- 3. venv ----------
echo [3/6] Checking .venv ...
if not exist ".venv\Scripts\python.exe" (
    echo First run, creating venv ...
    "!PY_EXE!" -m venv .venv
    if errorlevel 1 goto :err
    .venv\Scripts\python.exe -m pip install -U pip -i https://mirrors.aliyun.com/pypi/simple/
    .venv\Scripts\pip.exe install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
    if errorlevel 1 goto :err
    .venv\Scripts\pip.exe install -r requirements-pytorch-cu128.txt -i https://mirrors.aliyun.com/pypi/simple/
) else (
    echo OK venv exists
)
%PAUSE%

REM ---------- 4. node_modules ----------
echo [4/6] Checking apps\web\node_modules ...
if not exist "apps\web\node_modules" (
    echo First run, installing npm ...
    pushd apps\web
    call npm install --registry=https://registry.npmmirror.com
    set "RC=!errorlevel!"
    popd
    if !RC! neq 0 goto :err
) else (
    echo OK node_modules exists
)
%PAUSE%

REM ---------- 5. frontend production build ----------
echo [5/6] Checking frontend production build ...
if not exist "apps\web\.next\BUILD_ID" (
    echo Building frontend ^(first run^) ...
    pushd apps\web
    call npm run build
    set "RC=!errorlevel!"
    popd
    if !RC! neq 0 goto :err
) else (
    echo OK production build exists
    echo NOTE: after editing apps\web code, run "npm run build" in apps\web
    echo       ^(or delete apps\web\.next to force a rebuild^).
)
%PAUSE%

REM ---------- 6. start services ----------
echo [6/6] Starting backend on :8000 and frontend on :3000 ...
echo.

start "YouDubBackend8000" cmd /k "cd /d D:\index\YouDub && run_backend_loop.bat"
start "YouDubFrontend3000" cmd /k "cd /d D:\index\YouDub && npm --prefix apps\web run start"

echo ============================================================
echo  YouDub launched.
echo    Backend:  http://localhost:8000/docs
echo    Frontend: http://localhost:3000  ^(production mode^)
echo    Password: youdub2026
echo ============================================================
echo.
echo This window will close in 60 seconds. Close the two child
echo cmd windows to stop the services.
echo.

timeout /t 60 /nobreak >nul
exit /b 0

:err
echo.
echo [ERR] Setup failed. Read the message and check the steps above.
pause
exit /b 1