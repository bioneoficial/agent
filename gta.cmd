@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Git Terminal Assistant - Windows CMD launcher

REM Resolve script directory (ends with backslash)
set "SCRIPT_DIR=%~dp0"

REM Load environment variables from .env if present
set "DOTENV=%SCRIPT_DIR%.env"
if exist "%DOTENV%" (
  for /f "usebackq tokens=* delims=" %%A in ("%DOTENV%") do (
    set "line=%%A"
    if not "!line!"=="" (
      if /i not "!line:~0,1!"=="#" (
        for /f "tokens=1* delims==" %%K in ("!line!") do (
          set "key=%%~K"
          set "val=%%~L"
          if defined val (
            rem Strip surrounding double quotes if present
            for %%Z in ("!val!") do set "val=%%~Z"
            rem Strip surrounding single quotes if present
            if "!val:~0,1!"=="'" set "val=!val:~1!"
            if "!val:~-1!"=="'" set "val=!val:~0,-1!"
          )
          set "!key!=!val!"
        )
      )
    )
  )
  echo Loaded environment from %DOTENV%
  if defined LLM_PROVIDER echo Using provider: %LLM_PROVIDER% with model: %LLM_MODEL%
)

set "MAINSCRIPT=%SCRIPT_DIR%main.py"

REM Prefer virtual environment Python
set "PYEXE="
if exist "%SCRIPT_DIR%venv\Scripts\python.exe" set "PYEXE=%SCRIPT_DIR%venv\Scripts\python.exe"
if not defined PYEXE if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" set "PYEXE=%SCRIPT_DIR%.venv\Scripts\python.exe"
if not defined PYEXE (
  where py >nul 2>nul && set "PYEXE=py"
)
if not defined PYEXE (
  where python >nul 2>nul && set "PYEXE=python"
)
if not defined PYEXE (
  where python3 >nul 2>nul && set "PYEXE=python3"
)
if not defined PYEXE set "PYEXE=python"

REM Execute
if /i "%PYEXE%"=="py" (
  %PYEXE% "%MAINSCRIPT%" %*
) else (
  "%PYEXE%" "%MAINSCRIPT%" %*
)

endlocal
