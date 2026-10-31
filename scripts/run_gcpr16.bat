@echo off
setlocal
if defined GCPR16_PYTHON (
  "%GCPR16_PYTHON%" "%~dp0run_gcpr16.py" %*
  exit /b
)
if defined CONDA_EXE if exist "%CONDA_EXE%" goto conda_env
for %%C in (conda.exe) do set "CONDA_EXE=%%~$PATH:C"
if defined CONDA_EXE goto conda_env
if exist "%~d0\miniconda3\Scripts\conda.exe" set "CONDA_EXE=%~d0\miniconda3\Scripts\conda.exe"
if defined CONDA_EXE goto conda_env
if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" set "CONDA_EXE=%USERPROFILE%\miniconda3\Scripts\conda.exe"
if defined CONDA_EXE goto conda_env
echo Conda was not found. Set GCPR16_PYTHON to the shapepriors-gcpr16 environment Python.
exit /b 1
:conda_env
"%CONDA_EXE%" run --no-capture-output -n shapepriors-gcpr16 python "%~dp0run_gcpr16.py" %*
exit /b
