@echo off
REM Build all sub-projects, copy wheel files to root, and clean all projects

echo ========================================
echo Building all sub-projects...
echo ========================================
python build.py --build
if %errorlevel% neq 0 (
    echo Build failed!
    exit /b 1
)

echo.
echo ========================================
echo Copying wheel files to project root...
echo ========================================
copy dist\*.whl . >nul
if %errorlevel% neq 0 (
    echo Failed to copy wheel files!
    exit /b 1
)
echo Wheel files copied successfully.

echo.
echo ========================================
echo Cleaning all projects...
echo ========================================
python build.py --clean
if %errorlevel% neq 0 (
    echo Clean failed!
    exit /b 1
)

echo.
echo ========================================
echo All tasks completed successfully!
echo ========================================
echo.
echo Wheel files are now in the project root:
dir /b *.whl

exit /b 0
