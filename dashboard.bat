@echo off
echo Starting Professional Dashboard V2...
cd /d "%~dp0"
python -m src.dashboard.dashboard_server --port 8000
pause
