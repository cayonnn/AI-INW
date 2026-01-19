@echo off
TITLE GUARDIAN AI - MULTI-SYMBOL (V4)
COLOR 0B

echo ==================================================
echo   🚀 GUARDIAN AI TRADING SYSTEM
echo   Multi-Symbol Mode (V4)
echo ==================================================
echo.

:: Launch Dashboard V4
echo [1/2] Starting Dashboard V4...
start "Guardian Dashboard V4" /MIN cmd /c "python -m streamlit run dashboard_v4.py --server.port 8501"

:: Launch Watchdog V4
echo [2/2] Starting Watchdog Supervisor V4...
echo.
echo --------------------------------------------------
echo   Monitoring: XAUUSD, EURUSD, BTCUSD, GBPJPY
echo   Press Ctrl+C to stop.
echo --------------------------------------------------
echo.

python guardian_watchdog_v4.py

pause
