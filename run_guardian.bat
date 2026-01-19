@echo off
TITLE GUARDIAN AI - FULL LIVE MODE
COLOR 0A

echo ==================================================
echo   🔥 GUARDIAN AI TRADING SYSTEM
echo   FULL LIVE MODE (Alpha PPO + Guardian)
echo ==================================================
echo.

:: Mode Selection
echo Select Trading Mode:
echo   [1] Shadow   - PPO thinks, Rule executes (SAFE)
echo   [2] Hybrid   - PPO assists when confident
echo   [3] Full     - PPO + Guardian LIVE (RECOMMENDED)
echo.
set /p MODE_CHOICE="Enter choice (1-3) [default=3]: "

if "%MODE_CHOICE%"=="1" (
    set ALPHA_MODE=shadow
) else if "%MODE_CHOICE%"=="2" (
    set ALPHA_MODE=hybrid
) else (
    set ALPHA_MODE=full
)

echo.
echo --------------------------------------------------
echo   MODE: %ALPHA_MODE%
echo --------------------------------------------------
echo.
echo   Components Active:
echo   [✓] Alpha PPO V1     : LIVE
echo   [✓] Guardian PPO     : LIVE
echo   [✓] Rule-based       : FALLBACK
echo   [✓] Auto-Train       : ON
echo   [✓] Auto-Tuner       : ON
echo   [✓] KillSwitch       : ON
echo   [✓] Daily Governance : ON
echo --------------------------------------------------
echo.

:: Launch Dashboard
echo [1/2] Starting Dashboard...
start "Guardian Dashboard" /MIN cmd /c "streamlit run dashboard/app.py --server.port 8501"

:: Launch Watchdog with Alpha Mode
echo [2/2] Starting Guardian Watchdog...
echo.
echo ==================================================
echo   Press Ctrl+C to stop the system safely.
echo ==================================================
echo.

python guardian_watchdog.py --alpha-mode %ALPHA_MODE%

pause
