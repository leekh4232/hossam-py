@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

echo ========================================
echo ⬇️  Git Pull (with Submodules)
echo ========================================

REM Git 저장소 여부 확인
git rev-parse --is-inside-work-tree > nul 2>&1
if errorlevel 1 (
    echo ❌ Git 저장소가 아님
    pause
    exit /b 1
)

REM 상위 저장소 pull (서브모듈 포함)
echo.
echo 📦 [MAIN] git pull --recurse-submodules
git pull --recurse-submodules
if errorlevel 1 (
    echo ❌ main repo pull 실패
    pause
    exit /b 1
)

REM 서브모듈 최신 원격 기준으로 갱신
echo.
echo 🔄 [SUBMODULES] update --remote --recursive
git submodule update --remote --recursive
if errorlevel 1 (
    echo ❌ submodule update 실패
    pause
    exit /b 1
)

echo.
echo ========================================
echo ✅ 모든 pull 완료
echo ========================================
pause
