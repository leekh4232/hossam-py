@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

echo ========================================
echo ⬆️  Git Push (with Submodules)
echo ========================================

REM Git 저장소 확인
git rev-parse --is-inside-work-tree > nul 2>&1
if errorlevel 1 (
    echo ❌ Git 저장소 아님
    pause
    exit /b 1
)

REM -------------------------------
REM 서브모듈 처리
REM -------------------------------
for /f "tokens=2" %%S in ('git submodule status') do (
    echo.
    echo 📦 [SUBMODULE] %%S
    pushd %%S

    REM 현재 브랜치 확인
    for /f %%B in ('git branch --show-current') do set BRANCH=%%B

    if "!BRANCH!"=="" (
        echo ⚠️ detached HEAD 감지 → main 브랜치로 전환
        git switch main || git checkout main
        if errorlevel 1 (
            echo ❌ 브랜치 전환 실패
            popd
            exit /b 1
        )
    )

    REM 변경사항 확인
    git status --porcelain > nul
    if errorlevel 1 (
        echo ℹ️ 변경 없음
    ) else (
        git add -A
        git commit -m "auto update"
        git push
        if errorlevel 1 (
            echo ❌ 서브모듈 push 실패
            popd
            exit /b 1
        )
    )

    popd
)

REM -------------------------------
REM 메인 저장소 처리
REM -------------------------------
echo.
echo 📦 [MAIN REPO]

git add -A
git commit -m "update submodules" > nul 2>&1
git push
if errorlevel 1 (
    echo ❌ 메인 repo push 실패
    pause
    exit /b 1
)

echo.
echo ========================================
echo ✅ 모든 push 완료
echo ========================================
pause
