@echo off
chcp 65001 > nul
setlocal EnableDelayedExpansion

REM ========================================
REM 커밋 메시지 처리
REM ========================================

if "%~1"=="" (
    set COMMIT_MSG=auto commit %DATE% %TIME%
) else (
    set COMMIT_MSG=%~1
)

echo 📝 Commit message: "!COMMIT_MSG!"
echo.

echo ========================================
echo ⬆️  Submodule push start
echo ========================================

REM ========================================
REM 서브모듈 순회
REM ========================================

for /f "tokens=2" %%S in ('git submodule status') do (

    echo.
    echo 📦 [서브모듈] %%S
    echo ----------------------------------------

    pushd %%S

    REM detached HEAD 방지
    git checkout main > nul 2>&1

    REM 🔹 로컬 변경 여부 확인
    git status --porcelain > nul
    if not errorlevel 1 (
        echo ✏️  로컬 변경 감지 → pull 생략
    ) else (
        echo ⬇️  로컬 변경 없음 → pull 수행
        git pull --rebase origin main
        if errorlevel 1 (
            echo ❌ pull 실패: %%S
            popd
            exit /b 1
        )
    )

    REM 🔹 다시 변경 여부 확인 (pull 결과 포함)
    git status --porcelain > nul
    if not errorlevel 1 (
        echo ✏️  commit & push
        git add -A
        git commit -m "!COMMIT_MSG!"
        git push origin main
        if errorlevel 1 (
            echo ❌ push 실패: %%S
            popd
            exit /b 1
        )
    ) else (
        echo ✅ 변경 없음
    )

    popd
)

echo.
echo ========================================
echo ⬆️  Main repository push
echo ========================================

REM ========================================
REM 메인 repo 처리
REM ========================================

git status --porcelain > nul
if not errorlevel 1 (
    echo ✏️  메인 repo 변경 감지
    git add -A
    git commit -m "!COMMIT_MSG!"
)

git push origin main
if errorlevel 1 (
    echo ❌ 메인 repo push 실패
    exit /b 1
)

echo.
echo 🎉 모든 push 완료
echo ========================================

endlocal
