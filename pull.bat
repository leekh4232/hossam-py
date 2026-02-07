@echo off
chcp 65001 > nul
setlocal EnableDelayedExpansion

echo ========================================
echo ⬇️  Git pull pipeline start
echo ========================================

REM ========================================
REM 1️⃣ 서브모듈 존재 여부 검사
REM ========================================

git submodule status > nul 2>&1
if errorlevel 1 (
    set HAS_SUBMODULE=0
) else (
    for /f %%i in ('git submodule status') do (
        set HAS_SUBMODULE=1
        goto :CHECK_DONE
    )
    set HAS_SUBMODULE=0
)

:CHECK_DONE

REM ========================================
REM 2️⃣ 서브모듈이 있는 경우
REM ========================================

if "%HAS_SUBMODULE%"=="1" (

    echo 📦 서브모듈 감지됨
    echo ----------------------------------------

    REM 🔹 서브모듈 먼저 pull
    for /f "tokens=2" %%S in ('git submodule status') do (

        echo.
        echo ⬇️  [서브모듈 pull] %%S
        echo ----------------------------------------

        pushd %%S

        REM detached HEAD 방지
        git checkout main > nul 2>&1

        git pull --rebase origin main
        if errorlevel 1 (
            echo ❌ 서브모듈 pull 실패: %%S
            popd
            exit /b 1
	    pause
        )

        popd
    )

    echo.
    echo ⬇️  메인 저장소 pull (서브모듈 포인터 갱신 포함)
    echo ----------------------------------------

    git pull
    if errorlevel 1 (
        echo ❌ 메인 저장소 pull 실패
        exit /b 1
	pause
    )

) else (

REM ========================================
REM 3️⃣ 서브모듈이 없는 경우
REM ========================================

    echo 📦 서브모듈 없음
    echo ----------------------------------------
    echo ⬇️  메인 저장소만 pull

    git pull
    if errorlevel 1 (
        echo ❌ 메인 저장소 pull 실패
        exit /b 1
	pause
    )
)

echo.
echo ✅ Git pull 완료
echo ========================================

endlocal

pause
