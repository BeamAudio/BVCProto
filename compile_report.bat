@echo off
echo Cleaning auxiliary files...
del docs\*.aux docs\*.log docs\*.out docs\*.toc 2>nul

echo Compiling BVC Technical Report (Pass 1)...
"C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe" -output-directory docs docs/BVC_Technical_Report.tex

echo Compiling BVC Technical Report (Pass 2)...
"C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe" -output-directory docs docs/BVC_Technical_Report.tex

if %errorlevel% neq 0 (
    echo.
    echo Error: pdflatex failed.
    pause
    exit /b %errorlevel%
)

echo.
echo Compilation successful! Output: docs/BVC_Technical_Report.pdf
pause