@echo off
chcp 65001 >nul
cd /d "%~dp0"
REM 数据在上两级目录 pagerank大作业\Data.txt；结果写入同级上级目录下的 实验结果\Res.txt
"%~dp0PageRank.exe" "%~dp0..\..\Data.txt" "%~dp0..\实验结果\Res.txt"
if errorlevel 1 pause
echo.
echo 已写入 ..\实验结果\Res.txt （相对于「可执行文件」文件夹）
pause
