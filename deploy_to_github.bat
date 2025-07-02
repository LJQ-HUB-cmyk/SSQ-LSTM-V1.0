@echo off
chcp 65001
echo ====================================
echo   双色球LSTM预测系统 - GitHub部署
echo ====================================
echo.

echo [1/4] 检查Git状态...
git status

echo.
echo [2/4] 添加所有文件...
git add .

echo.
echo [3/4] 提交更改...
set /p commit_msg="请输入提交信息 (默认: 更新双色球LSTM预测系统): "
if "%commit_msg%"=="" set commit_msg=更新双色球LSTM预测系统
git commit -m "%commit_msg%"

echo.
echo [4/4] 推送到GitHub...
git push origin main

echo.
echo ✅ 部署完成！
echo 🔗 项目地址: https://github.com/LJQ-HUB-cmyk/SSQ-LSTM-V1.0
echo 📊 Actions状态: https://github.com/LJQ-HUB-cmyk/SSQ-LSTM-V1.0/actions
echo.
pause 