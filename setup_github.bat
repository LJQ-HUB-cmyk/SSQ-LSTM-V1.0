@echo off
chcp 65001
echo ====================================
echo   双色球LSTM预测系统 - 首次GitHub设置
echo ====================================
echo.

echo [1/5] 初始化Git仓库...
git init

echo.
echo [2/5] 添加所有文件...
git add .

echo.
echo [3/5] 首次提交...
git commit -m "Initial commit: 双色球LSTM预测系统"

echo.
echo [4/5] 添加远程仓库...
git remote add origin https://github.com/LJQ-HUB-cmyk/SSQ-LSTM-V1.0.git
git branch -M main

echo.
echo [5/5] 推送到GitHub...
git push -u origin main

echo.
echo ✅ 首次设置完成！
echo.
echo 🔗 项目地址: https://github.com/LJQ-HUB-cmyk/SSQ-LSTM-V1.0
echo.
echo 📋 接下来需要配置GitHub Secrets：
echo    1. 访问: https://github.com/LJQ-HUB-cmyk/SSQ-LSTM-V1.0/settings/secrets/actions
echo    2. 添加 WXPUSHER_APP_TOKEN = AT_FInZJJ0mUU8xvQjKRP7v6omvuHN3Fdqw
echo    3. 添加 WXPUSHER_USER_UIDS = UID_yYObqdMVScIa66DGR2n2PCRFL10w
echo.
echo 📊 查看Actions: https://github.com/LJQ-HUB-cmyk/SSQ-LSTM-V1.0/actions
echo.
pause 