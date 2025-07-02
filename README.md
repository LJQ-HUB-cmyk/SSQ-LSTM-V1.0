# 🎱 双色球LSTM预测系统

基于深度学习LSTM神经网络的双色球预测系统，具备自动化数据获取、模型训练、预测和微信推送功能。

## ✨ 主要功能

- 🔄 **自动数据获取**: 从网络自动获取最新双色球开奖数据
- 🧠 **LSTM深度学习**: 使用循环神经网络进行时序模式学习
- 🎯 **智能预测**: 基于历史数据预测下期号码
- 📊 **回测验证**: 自动验证历史预测准确率
- 📱 **微信推送**: 自动推送预测结果和验证报告
- 🤖 **GitHub Actions**: 全自动化定时运行

## 🚀 快速开始

### 项目部署到GitHub

**首次部署步骤：**

1. **初始化Git仓库**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: 双色球LSTM预测系统"
   ```

2. **添加远程仓库**
   ```bash
   git remote add origin https://github.com/LJQ-HUB-cmyk/SSQ-LSTM-V1.0.git
   git branch -M main
   ```

3. **推送到GitHub**
   ```bash
   git push -u origin main
   ```

**后续更新步骤：**
```bash
git add .
git commit -m "更新描述"
git push origin main
```

**🚀 快速部署脚本（Windows）：**

*首次设置：*
```bash
# 双击运行 setup_github.bat 或在命令行执行：
setup_github.bat
```

*后续更新：*
```bash
# 双击运行 deploy_to_github.bat 或在命令行执行：
deploy_to_github.bat
```

### 本地运行

1. **克隆项目**
   ```bash
   git clone https://github.com/LJQ-HUB-cmyk/SSQ-LSTM-V1.0.git
   cd SSQ-LSTM-V1.0
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **运行Web界面**
   ```bash
   streamlit run app.py
   ```

4. **运行自动化脚本**
   ```bash
   python ssq_automation.py
   ```

### GitHub Actions 自动化

本项目已配置GitHub Actions工作流，将在每周一、三、五的北京时间6:00自动运行。

#### 完整部署流程

1. **项目同步到GitHub**
   - 按照上述步骤将项目推送到 [SSQ-LSTM-V1.0](https://github.com/LJQ-HUB-cmyk/SSQ-LSTM-V1.0) 仓库

2. **配置GitHub Secrets**
   
   访问 https://github.com/LJQ-HUB-cmyk/SSQ-LSTM-V1.0/settings/secrets/actions
   
   添加以下环境变量：
   - `WXPUSHER_APP_TOKEN`:
   - `WXPUSHER_USER_UIDS`: 
   - `WXPUSHER_TOPIC_IDS`: (可选，如有主题ID)

3. **启用GitHub Actions**
   - 推送代码后，GitHub Actions会自动启用
   - 查看运行状态：https://github.com/LJQ-HUB-cmyk/SSQ-LSTM-V1.0/actions

4. **验证自动化**
   - 系统将在每周一、三、五的北京时间6:00自动运行
   - 可手动触发：Actions → Shuangseqiu LSTM Prediction System → Run workflow

## 📁 项目结构

```
SSQ-LSTM/
├── .github/workflows/
│   └── ssq-prediction.yml     # GitHub Actions工作流
├── app.py                     # Streamlit Web应用主程序
├── data_fetcher.py           # 数据获取模块
├── lstm_model.py             # LSTM模型核心算法
├── prediction_history.py     # 预测历史管理
├── ssq_wxpusher.py          # 微信推送模块
├── ssq_automation.py        # 自动化运行脚本
├── requirements.txt         # 项目依赖
├── setup_github.bat         # GitHub首次设置脚本
├── deploy_to_github.bat     # GitHub更新部署脚本
├── run.bat                  # Windows批处理脚本
└── README.md               # 项目说明
```

## 🛠️ 核心模块

### LSTM模型 (`lstm_model.py`)
- 基于TensorFlow/Keras实现的双向LSTM网络
- 支持模型保存和加载
- 自动数据预处理和标准化
- 留一法交叉验证回测

### 数据获取 (`data_fetcher.py`)
- 自动从网络获取最新开奖数据
- 本地数据缓存和管理
- 数据格式化和验证

### 预测历史 (`prediction_history.py`)
- 预测结果持久化存储
- 自动验证预测准确率
- 统计分析和报告生成

### 微信推送 (`ssq_wxpusher.py`)
- 预测报告自动推送
- 验证结果通知
- 错误异常报警
- 系统运行日报

## 📊 使用说明

### Web界面功能

1. **预测页面**: 
   - 选择训练模式（新训练/已保存模型）
   - 调整模型参数
   - 查看预测结果

2. **回测页面**:
   - 历史数据验证
   - 准确率统计
   - 性能评估

3. **历史记录**:
   - 预测历史查看
   - 命中统计分析

4. **统计分析**:
   - 数据可视化
   - 趋势分析图表

### 自动化运行

系统每周一、三、五自动执行以下流程：

1. 获取最新开奖数据
2. 验证历史预测结果
3. 训练/加载LSTM模型
4. 预测下期号码
5. 生成预测报告
6. 发送微信推送通知
7. 提交结果到Git仓库

## ⚙️ 配置说明

### 模型参数

- `window_length`: 时间窗口长度（默认7期）
- `training_periods`: 训练数据期数（默认300期）
- `epochs`: 训练轮数（默认1200轮）
- `batch_size`: 批次大小（默认150）
- `learning_rate`: 学习率（默认0.0001）

### 微信推送配置

需要申请WxPusher服务并配置相关参数：
1. 注册WxPusher账号
2. 创建应用获取APP_TOKEN
3. 获取用户UID或创建主题

## 📈 预测准确率

系统基于深度学习LSTM网络，能够学习历史开奖数据中的时序模式。虽然双色球本质上是随机事件，但本系统可以：

- 识别号码出现的周期性模式
- 学习不同号码间的关联关系
- 基于历史趋势进行智能预测

**免责声明**: 本系统仅供学习研究使用，预测结果仅供参考，不构成投注建议。彩票投注有风险，请理性参与。

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目！

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。