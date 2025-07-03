# GitHub Secrets 配置说明

## 微信推送Token配置问题解决方案

当您看到以下错误信息时：
```
Error: 训练和预测过程出错: 'period'
Error: 微信推送Token未配置
```

## 问题修复

### 1. 'period' 错误已修复
- 已将代码中的 `df.iloc[-1]['period']` 修正为 `df.iloc[-1]['Seq']`
- 这个错误是因为数据框中期号字段名称不匹配导致的

### 2. 微信推送Token配置

#### 方式一：使用GitHub Secrets（推荐）

1. 进入您的GitHub仓库
2. 点击 `Settings` -> `Secrets and variables` -> `Actions`
3. 点击 `New repository secret`
4. 添加以下Secrets：

| Secret名称 | 描述 | 示例值 |
|-----------|------|--------|
| `WXPUSHER_APP_TOKEN` | 微信推送应用Token | `AT_xxxxxxxxxxxxxxxxxx` |
| `WXPUSHER_USER_UIDS` | 用户UID（可选） | `UID_xxxxxxxxxxxxxxxxxx` |
| `WXPUSHER_TOPIC_IDS` | 主题ID（可选） | `123,456` |

#### 方式二：使用默认Token（已配置）

如果您不配置GitHub Secrets，系统会自动使用默认的Token：
- 默认用户UID: `UID_yYObqdMVScIa66DGR2n2PCRFL10w`
- 默认APP_TOKEN: 已内置在代码中

#### 获取微信推送Token

1. 访问 [WxPusher官网](http://wxpusher.zjiecode.com/)
2. 注册账号并创建应用
3. 获取APP_TOKEN
4. 获取用户UID（关注您的应用）

## 错误处理改进

系统现在已经改进了错误处理机制：

1. **微信推送失败不会中断预测流程**
   - 当微信推送失败时，系统会记录警告但继续执行预测
   - 预测结果仍会正常保存到历史记录

2. **更好的Token验证**
   - 检查Token格式是否有效
   - 提供更详细的错误信息

3. **GitHub Actions优化**
   - 添加了配置检查步骤
   - 更清晰的日志输出

## 测试配置

运行以下命令测试微信推送配置：

```bash
python ssq_wxpusher.py
```

或者运行系统测试：

```bash
python ssq_automation.py --test
```

## 注意事项

1. 如果您不需要微信推送功能，系统仍会正常运行预测功能
2. 建议配置自己的微信推送Token以确保推送功能正常
3. 默认Token可能会因为使用人数过多而失效

## 成功运行标志

当看到以下日志时，说明系统运行正常：

```
::notice title=预测完成::第XXXX期预测号码: [xx, xx, xx, xx, xx, xx, xx]
::notice title=流程完成::双色球LSTM预测系统执行成功
```

如果仍有问题，请检查GitHub Actions日志中的详细错误信息。 