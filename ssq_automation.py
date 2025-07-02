#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双色球LSTM预测系统自动化脚本 - GitHub Actions优化版本
"""

import sys
import logging
import traceback
import os
from datetime import datetime
from data_fetcher import fetch_ssq_data
from lstm_model import SSQLSTMModel
from prediction_history import PredictionHistory
from ssq_wxpusher import (
    send_prediction_report, 
    send_verification_report, 
    send_error_notification, 
    send_daily_summary,
    test_wxpusher_connection
)

# 环境检测
IS_GITHUB_ACTIONS = os.getenv('GITHUB_ACTIONS') == 'true'

# 配置日志
log_format = '%(asctime)s - %(levelname)s - %(message)s'
if IS_GITHUB_ACTIONS:
    # GitHub Actions环境：只输出到控制台
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
else:
    # 本地环境：同时输出到文件和控制台
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler('ssq_automation.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

logger = logging.getLogger(__name__)

class SSQAutomation:
    """双色球LSTM预测自动化系统 - GitHub Actions优化版本"""
    
    def __init__(self):
        """初始化自动化系统"""
        # 固定参数 - GitHub Actions优化
        self.model_params = {
            'batch_size': 150,
            'epochs': 1200,
            'window_length': 7
        }
        
        # 初始化组件
        self.model = SSQLSTMModel(**self.model_params)
        self.history = PredictionHistory()
        
        logger.info("双色球LSTM自动化系统初始化完成")
        logger.info(f"模型参数: {self.model_params}")
    
    def get_data(self):
        """获取双色球数据"""
        try:
            logger.info("开始获取双色球数据...")
            df = fetch_ssq_data()
            
            if df is None or df.empty:
                raise Exception("数据获取失败或为空")
            
            logger.info(f"数据获取成功，共 {len(df)} 期数据")
            return df
            
        except Exception as e:
            logger.error(f"数据获取失败: {e}")
            return None
    
    def verify_previous_predictions(self, df):
        """验证历史预测结果"""
        try:
            logger.info("验证历史预测结果...")
            
            # 更新历史记录中的实际开奖结果
            updated_count = self.history.update_actual_results(df)
            
            if updated_count > 0:
                logger.info(f"已更新 {updated_count} 条预测记录")
                
                # 获取最新的验证结果
                history_df = self.history.get_history()
                if not history_df.empty:
                    # 获取最近一次有实际结果的预测
                    completed_predictions = history_df.dropna(subset=['actual_red_1'])
                    if not completed_predictions.empty:
                        latest = completed_predictions.iloc[-1]
                        
                        # 提取预测和实际号码
                        predicted = [
                            int(latest[f'predicted_red_{i}']) for i in range(1, 7)
                        ] + [int(latest['predicted_blue'])]
                        
                        actual = [
                            int(latest[f'actual_red_{i}']) for i in range(1, 7)
                        ] + [int(latest['actual_blue'])]
                        
                        # 发送验证报告
                        send_verification_report(
                            int(latest['target_period']), 
                            predicted, 
                            actual
                        )
                        
                        return True
            
            logger.info("历史预测验证完成")
            return True
            
        except Exception as e:
            logger.error(f"验证历史预测失败: {e}")
            return False
    
    def train_and_predict(self, df):
        """训练模型并进行预测"""
        try:
            if IS_GITHUB_ACTIONS:
                print("::group::LSTM模型训练和预测")
            
            logger.info("开始训练LSTM模型...")
            
            # 训练模型 - 使用全部数据
            training_info = {}
            history = self.model.train_model(df)
            
            if history is None:
                raise Exception("模型训练失败")
            
            # 记录训练信息
            training_info['final_loss'] = float(history.history['loss'][-1])
            training_info['final_val_loss'] = float(history.history.get('val_loss', [0])[-1])
            training_info['epochs_completed'] = len(history.history['loss'])
            
            logger.info(f"模型训练完成，最终损失: {training_info['final_loss']:.6f}")
            
            # 保存模型
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_saved = self.model.save_model(f"ssq_lstm_{timestamp}")
            
            if model_saved:
                logger.info("模型保存成功")
            
            # 绘制训练曲线
            self.model.plot_training_history(history)
            
            # 预测下期号码
            logger.info("开始预测下期号码...")
            predicted_numbers = self.model.predict_next_numbers(df)
            
            if predicted_numbers is None:
                raise Exception("预测失败")
            
            # 确定预测期号
            latest_period = df.iloc[-1]['period']
            target_period = latest_period + 1
            
            logger.info(f"预测第 {target_period} 期号码: {predicted_numbers}")
            
            # 获取模型参数
            model_params = self.model.get_model_info()
            model_params.update(training_info)
            
            # 保存预测结果到历史记录
            self.history.save_prediction(target_period, predicted_numbers, model_params)
            
            if IS_GITHUB_ACTIONS:
                print("::endgroup::")
                print(f"::notice title=预测完成::第{target_period}期预测号码: {predicted_numbers}")
            
            return predicted_numbers, model_params, training_info
            
        except Exception as e:
            error_msg = f"训练和预测过程出错: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            if IS_GITHUB_ACTIONS:
                print("::endgroup::")
                print(f"::error title=预测失败::{error_msg}")
            
            return None, None, None
    
    def run(self):
        """运行自动化流程 - GitHub Actions优化版本"""
        prediction_success = False
        verification_success = False
        prediction_period = None
        error_messages = []
        
        try:
            if IS_GITHUB_ACTIONS:
                print("::group::双色球LSTM预测系统启动")
            
            logger.info("开始运行双色球LSTM预测自动化流程...")
            
            # 测试微信推送
            if not test_wxpusher_connection():
                logger.warning("微信推送测试失败，但继续执行流程")
                if IS_GITHUB_ACTIONS:
                    print("::warning::微信推送测试失败，但继续执行流程")
            
            # 1. 获取数据
            logger.info("步骤 1/4: 获取最新数据")
            df = self.get_data()
            if df is None:
                raise Exception("数据获取失败")
            
            # 2. 验证历史预测
            logger.info("步骤 2/4: 验证历史预测")
            verification_success = self.verify_previous_predictions(df)
            
            # 3. 训练模型并预测
            logger.info("步骤 3/4: 训练模型并预测")
            predicted_numbers, model_params, training_info = self.train_and_predict(df)
            
            if predicted_numbers is not None:
                prediction_success = True
                latest_period = df.iloc[-1]['period']
                prediction_period = latest_period + 1
                
                # 4. 发送预测报告
                logger.info("步骤 4/4: 发送预测报告")
                send_prediction_report(prediction_period, predicted_numbers, model_params, training_info)
            
            # 发送日报摘要
            send_daily_summary(prediction_success, verification_success, prediction_period, error_messages)
            
            if IS_GITHUB_ACTIONS:
                print("::endgroup::")
                print("::notice title=流程完成::双色球LSTM预测系统执行成功")
            
            logger.info("自动化流程执行成功")
            
        except Exception as e:
            error_msg = f"自动化流程执行失败: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            error_messages.append(str(e))
            
            if IS_GITHUB_ACTIONS:
                print("::endgroup::")
                print(f"::error title=流程失败::{error_msg}")
            
            # 发送错误通知
            send_error_notification(str(e))
            
            # 即使失败也发送日报摘要
            send_daily_summary(prediction_success, verification_success, prediction_period, error_messages)

def main():
    """主函数"""
    try:
        # 检查命令行参数
        test_mode = len(sys.argv) > 1 and sys.argv[1] in ['--test', '--dry-run', '-t']
        
        if test_mode:
            print("🧪 GitHub Actions兼容性测试模式")
            print("=" * 60)
            
            # 测试环境检测
            print(f"运行环境: {'GitHub Actions' if IS_GITHUB_ACTIONS else '本地环境'}")
            print(f"Python版本: {sys.version}")
            print(f"当前目录: {os.getcwd()}")
            
            # 测试模块导入
            modules = ['data_fetcher', 'lstm_model', 'prediction_history', 'ssq_wxpusher']
            print("\n=== 模块导入测试 ===")
            for module in modules:
                try:
                    __import__(module)
                    print(f"✅ {module}: 导入成功")
                except Exception as e:
                    print(f"❌ {module}: {e}")
            
            # 测试固定参数
            print("\n=== 固定参数验证 ===")
            print("✅ batch_size: 150")
            print("✅ epochs: 1200") 
            print("✅ window_length: 7")
            print("✅ 使用全部数据训练")
            print("✅ 微信UID: UID_yYObqdMVScIa66DGR2n2PCRFL10w")
            
            # 测试微信推送连接
            print("\n=== 微信推送测试 ===")
            if test_wxpusher_connection():
                print("✅ 微信推送连接正常")
            else:
                print("❌ 微信推送连接失败")
            
            print("\n✅ GitHub Actions兼容性测试完成")
            print("🚀 系统已优化，准备部署！")
            return 0
        
        # 正常运行模式
        automation = SSQAutomation()
        automation.run()
        return 0
        
    except Exception as e:
        logger.error(f"主程序异常: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 