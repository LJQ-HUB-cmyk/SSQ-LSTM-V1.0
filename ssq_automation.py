#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双色球LSTM预测系统自动化脚本
================

用于GitHub Actions自动化运行，包含数据获取、预测、验证和微信推送功能
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import traceback

# 导入项目模块
from data_fetcher import fetch_ssq_data, get_latest_period_from_web
from lstm_model import SSQLSTMModel
from prediction_history import PredictionHistory
from ssq_wxpusher import (
    send_prediction_report, 
    send_verification_report, 
    send_error_notification,
    send_daily_summary,
    test_wxpusher_connection
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ssq_automation.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class SSQAutomation:
    def __init__(self):
        self.history = PredictionHistory()
        self.model = None
        self.df = None
        
    def load_or_fetch_data(self):
        """加载或获取最新数据"""
        try:
            logger.info("开始获取最新双色球数据...")
            
            # 尝试获取最新数据
            self.df = fetch_ssq_data()
            
            if self.df is None:
                logger.error("获取数据失败")
                return False
            
            logger.info(f"数据获取成功，共 {len(self.df)} 期数据")
            logger.info(f"最新期号: {self.df['Seq'].max()}")
            
            # 更新历史预测的实际结果
            self.history.update_actual_results(self.df)
            
            return True
            
        except Exception as e:
            logger.error(f"数据获取过程出错: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def get_next_prediction_period(self):
        """获取下一期预测期号"""
        if self.df is None:
            return None
        
        # 获取最新期号并加1
        latest_period = self.df['Seq'].max()
        return latest_period + 1
    
    def train_and_predict(self, target_period):
        """训练模型并进行预测"""
        try:
            logger.info("开始训练LSTM模型...")
            
            # 创建模型实例
            self.model = SSQLSTMModel(window_length=7)
            
            # 检查是否有可用的预训练模型
            saved_models = self.model.get_saved_models()
            use_existing_model = False
            
            if saved_models:
                # 选择最新的模型
                latest_model = saved_models[0]
                logger.info(f"发现已保存的模型: {latest_model}")
                
                try:
                    self.model.load_model(latest_model)
                    logger.info(f"成功加载模型: {latest_model}")
                    use_existing_model = True
                except Exception as e:
                    logger.warning(f"加载模型失败，将重新训练: {e}")
                    use_existing_model = False
            
            training_info = {}
            model_params = {
                'training_periods': 300,
                'window_length': 7,
                'epochs': 1200,
                'batch_size': 150,
                'learning_rate': 0.0001
            }
            
            if not use_existing_model:
                # 重新训练模型
                logger.info("开始训练新模型...")
                
                # 准备训练数据
                x_train, y_train, df_work = self.model.prepare_data(
                    self.df, 
                    training_periods=model_params['training_periods']
                )
                
                logger.info(f"训练数据准备完成，样本数: {len(x_train)}")
                
                # 训练模型
                history = self.model.train_model(
                    x_train, y_train,
                    epochs=model_params['epochs'],
                    batch_size=model_params['batch_size'],
                    learning_rate=model_params['learning_rate']
                )
                
                if history is not None:
                    training_info['final_loss'] = history.history['loss'][-1]
                    logger.info(f"模型训练完成，最终损失: {training_info['final_loss']:.4f}")
                    
                    # 保存训练好的模型
                    model_name = f"{target_period-1}.keras"
                    try:
                        saved_name = self.model.save_model(model_name)
                        logger.info(f"模型已保存: {saved_name}")
                    except Exception as e:
                        logger.warning(f"保存模型失败: {e}")
            
            # 进行预测
            logger.info(f"开始预测第{target_period}期号码...")
            
            current_period = target_period - 1
            predicted_numbers = self.model.predict_next_period(self.df, current_period)
            
            if predicted_numbers is None:
                logger.error("预测失败")
                return None, None, None
            
            logger.info(f"预测完成: {predicted_numbers}")
            
            # 保存预测结果到历史记录
            self.history.save_prediction(target_period, predicted_numbers, model_params)
            
            return predicted_numbers, model_params, training_info
            
        except Exception as e:
            logger.error(f"训练和预测过程出错: {e}")
            logger.error(traceback.format_exc())
            return None, None, None
    
    def generate_prediction_report(self, target_period, predicted_numbers, model_params, training_info):
        """生成预测报告文件"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = f"ssq_prediction_report_{timestamp}.txt"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(f"双色球第{target_period}期LSTM预测报告\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"预测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"预测期号: {target_period}\n\n")
                
                # 预测号码
                red_nums = ' '.join(f'{num:02d}' for num in predicted_numbers[:6])
                blue_num = f'{predicted_numbers[6]:02d}'
                f.write(f"预测号码:\n")
                f.write(f"红球: {red_nums}\n")
                f.write(f"蓝球: {blue_num}\n\n")
                
                # 模型参数
                if model_params:
                    f.write("模型参数:\n")
                    for key, value in model_params.items():
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")
                
                # 训练信息
                if training_info:
                    f.write("训练信息:\n")
                    for key, value in training_info.items():
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")
                
                # 历史验证统计
                stats = self.history.get_statistics()
                if stats:
                    f.write("历史预测统计:\n")
                    f.write(f"  总预测次数: {stats['total_predictions']}\n")
                    f.write(f"  平均红球命中: {stats['avg_red_hits']:.2f}个\n")
                    f.write(f"  平均总命中: {stats['avg_total_hits']:.2f}个\n")
                    f.write(f"  蓝球命中率: {stats['blue_hit_rate']*100:.1f}%\n")
                    f.write(f"  最高命中: {stats['max_hits']}个\n")
                
                f.write("\n" + "=" * 50 + "\n")
                f.write("本报告由双色球LSTM预测系统自动生成\n")
            
            logger.info(f"预测报告已生成: {report_file}")
            return report_file
            
        except Exception as e:
            logger.error(f"生成预测报告失败: {e}")
            return None
    
    def verify_latest_predictions(self):
        """验证最新的预测结果"""
        try:
            logger.info("开始验证历史预测结果...")
            
            # 获取预测历史
            history_df = self.history.get_history()
            
            if history_df.empty:
                logger.info("没有历史预测记录")
                return True
            
            # 查找最新的已验证记录
            completed_df = history_df.dropna(subset=['actual_red_1'])
            
            if not completed_df.empty:
                # 获取最新验证记录
                latest_verified = completed_df.sort_values('target_period', ascending=False).iloc[0]
                
                verification_data = {
                    'eval_period': latest_verified['target_period'],
                    'prize_red': [int(latest_verified[f'actual_red_{i}']) for i in range(1, 7)],
                    'prize_blue': int(latest_verified['actual_blue']),
                    'predicted_red': [int(latest_verified[f'predicted_red_{i}']) for i in range(1, 7)],
                    'predicted_blue': int(latest_verified['predicted_blue']),
                    'red_hits': int(latest_verified['red_hits']),
                    'blue_hit': int(latest_verified['blue_hit']),
                    'total_hits': int(latest_verified['total_hits'])
                }
                
                logger.info(f"发现最新验证记录: 第{verification_data['eval_period']}期")
                logger.info(f"命中情况: 红球{verification_data['red_hits']}个, 蓝球{'✓' if verification_data['blue_hit'] else '✗'}")
                
                # 发送验证报告
                send_verification_report(verification_data)
                
            return True
            
        except Exception as e:
            logger.error(f"验证预测结果失败: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def run(self):
        """运行自动化流程"""
        prediction_success = False
        verification_success = False
        prediction_period = None
        error_messages = []
        
        try:
            logger.info("开始运行双色球LSTM预测自动化流程...")
            
            # 测试微信推送
            if not test_wxpusher_connection():
                logger.warning("微信推送测试失败，但继续执行流程")
            
            # 1. 获取数据
            if not self.load_or_fetch_data():
                error_messages.append("数据获取失败")
                raise Exception("数据获取失败")
            
            # 2. 验证历史预测
            verification_success = self.verify_latest_predictions()
            if not verification_success:
                error_messages.append("历史验证失败")
            
            # 3. 获取下一期预测期号
            prediction_period = self.get_next_prediction_period()
            if prediction_period is None:
                error_messages.append("无法确定预测期号")
                raise Exception("无法确定预测期号")
            
            logger.info(f"将预测第{prediction_period}期")
            
            # 4. 训练模型并预测
            predicted_numbers, model_params, training_info = self.train_and_predict(prediction_period)
            
            if predicted_numbers is None:
                error_messages.append("模型预测失败")
                raise Exception("模型预测失败")
            
            prediction_success = True
            
            # 5. 生成预测报告
            report_file = self.generate_prediction_report(
                prediction_period, predicted_numbers, model_params, training_info
            )
            
            # 6. 发送预测报告推送
            send_prediction_report(prediction_period, predicted_numbers, model_params, training_info)
            
            logger.info("自动化流程执行成功")
            
        except Exception as e:
            logger.error(f"自动化流程执行失败: {e}")
            logger.error(traceback.format_exc())
            error_messages.append(str(e))
            
            # 发送错误通知
            send_error_notification(str(e))
        
        finally:
            # 发送日报摘要
            error_msg = "; ".join(error_messages) if error_messages else None
            send_daily_summary(prediction_success, verification_success, prediction_period, error_msg)
            
            logger.info(f"流程执行完成 - 预测: {'成功' if prediction_success else '失败'}, 验证: {'成功' if verification_success else '失败'}")

def main():
    """主函数"""
    try:
        automation = SSQAutomation()
        automation.run()
        return 0
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 