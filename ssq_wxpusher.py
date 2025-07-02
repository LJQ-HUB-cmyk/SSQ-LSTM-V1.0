# -*- coding: utf-8 -*-
"""
双色球LSTM预测系统微信推送模块
================

提供微信推送功能，用于推送双色球LSTM预测报告和验证报告
"""

import requests
import logging
import json
import os
from datetime import datetime
from typing import Optional, List, Dict

# 微信推送配置
# 支持从环境变量读取配置（用于GitHub Actions等CI环境）
APP_TOKEN = os.getenv("WXPUSHER_APP_TOKEN", "AT_FInZJJ0mUU8xvQjKRP7v6omvuHN3Fdqw")
USER_UIDS = os.getenv("WXPUSHER_USER_UIDS", "UID_yYObqdMVScIa66DGR2n2PCRFL10w").split(",") if os.getenv("WXPUSHER_USER_UIDS") else ["UID_yYObqdMVScIa66DGR2n2PCRFL10w"]
TOPIC_IDS = [int(x) for x in os.getenv("WXPUSHER_TOPIC_IDS", "").split(",") if x.strip().isdigit()]

def get_latest_verification_result() -> Optional[Dict]:
    """获取最新的验证结果
    
    Returns:
        最新验证结果字典，如果没有则返回None
    """
    try:
        from prediction_history import PredictionHistory
        
        history = PredictionHistory()
        history_df = history.get_history()
        
        if history_df.empty:
            return None
        
        # 获取最新的已验证记录
        completed_df = history_df.dropna(subset=['actual_red_1'])
        
        if completed_df.empty:
            return None
        
        # 按期号降序排列，获取最新验证记录
        latest_record = completed_df.sort_values('target_period', ascending=False).iloc[0]
        
        result = {
            'eval_period': latest_record['target_period'],
            'prize_red': [int(latest_record[f'actual_red_{i}']) for i in range(1, 7)],
            'prize_blue': int(latest_record['actual_blue']),
            'predicted_red': [int(latest_record[f'predicted_red_{i}']) for i in range(1, 7)],
            'predicted_blue': int(latest_record['predicted_blue']),
            'red_hits': int(latest_record['red_hits']),
            'blue_hit': int(latest_record['blue_hit']),
            'total_hits': int(latest_record['total_hits'])
        }
        
        return result
        
    except Exception as e:
        logging.error(f"获取最新验证结果失败: {e}")
        return None

def send_wxpusher_message(content: str, title: str = None, topicIds: List[int] = None, uids: List[str] = None) -> Dict:
    """发送微信推送消息
    
    Args:
        content: 消息内容
        title: 消息标题
        topicIds: 主题ID列表，默认使用全局配置
        uids: 用户ID列表，默认使用全局配置
    
    Returns:
        API响应结果字典
    """
    if not APP_TOKEN:
        return {"success": False, "error": "微信推送未配置APP_TOKEN"}
    
    url = "https://wxpusher.zjiecode.com/api/send/message"
    headers = {"Content-Type": "application/json"}
    
    data = {
        "appToken": APP_TOKEN,
        "content": content,
        "uids": uids or USER_UIDS,
        "topicIds": topicIds or TOPIC_IDS,
        "summary": title or "双色球LSTM预测更新",
        "contentType": 1,  # 1=文本，2=HTML
    }
    
    try:
        response = requests.post(url, json=data, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        if result.get("success", False):
            logging.info(f"微信推送成功: {title}")
            return {"success": True, "data": result}
        else:
            logging.error(f"微信推送失败: {result.get('msg', '未知错误')}")
            return {"success": False, "error": result.get('msg', '推送失败')}
            
    except requests.exceptions.RequestException as e:
        logging.error(f"微信推送网络错误: {e}")
        return {"success": False, "error": f"网络错误: {str(e)}"}
    except Exception as e:
        logging.error(f"微信推送异常: {e}")
        return {"success": False, "error": f"未知异常: {str(e)}"}

def send_prediction_report(target_period: int, predicted_numbers: List[int], 
                          model_params: Dict = None, training_info: Dict = None) -> Dict:
    """发送双色球LSTM预测报告
    
    Args:
        target_period: 预测期号
        predicted_numbers: 预测号码列表 [red1, red2, red3, red4, red5, red6, blue]
        model_params: 模型参数字典
        training_info: 训练信息字典
    
    Returns:
        推送结果字典
    """
    title = f"🎯 双色球第{target_period}期LSTM预测报告"
    
    try:
        # 获取最新验证结果
        latest_verification = get_latest_verification_result()
        
        # 构建预测号码显示
        red_nums = ' '.join(f'{num:02d}' for num in predicted_numbers[:6])
        blue_num = f'{predicted_numbers[6]:02d}'
        
        # 构建验证结果摘要
        verification_summary = ""
        if latest_verification:
            eval_period = latest_verification.get('eval_period', '未知')
            actual_red = ' '.join(f'{n:02d}' for n in latest_verification.get('prize_red', []))
            actual_blue = f"{latest_verification.get('prize_blue', 0):02d}"
            red_hits = latest_verification.get('red_hits', 0)
            blue_hit = latest_verification.get('blue_hit', 0)
            total_hits = latest_verification.get('total_hits', 0)
            
            verification_summary = f"""
📅 最新验证（第{eval_period}期）：
🎱 开奖: 红球 {actual_red} 蓝球 {actual_blue}
🎯 命中: 红球{red_hits}个 蓝球{'✓' if blue_hit else '✗'} 总计{total_hits}个
"""
        
        # 构建模型信息
        model_info = ""
        if model_params:
            training_periods = model_params.get('training_periods', '全部')
            window_length = model_params.get('window_length', '未知')
            epochs = model_params.get('epochs', '未知')
            
            model_info = f"""
🤖 模型参数：
• 训练期数: {training_periods}期
• 时间窗口: {window_length}
• 训练轮数: {epochs}
"""
        
        # 构建训练信息
        training_summary = ""
        if training_info:
            final_loss = training_info.get('final_loss', 0)
            if final_loss > 0:
                training_summary = f"• 训练损失: {final_loss:.4f}\n"
        
        # 构建推送内容
        content = f"""🎯 双色球第{target_period}期LSTM深度学习预测

🎱 AI预测号码：
红球: {red_nums}
蓝球: {blue_num}
{verification_summary}{model_info}
📈 算法特点：
• 基于LSTM循环神经网络
• 深度学习时序模式识别
• 多维特征自动提取
• 历史数据模式学习
{training_summary}
⏰ 预测时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}

💡 仅供参考，理性投注！祝您好运！"""
        
        return send_wxpusher_message(content, title)
        
    except Exception as e:
        logging.error(f"构建预测报告推送内容失败: {e}")
        return {"success": False, "error": f"内容构建失败: {str(e)}"}

def send_verification_report(verification_data: Dict) -> Dict:
    """发送双色球验证报告
    
    Args:
        verification_data: 验证报告数据字典
    
    Returns:
        推送结果字典
    """
    try:
        period = verification_data.get('eval_period', '未知')
        title = f"✅ 双色球第{period}期LSTM预测验证报告"
        
        actual_red = verification_data.get('prize_red', [])
        actual_blue = verification_data.get('prize_blue', 0)
        predicted_red = verification_data.get('predicted_red', [])
        predicted_blue = verification_data.get('predicted_blue', 0)
        red_hits = verification_data.get('red_hits', 0)
        blue_hit = verification_data.get('blue_hit', 0)
        total_hits = verification_data.get('total_hits', 0)
        
        # 格式化号码显示
        actual_red_str = ' '.join(f'{n:02d}' for n in actual_red)
        actual_blue_str = f'{actual_blue:02d}'
        predicted_red_str = ' '.join(f'{n:02d}' for n in predicted_red)
        predicted_blue_str = f'{predicted_blue:02d}'
        
        # 计算奖级
        prize_level = "未中奖"
        if red_hits == 6 and blue_hit == 1:
            prize_level = "一等奖"
        elif red_hits == 6 and blue_hit == 0:
            prize_level = "二等奖"
        elif red_hits == 5 and blue_hit == 1:
            prize_level = "三等奖"
        elif (red_hits == 5 and blue_hit == 0) or (red_hits == 4 and blue_hit == 1):
            prize_level = "四等奖"
        elif (red_hits == 4 and blue_hit == 0) or (red_hits == 3 and blue_hit == 1):
            prize_level = "五等奖"
        elif blue_hit == 1:
            prize_level = "六等奖"
        
        # 构建验证报告内容
        content = f"""✅ 双色球第{period}期LSTM预测验证

🎱 开奖号码：
红球：{actual_red_str}
蓝球：{actual_blue_str}

🤖 AI预测：
红球：{predicted_red_str}
蓝球：{predicted_blue_str}

🎯 命中结果：
红球命中：{red_hits}个
蓝球命中：{'✓' if blue_hit else '✗'}
总命中数：{total_hits}个
奖级：{prize_level}

📊 预测准确率：
红球准确率：{red_hits/6*100:.1f}%
蓝球准确率：{blue_hit*100:.0f}%
总体准确率：{total_hits/7*100:.1f}%

⏰ 验证时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}"""
        
        return send_wxpusher_message(content, title)
        
    except Exception as e:
        logging.error(f"构建验证报告推送内容失败: {e}")
        return {"success": False, "error": f"内容构建失败: {str(e)}"}

def send_error_notification(error_msg: str, script_name: str = "双色球LSTM系统") -> Dict:
    """发送错误通知
    
    Args:
        error_msg: 错误信息
        script_name: 脚本名称
    
    Returns:
        推送结果字典
    """
    title = f"⚠️ {script_name}运行异常"
    
    content = f"""⚠️ 系统运行异常通知

📍 异常位置：{script_name}
🕒 发生时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
❌ 错误信息：
{error_msg}

请及时检查系统状态！"""
    
    return send_wxpusher_message(content, title)

def send_daily_summary(prediction_success: bool, verification_success: bool, 
                      prediction_period: int = None, error_msg: str = None) -> Dict:
    """发送每日运行摘要
    
    Args:
        prediction_success: 预测是否成功
        verification_success: 验证是否成功
        prediction_period: 预测期号
        error_msg: 错误信息（如有）
    
    Returns:
        推送结果字典
    """
    title = "📊 双色球LSTM系统日报"
    
    # 状态图标
    prediction_status = "✅" if prediction_success else "❌"
    verification_status = "✅" if verification_success else "❌"
    
    content = f"""📊 双色球LSTM预测系统日报

🕒 运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}

📈 任务执行状态：
{prediction_status} 数据获取与LSTM预测"""
    
    if prediction_period:
        content += f" (第{prediction_period}期)"
    
    content += f"\n{verification_status} 历史预测验证"
    
    if error_msg:
        content += f"\n\n⚠️ 异常信息：\n{error_msg}"
    
    content += "\n\n🔔 系统已自动完成定时任务"
    
    return send_wxpusher_message(content, title)

def test_wxpusher_connection() -> bool:
    """测试微信推送连接
    
    Returns:
        连接是否成功
    """
    if not APP_TOKEN:
        print("❌ 微信推送未配置，请设置环境变量WXPUSHER_APP_TOKEN")
        return False
    
    test_content = f"🔧 双色球LSTM推送系统测试\n\n测试时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n如收到此消息，说明推送功能正常！"
    result = send_wxpusher_message(test_content, "🔧 LSTM推送测试")
    return result.get("success", False)

if __name__ == "__main__":
    # 测试推送功能
    print("正在测试双色球LSTM微信推送功能...")
    
    # 测试基本推送
    if test_wxpusher_connection():
        print("✅ 微信推送测试成功！")
        
        # 测试预测报告推送
        test_prediction = [1, 6, 17, 22, 27, 32, 15]
        test_params = {
            'training_periods': 300,
            'window_length': 7,
            'epochs': 1200
        }
        
        print("测试预测报告推送...")
        send_prediction_report(2025071, test_prediction, test_params)
        
        print("测试验证报告推送...")
        test_verification = {
            'eval_period': 2025070,
            'prize_red': [2, 3, 15, 21, 22, 33],
            'prize_blue': 6,
            'predicted_red': [1, 6, 17, 22, 27, 32],
            'predicted_blue': 15,
            'red_hits': 2,
            'blue_hit': 0,
            'total_hits': 2
        }
        send_verification_report(test_verification)
        
    else:
        print("❌ 微信推送测试失败！请检查配置。") 