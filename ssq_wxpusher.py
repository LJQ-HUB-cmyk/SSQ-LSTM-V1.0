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

# 环境检测
IS_GITHUB_ACTIONS = os.getenv('GITHUB_ACTIONS') == 'true'

# 配置日志
logger = logging.getLogger(__name__)

# 微信推送配置 - GitHub Actions优化版本
APP_TOKEN = os.getenv("WXPUSHER_APP_TOKEN", "AT_FInZJJ0mUU8xvQjKRP7v6omvuHN3Fdqw")

# 固定用户UID - 根据用户要求
FIXED_USER_UID = "UID_yYObqdMVScIa66DGR2n2PCRFL10w"
USER_UIDS = [FIXED_USER_UID]

# 主题ID（可选）
TOPIC_IDS = []
if os.getenv("WXPUSHER_TOPIC_IDS"):
    TOPIC_IDS = [int(x) for x in os.getenv("WXPUSHER_TOPIC_IDS").split(",") if x.strip().isdigit()]

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

def send_message(content, summary=None, content_type=1, uids=None, topic_ids=None):
    """
    发送微信推送消息 - GitHub Actions优化版本
    """
    if not APP_TOKEN or APP_TOKEN.strip() == "":
        error_msg = "微信推送Token未配置"
        logger.error(error_msg)
        if IS_GITHUB_ACTIONS:
            print(f"::error title=配置错误::{error_msg}")
        return False
    
    # 检查Token格式是否有效
    if not APP_TOKEN.startswith("AT_") or len(APP_TOKEN) < 10:
        error_msg = f"微信推送Token格式无效: {APP_TOKEN[:10]}..."
        logger.error(error_msg)
        if IS_GITHUB_ACTIONS:
            print(f"::error title=Token格式错误::{error_msg}")
        return False
    
    # 构建消息数据
    data = {
        "appToken": APP_TOKEN,
        "content": content,
        "summary": summary or content[:50],
        "contentType": content_type
    }
    
    # 使用固定的用户UID
    data["uids"] = USER_UIDS
    
    # 添加主题（如果有）
    if topic_ids:
        data["topicIds"] = topic_ids
    elif TOPIC_IDS:
        data["topicIds"] = TOPIC_IDS
    
    try:
        if IS_GITHUB_ACTIONS:
            print(f"::group::发送微信推送")
            print(f"摘要: {data['summary']}")
        
        response = requests.post(
            "http://wxpusher.zjiecode.com/api/send/message",
            json=data,
            timeout=30
        )
        
        result = response.json()
        
        if result.get("success"):
            logger.info("微信推送发送成功")
            if IS_GITHUB_ACTIONS:
                print("::notice::微信推送发送成功")
                print("::endgroup::")
            return True
        else:
            error_msg = f"微信推送发送失败: {result.get('msg', '未知错误')}"
            logger.error(error_msg)
            if IS_GITHUB_ACTIONS:
                print(f"::error::{error_msg}")
                print("::endgroup::")
            return False
            
    except Exception as e:
        error_msg = f"发送微信推送时出错: {e}"
        logger.error(error_msg)
        if IS_GITHUB_ACTIONS:
            print(f"::error::{error_msg}")
            print("::endgroup::")
        return False

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
        
        return send_message(content, title)
        
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
        
        return send_message(content, title)
        
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
    
    return send_message(content, title)

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
    
    return send_message(content, title)

def test_wxpusher_connection():
    """测试微信推送连接"""
    if not APP_TOKEN or APP_TOKEN.strip() == "":
        logger.warning("微信推送Token未配置，跳过测试")
        if IS_GITHUB_ACTIONS:
            print("::warning::微信推送Token未配置，跳过测试")
        return False
    
    # 检查Token格式
    if not APP_TOKEN.startswith("AT_") or len(APP_TOKEN) < 10:
        logger.warning(f"微信推送Token格式可能无效: {APP_TOKEN[:10]}...")
        if IS_GITHUB_ACTIONS:
            print(f"::warning::微信推送Token格式可能无效")
        return False
    
    try:
        if IS_GITHUB_ACTIONS:
            print("::group::测试微信推送连接")
        
        test_content = f"""🔧 双色球LSTM预测系统连接测试

✅ 系统状态：正常运行
🕐 测试时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
👤 接收用户：{FIXED_USER_UID}

如收到此消息，说明微信推送功能正常！"""
        
        result = send_message(test_content, "🔧 系统连接测试")
        
        if result:
            logger.info("微信推送连接测试成功")
            if IS_GITHUB_ACTIONS:
                print("::notice::微信推送连接测试成功")
        else:
            logger.warning("微信推送连接测试失败")
            if IS_GITHUB_ACTIONS:
                print("::warning::微信推送连接测试失败")
        
        if IS_GITHUB_ACTIONS:
            print("::endgroup::")
        
        return result
        
    except Exception as e:
        logger.error(f"微信推送连接测试异常: {e}")
        if IS_GITHUB_ACTIONS:
            print(f"::error::微信推送连接测试异常: {e}")
            print("::endgroup::")
        return False

if __name__ == "__main__":
    # 测试推送功能
    print("🧪 测试微信推送功能...")
    success = test_wxpusher_connection()
    if success:
        print("✅ 微信推送测试成功")
    else:
        print("❌ 微信推送测试失败") 