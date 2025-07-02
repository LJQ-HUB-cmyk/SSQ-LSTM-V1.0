import os
import sys
import requests
import pandas as pd
import logging

# 配置日志
logger = logging.getLogger(__name__)

# 检测运行环境
IS_GITHUB_ACTIONS = os.getenv('GITHUB_ACTIONS') == 'true'
IS_STREAMLIT = False

# 尝试导入streamlit，如果失败则创建一个虚拟st对象
try:
    import streamlit as st
    IS_STREAMLIT = True
except ImportError:
    # 创建虚拟的streamlit对象用于非Streamlit环境
    class VirtualStreamlit:
        def spinner(self, text):
            if IS_GITHUB_ACTIONS:
                print(f"::group::{text}")
            else:
                print(f"🔄 {text}")
            return self
        def __enter__(self):
            return self
        def __exit__(self, *args):
            if IS_GITHUB_ACTIONS:
                print("::endgroup::")
        def progress(self, value):
            if IS_GITHUB_ACTIONS and hasattr(self, '_last_progress'):
                progress_pct = int(value * 100)
                if progress_pct != self._last_progress:
                    print(f"Progress: {progress_pct}%")
                    self._last_progress = progress_pct
            return self
        def success(self, text):
            if IS_GITHUB_ACTIONS:
                print(f"::notice title=Success::{text}")
            else:
                print(f"✅ {text}")
        def error(self, text):
            if IS_GITHUB_ACTIONS:
                print(f"::error title=Error::{text}")
            else:
                print(f"❌ {text}")
        def warning(self, text):
            if IS_GITHUB_ACTIONS:
                print(f"::warning title=Warning::{text}")
            else:
                print(f"⚠️ {text}")
    
    st = VirtualStreamlit()
    st._last_progress = -1

def fetch_ssq_data():
    """获取双色球数据"""
    # 定义保存路径和文件名
    save_directory = "data"
    file_name = "ssq.csv"
    file_path = os.path.join(save_directory, file_name)
    
    # 创建保存目录（如果不存在）
    try:
        os.makedirs(save_directory, exist_ok=True)
        if IS_GITHUB_ACTIONS:
            # 确保目录权限正确
            os.chmod(save_directory, 0o755)
    except Exception as e:
        logger.error(f"创建数据目录失败: {e}")
        if IS_STREAMLIT:
            st.error(f"创建数据目录失败: {e}")
        return None
    
    # 定义数据源URL
    url = "https://data.17500.cn/ssq_asc.txt"
    
    try:
        # 发送HTTP GET请求获取数据
        with st.spinner('正在获取最新双色球数据...'):
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # 添加重试机制
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.get(url, headers=headers, timeout=30)
                    response.raise_for_status()
                    break
                except requests.RequestException as e:
                    if attempt == max_retries - 1:
                        raise e
                    logger.warning(f"请求失败，重试 {attempt + 1}/{max_retries}: {e}")
                    if IS_GITHUB_ACTIONS:
                        print(f"::warning::请求失败，重试 {attempt + 1}/{max_retries}: {e}")
                    
    except requests.RequestException as e:
        error_msg = f"请求数据时出错: {e}"
        logger.error(error_msg)
        st.error(error_msg)
        return None
    
    data = []
    # 假设数据按Seq升序排列
    lines = response.text.strip().split('\n')
    
    progress_bar = st.progress(0)
    total_lines = len(lines)
    
    for i, line in enumerate(lines):
        # 更新进度条
        if IS_STREAMLIT:
            progress_bar.progress((i + 1) / total_lines)
        elif IS_GITHUB_ACTIONS and i % 5000 == 0:  # GitHub Actions: 每5000行打印一次
            progress_pct = int((i + 1) / total_lines * 100)
            print(f"::notice::数据处理进度: {progress_pct}% ({i+1}/{total_lines})")
        elif not IS_STREAMLIT and i % 1000 == 0:  # 本地运行: 每1000行打印一次
            print(f"处理进度: {i+1}/{total_lines} ({(i+1)/total_lines*100:.1f}%)")
        
        if len(line) < 10:
            continue  # 跳过无效行
        
        # 仅分割第一个逗号，忽略后续数据
        parts = line.split(',', 1)
        if not parts:
            continue  # 跳过空行
        
        first_part = parts[0].strip()
        fields = first_part.split()
        
        # 确保有至少 8 个字段（Seq + 日期 + 6个红球 + 1个蓝球）
        if len(fields) < 8:
            continue
        
        seq = fields[0]
        red_balls = fields[2:8]  # 提取6个红球
        blue_ball = fields[8] if len(fields) > 8 else None  # 提取蓝球
        
        # 检查红球和蓝球数量是否正确
        if len(red_balls) != 6 or not blue_ball:
            continue
        
        # 构建数据字典
        item = {'Seq': seq}
        for i in range(1, 7):
            item[f'red_{i}'] = red_balls[i-1]
        item['blue'] = blue_ball
        
        data.append(item)
    
    # 创建DataFrame
    df = pd.DataFrame(data, columns=['Seq'] + [f'red_{i}' for i in range(1, 7)] + ['blue'])
    
    if df.empty:
        st.error("没有提取到任何数据。请检查数据格式或数据源是否可用。")
        return None
    else:
        # 将Seq转换为整数以便排序
        try:
            df['Seq'] = df['Seq'].astype(int)
        except ValueError as e:
            st.error(f"转换Seq为整数时出错: {e}")
            return None
        
        # 按Seq升序排序
        df.sort_values(by='Seq', inplace=True)
        
        try:
            # 保存为CSV文件
            df.to_csv(file_path, encoding="utf-8", index=False)
            
            # 验证文件是否保存成功
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                success_msg = f"数据已成功保存到 {file_path}，共获取 {len(df)} 期数据"
                logger.info(success_msg)
                st.success(success_msg)
                
                if IS_GITHUB_ACTIONS:
                    print(f"::notice title=数据获取成功::共获取 {len(df)} 期数据")
                
                return df
            else:
                raise Exception("文件保存失败或文件为空")
                
        except Exception as e:
            error_msg = f"保存数据时出错: {e}"
            logger.error(error_msg)
            st.error(error_msg)
            return None

def load_local_data():
    """加载本地数据"""
    file_path = os.path.join("data", "ssq.csv")
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            st.error(f"加载本地数据时出错: {e}")
            return None
    else:
        return None

def get_latest_period_from_web():
    """从网络获取最新期数"""
    url = "https://data.17500.cn/ssq_asc.txt"
    try:
        response = requests.get(url, headers={'User-agent': 'chrome'})
        response.raise_for_status()
        lines = response.text.strip().split('\n')
        
        # 获取最后一行有效数据
        for line in reversed(lines):
            if len(line) > 10:
                fields = line.split()
                if len(fields) >= 8:
                    return int(fields[0])
        return None
    except:
        return None