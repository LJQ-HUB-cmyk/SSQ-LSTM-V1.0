import os
import requests
import pandas as pd

# 尝试导入streamlit，如果失败则创建一个虚拟st对象
try:
    import streamlit as st
    use_streamlit = True
except ImportError:
    # 创建虚拟的streamlit对象用于非Streamlit环境
    class VirtualStreamlit:
        def spinner(self, text):
            print(text)
            return self
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def progress(self, value):
            return self
        def success(self, text):
            print(f"✅ {text}")
        def error(self, text):
            print(f"❌ {text}")
        def warning(self, text):
            print(f"⚠️ {text}")
    
    st = VirtualStreamlit()
    use_streamlit = False

def fetch_ssq_data():
    """获取双色球数据"""
    # 定义保存路径和文件名
    save_directory = "data"
    file_name = "ssq.csv"
    file_path = os.path.join(save_directory, file_name)
    
    # 创建保存目录（如果不存在）
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    # 定义数据源URL
    url = "https://data.17500.cn/ssq_asc.txt"
    
    try:
        # 发送HTTP GET请求获取数据
        with st.spinner('正在获取最新双色球数据...'):
            response = requests.get(url, headers={'User-agent': 'chrome'})
            response.raise_for_status()  # 检查请求是否成功
    except requests.RequestException as e:
        st.error(f"请求数据时出错: {e}")
        return None
    
    data = []
    # 假设数据按Seq升序排列
    lines = response.text.strip().split('\n')
    
    progress_bar = st.progress(0)
    total_lines = len(lines)
    
    for i, line in enumerate(lines):
        # 更新进度条
        if use_streamlit:
            progress_bar.progress((i + 1) / total_lines)
        elif i % 1000 == 0:  # 每1000行打印一次进度
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
            st.success(f"数据已成功保存到 {file_path}，共获取 {len(df)} 期数据")
            return df
        except Exception as e:
            st.error(f"保存数据时出错: {e}")
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