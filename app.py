import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from data_fetcher import fetch_ssq_data, load_local_data, get_latest_period_from_web
from lstm_model import SSQLSTMModel
from prediction_history import PredictionHistory
import os
import sys

# 页面配置
st.set_page_config(
    page_title="双色球LSTM预测系统",
    page_icon="🎱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化会话状态
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = PredictionHistory()

# 检测运行环境
if __name__ == "__main__":
    # 如果在GitHub Actions或其他自动化环境中，不启动Streamlit
    if os.getenv('GITHUB_ACTIONS') == 'true':
        print("检测到GitHub Actions环境，跳过Streamlit应用启动")
        print("请使用 ssq_automation.py 进行自动化预测")
        sys.exit(0)
    
    # 检查是否通过streamlit run启动
    if 'streamlit' not in sys.modules:
        print("🎯 双色球LSTM预测系统")
        print("=" * 40)
        print("请使用以下命令启动Web界面:")
        print("streamlit run app.py")
        print("")
        print("或者使用自动化脚本:")
        print("python ssq_automation.py")
        print("")
        print("测试系统状态:")
        print("python ssq_automation.py --test")
        sys.exit(1)

def main():
    st.title("🎱 双色球LSTM预测系统")
    st.markdown("---")
    
    # 侧边栏
    with st.sidebar:
        st.header("📊 系统控制")
        
        # 数据管理
        st.subheader("数据管理")
        if st.button("🔄 获取最新数据", use_container_width=True):
            with st.spinner("正在获取最新数据..."):
                df = fetch_ssq_data()
                if df is not None:
                    st.session_state.df = df
                    st.session_state.data_loaded = True
                    # 更新历史记录中的实际开奖结果
                    st.session_state.prediction_history.update_actual_results(df)
        
        if st.button("📁 加载本地数据", use_container_width=True):
            df = load_local_data()
            if df is not None:
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.success(f"已加载本地数据，共 {len(df)} 期")
            else:
                st.error("未找到本地数据文件")
        
        # 显示数据状态
        if st.session_state.data_loaded:
            st.success(f"✅ 数据已加载: {len(st.session_state.df)} 期")
            latest_period = st.session_state.df['Seq'].max()
            st.info(f"最新期数: {latest_period}")
        else:
            st.warning("⚠️ 请先加载数据")
    
    # 主界面选项卡
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 预测", "🔄 回测", "📈 历史记录", "📊 统计分析"])
    
    with tab1:
        prediction_tab()
    
    with tab2:
        backtest_tab()
    
    with tab3:
        history_tab()
    
    with tab4:
        statistics_tab()

def prediction_tab():
    st.header("🎯 号码预测")
    
    if not st.session_state.data_loaded:
        st.warning("请先在侧边栏加载数据")
        return
    
    df = st.session_state.df
    
    # 模型选择
    st.subheader("🤖 模型选择")
    
    # 创建临时模型实例来获取已保存的模型列表
    temp_model = SSQLSTMModel()
    saved_models = temp_model.get_saved_models()
    
    model_option = st.radio(
        "选择模型类型",
        ["重新训练模型", "使用已保存的模型"],
        horizontal=True
    )
    
    selected_model_name = None
    if model_option == "使用已保存的模型":
        if saved_models:
            selected_model_name = st.selectbox(
                "选择已保存的模型",
                saved_models,
                help="选择一个已训练好的模型进行预测"
            )
        else:
            st.warning("没有找到已保存的模型，请先训练并保存模型")
            model_option = "重新训练模型"
    
    # 参数设置（仅在重新训练时显示）
    if model_option == "重新训练模型":
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("模型参数")
            
            # 训练期数选择
            training_option = st.selectbox(
                "训练数据期数",
                ["全部数据", "最近100期", "最近300期", "最近500期", "自定义"]
            )
            
            if training_option == "自定义":
                training_periods = st.number_input(
                    "自定义期数", 
                    min_value=50, 
                    max_value=len(df), 
                    value=300
                )
            elif training_option == "最近100期":
                training_periods = 100
            elif training_option == "最近300期":
                training_periods = 300
            elif training_option == "最近500期":
                training_periods = 500
            else:
                training_periods = None
            
            window_length = st.number_input("时间窗口长度", min_value=3, max_value=20, value=7)
            
        with col2:
            st.subheader("训练参数")
            epochs = st.number_input("训练轮数", min_value=100, max_value=5000, value=1200)
            batch_size = st.number_input("批次大小", min_value=32, max_value=512, value=150)
            learning_rate = st.number_input("学习率", min_value=0.0001, max_value=0.01, value=0.0001, format="%.4f")
    else:
        # 使用已保存模型时的默认参数
        training_periods = None
        window_length = 7
        epochs = 1200
        batch_size = 150
        learning_rate = 0.0001
    
    # 预测按钮
    if st.button("🚀 开始预测", use_container_width=True, type="primary"):
        start_time = time.time()
        
        try:
            # 创建模型
            model = SSQLSTMModel(window_length=window_length)
            history = None
            
            if model_option == "使用已保存的模型":
                # 加载已保存的模型
                with st.spinner(f"正在加载模型: {selected_model_name}..."):
                    model.load_model(selected_model_name)
                st.success(f"模型加载成功: {selected_model_name}")
            else:
                # 重新训练模型
                # 准备数据
                with st.spinner("正在准备训练数据..."):
                    x_train, y_train, df_work = model.prepare_data(df, training_periods)
                
                st.success(f"训练数据准备完成，样本数: {len(x_train)}")
                
                # 训练模型
                st.subheader("🔄 模型训练")
                history = model.train_model(x_train, y_train, epochs, batch_size, learning_rate)
                
                # 显示训练损失曲线
                if history is not None:
                    st.subheader("📈 训练损失曲线")
                    fig = model.plot_training_history(history)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 自动保存模型
                    st.subheader("💾 自动保存模型")
                    # 获取最新期号作为模型名称
                    latest_period = df['Seq'].max()
                    model_name = f"{latest_period}.keras"
                    
                    try:
                        saved_name = model.save_model(model_name)
                        st.success(f"模型已自动保存: {saved_name}")
                    except Exception as e:
                        st.error(f"自动保存模型失败: {e}")
            
            # 预测下一期
            latest_period = df['Seq'].max()
            next_period = latest_period + 1
            
            with st.spinner("正在预测下一期号码..."):
                predicted_numbers = model.predict_next_period(df, next_period)
            
            # 显示预测结果（完全按照原程序格式）
            st.subheader("🎯 预测结果")
            
            # 获取测试期数据（最后一期）
            test_period = df['Seq'].max()
            test_actual = df[df['Seq'] == test_period].iloc[0, 1:].values  # 获取实际开奖号码
            
            # 模拟测试期预测（使用倒数第二期到倒数第八期的数据预测最后一期）
            test_data = df.iloc[-8:-1, 1:].values  # 获取用于预测测试期的数据
            test_data_scaled = model.scaler.transform(test_data)
            test_prediction = model.model.predict(np.array([test_data_scaled]))
            test_predicted = model.scaler.inverse_transform(test_prediction).astype(int)[0]
            
            # 按照原程序的三行输出格式
            output_lines = [
                f"{test_period} 期预测： {test_predicted} {test_predicted + 1}",
                f"{test_period} 期开奖： {test_actual}",
                f"{next_period} 期预测： {predicted_numbers} {predicted_numbers + 1}"
            ]
            
            # 使用st.code显示完整输出
            full_output = "\n".join(output_lines)
            st.code(full_output, language=None)
            
            # 详细显示（可选）
            with st.expander("详细预测信息", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("预测期数", f"{next_period}期")
                    
                    # 显示红球
                    red_balls = predicted_numbers[:6]
                    red_balls_str = " ".join([f"{num:02d}" for num in sorted(red_balls)])
                    st.markdown(f"**红球**: <span style='color: red; font-size: 24px; font-weight: bold;'>{red_balls_str}</span>", unsafe_allow_html=True)
                    
                    # 显示蓝球
                    blue_ball = predicted_numbers[6]
                    st.markdown(f"**蓝球**: <span style='color: blue; font-size: 24px; font-weight: bold;'>{blue_ball:02d}</span>", unsafe_allow_html=True)
                
                with col2:
                    # 训练参数信息
                    st.write("**训练参数:**")
                    st.write(f"- 训练期数: {training_periods if training_periods else '全部'}")
                    st.write(f"- 时间窗口: {window_length}")
                    st.write(f"- 训练轮数: {epochs}")
                    st.write(f"- 批次大小: {batch_size}")
                    st.write(f"- 学习率: {learning_rate}")
            
            # 保存预测结果
            params = {
                'training_periods': training_periods if training_periods else len(df),
                'window_length': window_length,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate
            }
            
            st.session_state.prediction_history.save_prediction(
                next_period, predicted_numbers, params
            )
            
            # 显示耗时（保持与原程序一致的格式）
            end_time = time.time()
            total_time = end_time - start_time
            hours, remainder = divmod(total_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            time_output = f"耗时: {int(hours)}:{int(minutes)}:{int(seconds)} "
            st.code(time_output, language=None)
            st.success("预测完成！")
            
        except Exception as e:
            st.error(f"预测过程中出现错误: {e}")

def backtest_tab():
    st.header("🔄 模型回测")
    
    if not st.session_state.data_loaded:
        st.warning("请先在侧边栏加载数据")
        return
    
    df = st.session_state.df
    
    # 回测参数设置
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("回测参数")
        
        # 回测数据期数
        backtest_option = st.selectbox(
            "回测数据期数",
            ["最近50期", "最近100期", "最近200期", "自定义"]
        )
        
        if backtest_option == "自定义":
            backtest_periods = st.number_input(
                "自定义回测期数", 
                min_value=20, 
                max_value=min(500, len(df)), 
                value=50
            )
        elif backtest_option == "最近50期":
            backtest_periods = 50
        elif backtest_option == "最近100期":
            backtest_periods = 100
        else:
            backtest_periods = 200
        
        window_length = st.number_input("时间窗口长度", min_value=3, max_value=20, value=7, key="backtest_window")
    
    with col2:
        st.subheader("训练参数")
        epochs = st.number_input("训练轮数", min_value=50, max_value=1000, value=100, key="backtest_epochs")
        batch_size = st.number_input("批次大小", min_value=32, max_value=512, value=150, key="backtest_batch")
        learning_rate = st.number_input("学习率", min_value=0.0001, max_value=0.01, value=0.0001, format="%.4f", key="backtest_lr")
    
    # 回测按钮
    if st.button("🔄 开始回测", use_container_width=True, type="primary"):
        start_time = time.time()
        
        try:
            # 创建模型
            model = SSQLSTMModel(window_length=window_length)
            
            st.subheader("🔄 回测进行中")
            st.info(f"使用留一验证法回测最近 {backtest_periods} 期数据")
            
            # 执行回测
            results = model.backtest_leave_one_out(
                df, backtest_periods, epochs, batch_size, learning_rate
            )
            
            if results:
                # 显示回测结果
                st.subheader("📊 回测结果")
                
                # 统计信息
                stats = model.get_backtest_statistics(results)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("回测期数", stats['total_tests'])
                
                with col2:
                    st.metric("平均红球命中", f"{stats['avg_red_hits']:.2f}")
                
                with col3:
                    st.metric("蓝球命中率", f"{stats['blue_hit_rate']:.2%}")
                
                with col4:
                    st.metric("最高命中数", stats['max_hits'])
                
                # 命中分布图
                st.subheader("📈 命中分布")
                
                hit_dist = stats['hit_distribution']
                fig = px.bar(
                    x=list(hit_dist.keys()),
                    y=list(hit_dist.values()),
                    labels={'x': '命中个数', 'y': '次数'},
                    title="回测命中分布"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 详细结果表格
                st.subheader("📋 详细回测结果")
                
                results_df = pd.DataFrame([
                    {
                        '期数': r['period'],
                        '预测红球': ' '.join([f"{num:02d}" for num in sorted(r['predicted_red'])]),
                        '预测蓝球': f"{r['predicted_blue']:02d}",
                        '实际红球': ' '.join([f"{num:02d}" for num in sorted(r['actual_red'])]),
                        '实际蓝球': f"{r['actual_blue']:02d}",
                        '红球命中': r['red_hits'],
                        '蓝球命中': '✓' if r['blue_hit'] else '✗',
                        '总命中': r['total_hits']
                    }
                    for r in results
                ])
                
                st.dataframe(results_df, use_container_width=True)
                
                # 显示耗时（保持与原程序一致的格式）
                end_time = time.time()
                total_time = end_time - start_time
                hours, remainder = divmod(total_time, 3600)
                minutes, seconds = divmod(remainder, 60)
                time_output = f"耗时: {int(hours)}:{int(minutes)}:{int(seconds)} "
                st.code(time_output, language=None)
                st.success("回测完成！")
            
        except Exception as e:
            st.error(f"回测过程中出现错误: {e}")

def history_tab():
    st.header("📈 预测历史记录")
    
    # 获取历史记录
    history_df = st.session_state.prediction_history.get_history()
    
    if history_df.empty:
        st.info("暂无预测历史记录")
        return
    
    # 显示历史记录表格
    st.subheader("📋 历史预测记录")
    
    # 格式化显示
    display_df = history_df.copy()
    
    # 格式化预测号码
    for i in range(1, 7):
        display_df[f'predicted_red_{i}'] = display_df[f'predicted_red_{i}'].apply(lambda x: f"{int(x):02d}" if pd.notna(x) else "")
    display_df['predicted_blue'] = display_df['predicted_blue'].apply(lambda x: f"{int(x):02d}" if pd.notna(x) else "")
    
    # 格式化实际号码
    for i in range(1, 7):
        display_df[f'actual_red_{i}'] = display_df[f'actual_red_{i}'].apply(lambda x: f"{int(x):02d}" if pd.notna(x) else "")
    display_df['actual_blue'] = display_df['actual_blue'].apply(lambda x: f"{int(x):02d}" if pd.notna(x) else "")
    
    # 重新组织列
    display_columns = [
        'prediction_time', 'target_period',
        'predicted_red_1', 'predicted_red_2', 'predicted_red_3', 'predicted_red_4', 'predicted_red_5', 'predicted_red_6', 'predicted_blue',
        'actual_red_1', 'actual_red_2', 'actual_red_3', 'actual_red_4', 'actual_red_5', 'actual_red_6', 'actual_blue',
        'red_hits', 'blue_hit', 'total_hits',
        'training_periods', 'window_length', 'epochs', 'batch_size', 'learning_rate'
    ]
    
    # 重命名列
    column_names = {
        'prediction_time': '预测时间',
        'target_period': '预测期数',
        'predicted_red_1': '预测红1', 'predicted_red_2': '预测红2', 'predicted_red_3': '预测红3',
        'predicted_red_4': '预测红4', 'predicted_red_5': '预测红5', 'predicted_red_6': '预测红6',
        'predicted_blue': '预测蓝',
        'actual_red_1': '实际红1', 'actual_red_2': '实际红2', 'actual_red_3': '实际红3',
        'actual_red_4': '实际红4', 'actual_red_5': '实际红5', 'actual_red_6': '实际红6',
        'actual_blue': '实际蓝',
        'red_hits': '红球命中', 'blue_hit': '蓝球命中', 'total_hits': '总命中',
        'training_periods': '训练期数', 'window_length': '窗口长度', 'epochs': '训练轮数',
        'batch_size': '批次大小', 'learning_rate': '学习率'
    }
    
    display_df = display_df[display_columns].rename(columns=column_names)
    
    st.dataframe(display_df, use_container_width=True)
    
    # 导出功能
    if st.button("📥 导出历史记录"):
        csv = display_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="下载CSV文件",
            data=csv,
            file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def statistics_tab():
    st.header("📊 统计分析")
    
    # 获取预测统计
    stats = st.session_state.prediction_history.get_statistics()
    
    if stats is None:
        st.info("暂无完整的预测记录用于统计分析")
        return
    
    # 显示总体统计
    st.subheader("📈 总体统计")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("总预测次数", stats['total_predictions'])
    
    with col2:
        st.metric("平均红球命中", f"{stats['avg_red_hits']:.2f}")
    
    with col3:
        st.metric("蓝球命中率", f"{stats['blue_hit_rate']:.2%}")
    
    with col4:
        st.metric("最高命中数", stats['max_hits'])
    
    # 命中分布图
    st.subheader("📊 命中分布分析")
    
    hit_dist = stats['hit_distribution']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 柱状图
        fig_bar = px.bar(
            x=list(hit_dist.index),
            y=list(hit_dist.values),
            labels={'x': '命中个数', 'y': '次数'},
            title="命中分布柱状图"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # 饼图
        fig_pie = px.pie(
            values=list(hit_dist.values),
            names=[f"{i}个" for i in hit_dist.index],
            title="命中分布饼图"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # 历史趋势
    history_df = st.session_state.prediction_history.get_history()
    completed_df = history_df.dropna(subset=['actual_red_1'])
    
    if not completed_df.empty:
        st.subheader("📈 历史趋势")
        
        # 按时间排序
        completed_df['prediction_time'] = pd.to_datetime(completed_df['prediction_time'])
        completed_df = completed_df.sort_values('prediction_time')
        
        # 绘制趋势图
        fig_trend = go.Figure()
        
        fig_trend.add_trace(go.Scatter(
            x=completed_df['prediction_time'],
            y=completed_df['total_hits'],
            mode='lines+markers',
            name='总命中数',
            line=dict(color='blue')
        ))
        
        fig_trend.add_trace(go.Scatter(
            x=completed_df['prediction_time'],
            y=completed_df['red_hits'],
            mode='lines+markers',
            name='红球命中数',
            line=dict(color='red')
        ))
        
        fig_trend.update_layout(
            title="预测命中趋势",
            xaxis_title="预测时间",
            yaxis_title="命中个数",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)

if __name__ == "__main__":
    main()