import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam
import streamlit as st
from sklearn.model_selection import TimeSeriesSplit
import os
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime

class SSQLSTMModel:
    def __init__(self, window_length=7):
        self.window_length = window_length
        self.scaler = StandardScaler()
        self.model = None
        self.number_of_features = None
        self.model_dir = "models"
        self.ensure_model_dir()
    
    def ensure_model_dir(self):
        """确保模型目录存在"""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
    
    def save_model(self, model_name=None):
        """保存训练好的模型和标准化器"""
        if self.model is None:
            raise ValueError("没有训练好的模型可以保存")
        
        if self.number_of_features is None:
            raise ValueError("模型特征数量未设置，请先训练模型")
        
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"ssq_lstm_model_{timestamp}"
        
        try:
            # 处理模型名称，如果已经包含.keras后缀则直接使用，否则添加.keras后缀
            if model_name.endswith('.keras'):
                base_name = model_name[:-6]  # 移除.keras后缀
            else:
                base_name = model_name
                model_name = f"{model_name}.keras"
            
            # 保存模型
            model_path = os.path.join(self.model_dir, model_name)
            self.model.save(model_path)
            
            # 保存标准化器和其他参数
            scaler_path = os.path.join(self.model_dir, f"{base_name}_scaler.pkl")
            params_path = os.path.join(self.model_dir, f"{base_name}_params.pkl")
            
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            params = {
                'window_length': self.window_length,
                'number_of_features': self.number_of_features
            }
            
            with open(params_path, 'wb') as f:
                pickle.dump(params, f)
            
            return base_name
        except Exception as e:
            raise Exception(f"保存模型时发生错误: {str(e)}")
    
    def load_model(self, model_name):
        """加载已保存的模型"""
        # 首先尝试.keras格式，然后尝试.h5格式（向后兼容）
        keras_path = os.path.join(self.model_dir, f"{model_name}.keras")
        h5_path = os.path.join(self.model_dir, f"{model_name}.h5")
        
        if os.path.exists(keras_path):
            model_path = keras_path
        elif os.path.exists(h5_path):
            model_path = h5_path
        else:
            raise FileNotFoundError(f"找不到模型文件: {model_name}")
        
        scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.pkl")
        params_path = os.path.join(self.model_dir, f"{model_name}_params.pkl")
        
        if not all(os.path.exists(path) for path in [scaler_path, params_path]):
            raise FileNotFoundError(f"模型文件不完整: {model_name}")
        
        # 加载模型
        self.model = load_model(model_path)
        
        # 加载标准化器
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # 加载参数
        with open(params_path, 'rb') as f:
            params = pickle.load(f)
            self.window_length = params['window_length']
            self.number_of_features = params['number_of_features']
        
        return True
    
    def get_saved_models(self):
        """获取所有已保存的模型列表"""
        if not os.path.exists(self.model_dir):
            return []
        
        models = []
        for file in os.listdir(self.model_dir):
            model_name = None
            if file.endswith('.keras'):
                model_name = file[:-6]  # 移除.keras后缀
            elif file.endswith('.h5'):
                model_name = file[:-3]  # 移除.h5后缀
            
            if model_name:
                # 检查是否有对应的scaler和params文件
                scaler_file = f"{model_name}_scaler.pkl"
                params_file = f"{model_name}_params.pkl"
                
                if (os.path.exists(os.path.join(self.model_dir, scaler_file)) and 
                    os.path.exists(os.path.join(self.model_dir, params_file))):
                    models.append(model_name)
        
        return sorted(list(set(models)), reverse=True)  # 去重并按时间倒序排列
    
    def plot_training_history(self, history):
        """绘制训练损失曲线"""
        if history is None:
            return None
        
        # 创建plotly图表
        fig = go.Figure()
        
        # 添加训练损失曲线
        fig.add_trace(go.Scatter(
            x=list(range(1, len(history.history['loss']) + 1)),
            y=history.history['loss'],
            mode='lines',
            name='训练损失',
            line=dict(color='blue')
        ))
        
        # 如果有验证损失，也添加到图表中
        if 'val_loss' in history.history:
            fig.add_trace(go.Scatter(
                x=list(range(1, len(history.history['val_loss']) + 1)),
                y=history.history['val_loss'],
                mode='lines',
                name='验证损失',
                line=dict(color='red')
            ))
        
        fig.update_layout(
            title='训练损失曲线',
            xaxis_title='训练轮数 (Epochs)',
            yaxis_title='损失值 (Loss)',
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    
    def prepare_data(self, df, training_periods=None):
        """准备训练数据"""
        # 复制数据并移除Seq列
        df_work = df.copy()
        if training_periods and training_periods > 0:
            df_work = df_work.tail(training_periods)
        
        df_work.drop(['Seq'], axis=1, inplace=True)
        self.number_of_features = df_work.shape[1]
        
        train_rows = df_work.values.shape[0]
        
        # 创建训练样本和标签
        train_samples = np.empty([train_rows - self.window_length, self.window_length, self.number_of_features], dtype=float)
        train_labels = np.empty([train_rows - self.window_length, self.number_of_features], dtype=float)
        
        for i in range(0, train_rows - self.window_length):
            train_samples[i] = df_work.iloc[i: i + self.window_length, 0: self.number_of_features]
            train_labels[i] = df_work.iloc[i + self.window_length: i + self.window_length + 1, 0: self.number_of_features]
        
        # 标准化数据
        transformed_dataset = self.scaler.fit_transform(df_work.values)
        scaled_train_samples = pd.DataFrame(data=transformed_dataset, index=df_work.index)
        
        x_train = np.empty([train_rows - self.window_length, self.window_length, self.number_of_features], dtype=float)
        y_train = np.empty([train_rows - self.window_length, self.number_of_features], dtype=float)
        
        for i in range(0, train_rows - self.window_length):
            x_train[i] = scaled_train_samples.iloc[i: i + self.window_length, 0: self.number_of_features]
            y_train[i] = scaled_train_samples.iloc[i + self.window_length: i + self.window_length + 1, 0: self.number_of_features]
        
        return x_train, y_train, df_work
    
    def build_model(self):
        """构建LSTM模型（保持原有架构）"""
        model = Sequential()
        
        # 添加输入层和第一个LSTM层
        model.add(Bidirectional(LSTM(240, input_shape=(self.window_length, self.number_of_features), return_sequences=True)))
        model.add(Dropout(0.2))
        
        # 添加第二个LSTM层
        model.add(Bidirectional(LSTM(240, input_shape=(self.window_length, self.number_of_features), return_sequences=True)))
        model.add(Dropout(0.2))
        
        # 添加第三个LSTM层
        model.add(Bidirectional(LSTM(240, input_shape=(self.window_length, self.number_of_features), return_sequences=True)))
        
        # 添加第四个LSTM层
        model.add(Bidirectional(LSTM(240, input_shape=(self.window_length, self.number_of_features), return_sequences=False)))
        model.add(Dropout(0.2))
        
        # 添加输出层
        model.add(Dense(33))
        model.add(Dense(self.number_of_features))
        
        return model
    
    def train_model(self, x_train, y_train, epochs=1200, batch_size=150, learning_rate=0.0001):
        """训练模型"""
        self.model = self.build_model()
        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['accuracy'])
        
        # 检查是否在Streamlit环境中
        try:
            import streamlit as st
            # 创建进度条
            progress_bar = st.progress(0)
            status_text = st.empty()
            use_streamlit = True
        except:
            use_streamlit = False
        
        from tensorflow.keras.callbacks import Callback
        
        class ProgressCallback(Callback):
            def __init__(self, progress_bar=None, status_text=None, total_epochs=1200, use_streamlit=False):
                super().__init__()
                self.progress_bar = progress_bar
                self.status_text = status_text
                self.total_epochs = total_epochs
                self.use_streamlit = use_streamlit
            
            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / self.total_epochs
                if self.use_streamlit and self.progress_bar and self.status_text:
                    self.progress_bar.progress(progress)
                    self.status_text.text(f'训练进度: {epoch + 1}/{self.total_epochs} epochs, Loss: {logs.get("loss", 0):.6f}')
                else:
                    # 控制台输出
                    if (epoch + 1) % 100 == 0 or epoch == 0:
                        print(f'训练进度: {epoch + 1}/{self.total_epochs} epochs, Loss: {logs.get("loss", 0):.6f}')
        
        # 自定义回调
        if use_streamlit:
            callback = ProgressCallback(progress_bar, status_text, epochs, True)
        else:
            callback = ProgressCallback(total_epochs=epochs, use_streamlit=False)
        
        # 训练模型
        history = self.model.fit(
            x=x_train, 
            y=y_train, 
            batch_size=batch_size, 
            epochs=epochs, 
            verbose=0,
            callbacks=[callback]
        )
        
        if use_streamlit:
            progress_bar.progress(1.0)
            status_text.text('训练完成！')
        else:
            print('训练完成！')
        
        return history
    
    def predict_next_period(self, df, current_period):
        """预测下一期号码"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        # 获取最近的window_length期数据
        df_work = df.copy()
        df_work.drop(['Seq'], axis=1, inplace=True)
        
        next_data = df_work.tail(self.window_length)
        next_data = np.array(next_data)
        x_next = self.scaler.transform(next_data)
        y_next_pred = self.model.predict(np.array([x_next]))
        
        # 反标准化预测结果
        predicted_numbers = self.scaler.inverse_transform(y_next_pred).astype(int)[0]
        
        return predicted_numbers
    
    def backtest_leave_one_out(self, df, training_periods=None, epochs=100, batch_size=150, learning_rate=0.0001):
        """留一验证法回测"""
        if training_periods:
            df_test = df.tail(training_periods + 1).copy()
        else:
            df_test = df.copy()
        
        results = []
        total_tests = len(df_test) - self.window_length - 1
        
        if total_tests <= 0:
            st.error("数据量不足以进行回测")
            return []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(total_tests):
            # 更新进度
            progress = (i + 1) / total_tests
            progress_bar.progress(progress)
            status_text.text(f'回测进度: {i + 1}/{total_tests}')
            
            # 准备训练数据（排除测试样本）
            train_data = df_test.iloc[:-(total_tests - i)].copy()
            test_data = df_test.iloc[-(total_tests - i)].copy()
            
            if len(train_data) < self.window_length + 1:
                continue
            
            # 训练模型
            x_train, y_train, _ = self.prepare_data(train_data)
            
            # 使用较少的epochs进行快速训练
            self.model = self.build_model()
            self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['accuracy'])
            self.model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=0)
            
            # 预测
            predicted_numbers = self.predict_next_period(train_data, test_data['Seq'])
            
            # 计算命中数
            actual_red = [test_data[f'red_{j}'] for j in range(1, 7)]
            actual_blue = test_data['blue']
            
            predicted_red = predicted_numbers[:6]
            predicted_blue = predicted_numbers[6]
            
            red_hits = len(set(predicted_red) & set(actual_red))
            blue_hit = 1 if predicted_blue == actual_blue else 0
            total_hits = red_hits + blue_hit
            
            results.append({
                'period': test_data['Seq'],
                'predicted_red': predicted_red,
                'predicted_blue': predicted_blue,
                'actual_red': actual_red,
                'actual_blue': actual_blue,
                'red_hits': red_hits,
                'blue_hit': blue_hit,
                'total_hits': total_hits
            })
        
        progress_bar.progress(1.0)
        status_text.text('回测完成！')
        
        return results
    
    def get_backtest_statistics(self, results):
        """计算回测统计信息"""
        if not results:
            return None
        
        total_tests = len(results)
        total_red_hits = sum(r['red_hits'] for r in results)
        total_blue_hits = sum(r['blue_hit'] for r in results)
        total_hits = sum(r['total_hits'] for r in results)
        
        hit_distribution = {}
        for r in results:
            hits = r['total_hits']
            hit_distribution[hits] = hit_distribution.get(hits, 0) + 1
        
        stats = {
            'total_tests': total_tests,
            'avg_red_hits': total_red_hits / total_tests,
            'avg_total_hits': total_hits / total_tests,
            'blue_hit_rate': total_blue_hits / total_tests,
            'max_hits': max(r['total_hits'] for r in results),
            'hit_distribution': hit_distribution
        }
        
        return stats