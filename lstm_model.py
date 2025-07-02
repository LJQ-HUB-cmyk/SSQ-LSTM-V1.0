import numpy as np
import pandas as pd
import os
import pickle
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# TensorFlow imports with error handling
try:
    import tensorflow as tf
    from keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout
    from tensorflow.keras.optimizers import Adam
    
    # 设置TensorFlow日志级别（GitHub Actions环境优化）
    if os.getenv('GITHUB_ACTIONS') == 'true':
        tf.get_logger().setLevel('ERROR')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
except ImportError as e:
    print(f"TensorFlow导入失败: {e}")
    raise

# 可选的绘图库
try:
    import matplotlib
    matplotlib.use('Agg')  # 无GUI后端，适合服务器环境
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# 环境检测
IS_GITHUB_ACTIONS = os.getenv('GITHUB_ACTIONS') == 'true'

# 配置日志
logger = logging.getLogger(__name__)

class SSQLSTMModel:
    def __init__(self, 
                 batch_size=150, 
                 epochs=1200, 
                 window_length=7,
                 red_ball_count=6, 
                 blue_ball_count=1,
                 model_dir="models"):
        """
        初始化LSTM模型
        
        GitHub Actions优化版本 - 使用固定参数
        """
        # 固定参数 - GitHub Actions优化
        self.batch_size = batch_size
        self.epochs = epochs  
        self.window_length = window_length
        
        # 彩票配置
        self.red_ball_count = red_ball_count
        self.blue_ball_count = blue_ball_count
        self.red_ball_range = (1, 33)
        self.blue_ball_range = (1, 16)
        
        # 模型配置
        self.model_dir = model_dir
        self.model = None
        self.red_scaler = StandardScaler()
        self.blue_scaler = StandardScaler()
        
        # 确保模型目录存在
        self.ensure_model_dir()
        
        logger.info(f"LSTM模型初始化完成 - 批次大小: {self.batch_size}, 训练轮数: {self.epochs}, 窗口长度: {self.window_length}")
        
    def ensure_model_dir(self):
        """确保模型目录存在"""
        try:
            os.makedirs(self.model_dir, exist_ok=True)
            if IS_GITHUB_ACTIONS:
                # 确保目录权限正确
                os.chmod(self.model_dir, 0o755)
        except Exception as e:
            logger.error(f"创建模型目录失败: {e}")
            raise
    
    def save_model(self, model_name=None):
        """保存模型和缩放器"""
        if self.model is None:
            logger.warning("模型未训练，无法保存")
            return False
            
        try:
            if model_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name = f"ssq_lstm_{timestamp}"
            
            # 保存模型
            model_path = os.path.join(self.model_dir, f"{model_name}.keras")
            self.model.save(model_path)
            
            # 保存缩放器
            scaler_path = os.path.join(self.model_dir, f"{model_name}_scalers.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump({
                    'red_scaler': self.red_scaler,
                    'blue_scaler': self.blue_scaler
                }, f)
            
            logger.info(f"模型已保存：{model_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
            return False
    
    def load_model(self, model_name):
        """加载模型和缩放器"""
        try:
            # 加载模型
            model_path = os.path.join(self.model_dir, f"{model_name}.keras")
            if os.path.exists(model_path):
                self.model = load_model(model_path)
                logger.info(f"模型已加载：{model_path}")
            else:
                logger.warning(f"模型文件不存在：{model_path}")
                return False
            
            # 加载缩放器
            scaler_path = os.path.join(self.model_dir, f"{model_name}_scalers.pkl")
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    scalers = pickle.load(f)
                    self.red_scaler = scalers['red_scaler']
                    self.blue_scaler = scalers['blue_scaler']
                logger.info(f"缩放器已加载：{scaler_path}")
            else:
                logger.warning(f"缩放器文件不存在：{scaler_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False
    
    def build_model(self, input_shape):
        """构建LSTM模型"""
        try:
            model = Sequential([
                Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
                Dropout(0.2),
                Bidirectional(LSTM(32, return_sequences=False)),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(self.red_ball_count + self.blue_ball_count, activation='sigmoid')
            ])
            
            # 使用较低的学习率以获得更稳定的训练
            optimizer = Adam(learning_rate=0.001)
            model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
            
            logger.info(f"模型构建完成，输入形状: {input_shape}")
            return model
            
        except Exception as e:
            logger.error(f"构建模型失败: {e}")
            return None
    
    def train_model(self, df, training_periods=None):
        """训练LSTM模型 - GitHub Actions优化版本"""
        try:
            if IS_GITHUB_ACTIONS:
                print("::group::LSTM模型训练")
            
            logger.info("开始训练LSTM模型...")
            
            # 使用全部数据进行训练
            if training_periods is None:
                training_periods = len(df)
            
            # 数据准备
            X_train, y_train = self.prepare_data(df, training_periods)
            
            if X_train is None or len(X_train) == 0:
                logger.error("训练数据准备失败")
                return None
            
            # 构建模型
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.model = self.build_model(input_shape)
            
            if self.model is None:
                logger.error("模型构建失败")
                return None
            
            # 定义回调函数
            from tensorflow.keras.callbacks import Callback
            
            class ProgressCallback(Callback):
                def __init__(self, total_epochs=1200):
                    super().__init__()
                    self.total_epochs = total_epochs
                    self.last_reported = -1
                
                def on_epoch_end(self, epoch, logs=None):
                    progress = (epoch + 1) / self.total_epochs
                    loss = logs.get("loss", 0) if logs else 0
                    
                    if IS_GITHUB_ACTIONS:
                        # GitHub Actions环境 - 每10%或每100轮报告一次
                        progress_pct = int(progress * 100)
                        if progress_pct != self.last_reported and (progress_pct % 10 == 0 or (epoch + 1) % 100 == 0):
                            print(f"::notice::训练进度 {progress_pct}% ({epoch + 1}/{self.total_epochs}) Loss: {loss:.6f}")
                            self.last_reported = progress_pct
                    else:
                        # 本地控制台环境
                        if (epoch + 1) % 100 == 0 or epoch == 0:
                            print(f'训练进度: {epoch + 1}/{self.total_epochs} epochs, Loss: {loss:.6f}')
            
            # 创建回调
            callback = ProgressCallback(self.epochs)
            
            # 训练模型
            logger.info(f"开始训练，参数：批次大小={self.batch_size}, 训练轮数={self.epochs}")
            
            history = self.model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.2,
                callbacks=[callback],
                verbose=0  # 静默训练，使用自定义回调显示进度
            )
            
            if IS_GITHUB_ACTIONS:
                print("::endgroup::")
                print("::notice title=训练完成::LSTM模型训练成功完成")
            else:
                print('✅ 训练完成！')
            
            return history
            
        except Exception as e:
            logger.error(f"训练过程出错: {e}")
            if IS_GITHUB_ACTIONS:
                print("::endgroup::")
                print(f"::error title=训练失败::{e}")
            return None
    
    def plot_training_history(self, history):
        """绘制训练损失曲线 - GitHub Actions优化版本"""
        if history is None:
            return None
        
        if HAS_MATPLOTLIB:
            try:
                # 使用matplotlib（GitHub Actions或本地环境）
                plt.figure(figsize=(10, 6))
                plt.plot(history.history['loss'], label='训练损失', color='blue')
                
                if 'val_loss' in history.history:
                    plt.plot(history.history['val_loss'], label='验证损失', color='red')
                
                plt.title('训练损失曲线')
                plt.xlabel('训练轮数 (Epochs)')
                plt.ylabel('损失值 (Loss)')
                plt.legend()
                plt.grid(True)
                
                # 保存图片（非交互环境）
                if IS_GITHUB_ACTIONS:
                    plot_path = 'training_history.png'
                    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                    print(f"::notice::训练曲线已保存到 {plot_path}")
                
                plt.close()  # 释放内存
                return None
                
            except Exception as e:
                logger.error(f"绘制训练曲线失败: {e}")
        else:
            logger.warning("matplotlib不可用，跳过训练曲线绘制")
        
        return None
    
    def prepare_data(self, df, training_periods=None):
        """数据预处理 - 使用全部数据"""
        try:
            if training_periods is None:
                # 使用全部数据
                df_train = df.copy()
                logger.info(f"使用全部数据进行训练，共 {len(df_train)} 期")
            else:
                df_train = df.tail(training_periods).copy()
                logger.info(f"使用最近 {training_periods} 期数据进行训练")
            
            if len(df_train) < self.window_length + 1:
                logger.error(f"数据量不足，需要至少 {self.window_length + 1} 期数据")
                return None, None
            
            # 特征列
            red_cols = [f'red_{i}' for i in range(1, 7)]
            blue_col = 'blue'
            
            # 提取红球和蓝球数据
            red_data = df_train[red_cols].values
            blue_data = df_train[blue_col].values.reshape(-1, 1)
            
            # 数据标准化
            red_scaled = self.red_scaler.fit_transform(red_data)
            blue_scaled = self.blue_scaler.fit_transform(blue_data)
            
            # 合并特征
            all_data = np.hstack([red_scaled, blue_scaled])
            
            # 创建序列数据
            X, y = [], []
            for i in range(len(all_data) - self.window_length):
                X.append(all_data[i:(i + self.window_length)])
                y.append(all_data[i + self.window_length])
            
            X = np.array(X)
            y = np.array(y)
            
            logger.info(f"数据预处理完成，训练集形状: X={X.shape}, y={y.shape}")
            return X, y
            
        except Exception as e:
            logger.error(f"数据预处理失败: {e}")
            return None, None
    
    def predict_next_numbers(self, df):
        """预测下期号码"""
        try:
            if self.model is None:
                logger.error("模型未训练或加载，无法进行预测")
                return None
            
            # 使用最近window_length期的数据进行预测
            recent_data = df.tail(self.window_length)
            
            # 特征列
            red_cols = [f'red_{i}' for i in range(1, 7)]
            blue_col = 'blue'
            
            # 提取数据
            red_data = recent_data[red_cols].values
            blue_data = recent_data[blue_col].values.reshape(-1, 1)
            
            # 标准化
            red_scaled = self.red_scaler.transform(red_data)
            blue_scaled = self.blue_scaler.transform(blue_data)
            
            # 合并特征
            input_data = np.hstack([red_scaled, blue_scaled])
            input_data = input_data.reshape(1, self.window_length, -1)
            
            # 预测
            prediction = self.model.predict(input_data, verbose=0)
            
            # 反标准化
            red_pred = prediction[0][:6]
            blue_pred = prediction[0][6:7]
            
            # 转换为实际号码
            red_numbers = self._convert_to_red_numbers(red_pred)
            blue_number = self._convert_to_blue_number(blue_pred)
            
            predicted_numbers = red_numbers + [blue_number]
            logger.info(f"预测完成：{predicted_numbers}")
            
            return predicted_numbers
            
        except Exception as e:
            logger.error(f"预测失败: {e}")
            return None
    
    def _convert_to_red_numbers(self, predictions):
        """将预测结果转换为红球号码"""
        # 将预测值映射到1-33的范围
        scaled_predictions = (predictions * 32) + 1
        
        # 四舍五入并确保在有效范围内
        numbers = np.round(scaled_predictions).astype(int)
        numbers = np.clip(numbers, 1, 33)
        
        # 确保没有重复的号码
        unique_numbers = []
        for num in numbers:
            while num in unique_numbers:
                num = num + 1 if num < 33 else 1
            unique_numbers.append(num)
        
        return sorted(unique_numbers)
    
    def _convert_to_blue_number(self, prediction):
        """将预测结果转换为蓝球号码"""
        # 将预测值映射到1-16的范围
        scaled_prediction = (prediction[0] * 15) + 1
        
        # 四舍五入并确保在有效范围内
        number = int(round(scaled_prediction))
        number = max(1, min(16, number))
        
        return number
    
    def get_model_info(self):
        """获取模型信息"""
        return {
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'window_length': self.window_length,
            'red_ball_count': self.red_ball_count,
            'blue_ball_count': self.blue_ball_count,
            'model_trained': self.model is not None
        }