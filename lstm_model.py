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
        
        # 添加兼容性属性 - 原程序需要的scaler属性
        self.scaler = StandardScaler()
        
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
                    'blue_scaler': self.blue_scaler,
                    'scaler': self.scaler  # 兼容性scaler
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
                    if 'scaler' in scalers:
                        self.scaler = scalers['scaler']
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
    
    def prepare_data(self, df, training_periods=None):
        """数据预处理 - 兼容原程序和新版本"""
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
                return None, None, None
            
            # 特征列
            red_cols = [f'red_{i}' for i in range(1, 7)]
            blue_col = 'blue'
            
            # 提取红球和蓝球数据
            red_data = df_train[red_cols].values
            blue_data = df_train[blue_col].values.reshape(-1, 1)
            
            # 数据标准化
            red_scaled = self.red_scaler.fit_transform(red_data)
            blue_scaled = self.blue_scaler.fit_transform(blue_data)
            
            # 合并特征，同时设置兼容性scaler
            all_data = np.hstack([red_scaled, blue_scaled])
            self.scaler = StandardScaler()
            all_data_normalized = self.scaler.fit_transform(all_data)
            
            # 创建序列数据
            X, y = [], []
            for i in range(len(all_data_normalized) - self.window_length):
                X.append(all_data_normalized[i:(i + self.window_length)])
                y.append(all_data_normalized[i + self.window_length])
            
            X = np.array(X)
            y = np.array(y)
            
            logger.info(f"数据预处理完成，训练集形状: X={X.shape}, y={y.shape}")
            return X, y, df_train
            
        except Exception as e:
            logger.error(f"数据预处理失败: {e}")
            return None, None, None
    
    def train_model(self, x_train=None, y_train=None, epochs=None, batch_size=None, learning_rate=None):
        """训练模型 - 兼容原程序接口
        
        支持两种调用方式：
        1. train_model(df) - 新的GitHub Actions方式  
        2. train_model(x_train, y_train, epochs, batch_size, learning_rate) - 原程序方式
        """
        # 如果第一个参数是DataFrame，使用新的方式
        if hasattr(x_train, 'columns'):  # 检查是否是DataFrame
            df = x_train
            return self._train_model_github_actions(df)
        
        # 否则使用原程序的方式
        return self._train_model_original(x_train, y_train, epochs, batch_size, learning_rate)
    
    def _train_model_github_actions(self, df, training_periods=None):
        """GitHub Actions优化的训练方式"""
        try:
            if IS_GITHUB_ACTIONS:
                print("::group::LSTM模型训练")
            
            logger.info("开始训练LSTM模型...")
            
            # 使用全部数据进行训练
            if training_periods is None:
                training_periods = len(df)
            
            # 数据准备
            X_train, y_train, df_work = self.prepare_data(df, training_periods)
            
            if X_train is None or len(X_train) == 0:
                logger.error("训练数据准备失败")
                return None
            
            # 保存训练数据信息用于后续获取模型信息
            self._last_training_df = df_work
            
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
                    self.total_epochs = total_epochs
                    self.last_reported = -1
                    
                def on_epoch_end(self, epoch, logs=None):
                    if IS_GITHUB_ACTIONS and epoch % 100 == 0:
                        progress = int((epoch + 1) / self.total_epochs * 100)
                        if progress != self.last_reported:
                            print(f"::notice::训练进度: {progress}% (Epoch {epoch + 1}/{self.total_epochs})")
                            self.last_reported = progress
            
            callbacks = [ProgressCallback(self.epochs)]
            
            # 训练模型
            history = self.model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=0 if IS_GITHUB_ACTIONS else 1
            )
            
            if IS_GITHUB_ACTIONS:
                print("::endgroup::")
            
            logger.info("模型训练完成")
            return history
            
        except Exception as e:
            logger.error(f"模型训练失败: {e}")
            if IS_GITHUB_ACTIONS:
                print("::endgroup::")
            return None
    
    def _train_model_original(self, x_train, y_train, epochs, batch_size, learning_rate):
        """原程序的训练方式"""
        try:
            # 更新参数
            if epochs:
                self.epochs = epochs
            if batch_size:
                self.batch_size = batch_size
            
            # 构建模型
            input_shape = (x_train.shape[1], x_train.shape[2])
            self.model = self.build_model(input_shape)
            
            if self.model is None:
                return None
            
            # 如果指定了学习率，重新编译模型
            if learning_rate:
                from tensorflow.keras.optimizers import Adam
                optimizer = Adam(learning_rate=learning_rate)
                self.model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
            
            # 训练模型
            history = self.model.fit(
                x_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.2,
                verbose=0 if IS_GITHUB_ACTIONS else 1
            )
            
            logger.info("模型训练完成")
            return history
            
        except Exception as e:
            logger.error(f"模型训练失败: {e}")
            return None
    
    def plot_training_history(self, history):
        """绘制训练历史 - GitHub Actions优化版本"""
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib不可用，跳过训练曲线绘制")
            return None
            
        try:
            plt.figure(figsize=(12, 4))
            
            # 训练损失
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='训练损失')
            if 'val_loss' in history.history:
                plt.plot(history.history['val_loss'], label='验证损失')
            plt.title('模型损失')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # 训练准确率
            plt.subplot(1, 2, 2)
            if 'accuracy' in history.history:
                plt.plot(history.history['accuracy'], label='训练准确率')
            if 'val_accuracy' in history.history:
                plt.plot(history.history['val_accuracy'], label='验证准确率')
            plt.title('模型准确率')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            plt.tight_layout()
            
            # 保存图片
            if IS_GITHUB_ACTIONS:
                plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
                print("::notice::训练历史图表已保存到 training_history.png")
            else:
                plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
                logger.info("训练历史图表已保存")
            
            plt.close()
            return None
                
        except Exception as e:
            logger.error(f"绘制训练曲线失败: {e}")
            return None
    
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
            
            # 使用兼容性scaler进行标准化
            if hasattr(self.scaler, 'transform'):
                input_data = self.scaler.transform(input_data)
            
            input_data = input_data.reshape(1, self.window_length, -1)
            
            # 预测
            prediction = self.model.predict(input_data, verbose=0)
            
            # 反标准化
            if hasattr(self.scaler, 'inverse_transform'):
                prediction_denorm = self.scaler.inverse_transform(prediction)
            else:
                prediction_denorm = prediction
            
            # 转换为实际号码
            red_numbers = self._convert_to_red_numbers(prediction_denorm[0][:6])
            blue_number = self._convert_to_blue_number(prediction_denorm[0][6:7])
            
            predicted_numbers = red_numbers + [blue_number]
            logger.info(f"预测完成：{predicted_numbers}")
            
            return predicted_numbers
            
        except Exception as e:
            logger.error(f"预测失败: {e}")
            return None
    
    def predict_next_period(self, df, next_period):
        """原程序的核心预测方法 - 兼容性接口
        
        Args:
            df: 数据框架
            next_period: 下期期号
            
        Returns:
            预测的号码数组
        """
        return self.predict_next_numbers(df)
    
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
            'model_trained': self.model is not None,
            'training_periods': len(getattr(self, '_last_training_df', [])) if hasattr(self, '_last_training_df') else '全部'
        } 