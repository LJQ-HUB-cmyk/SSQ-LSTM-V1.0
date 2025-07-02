import pandas as pd
import os
import json
from datetime import datetime
import streamlit as st

class PredictionHistory:
    def __init__(self):
        self.history_dir = "prediction_history"
        self.history_file = os.path.join(self.history_dir, "predictions.csv")
        self.params_file = os.path.join(self.history_dir, "parameters.json")
        
        # 创建历史记录目录
        if not os.path.exists(self.history_dir):
            os.makedirs(self.history_dir)
        
        # 初始化历史记录文件
        self._init_history_file()
    
    def _init_history_file(self):
        """初始化历史记录文件"""
        if not os.path.exists(self.history_file):
            columns = [
                'prediction_time', 'target_period', 'predicted_red_1', 'predicted_red_2', 
                'predicted_red_3', 'predicted_red_4', 'predicted_red_5', 'predicted_red_6',
                'predicted_blue', 'actual_red_1', 'actual_red_2', 'actual_red_3', 
                'actual_red_4', 'actual_red_5', 'actual_red_6', 'actual_blue',
                'red_hits', 'blue_hit', 'total_hits', 'training_periods', 'window_length',
                'epochs', 'batch_size', 'learning_rate'
            ]
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.history_file, index=False, encoding='utf-8')
    
    def save_prediction(self, target_period, predicted_numbers, params):
        """保存预测结果"""
        try:
            # 读取现有历史记录
            df = pd.read_csv(self.history_file, encoding='utf-8')
            
            # 检查是否已存在相同期数和参数的预测
            existing_mask = (
                (df['target_period'] == target_period) &
                (df['training_periods'] == params['training_periods']) &
                (df['window_length'] == params['window_length']) &
                (df['epochs'] == params['epochs']) &
                (df['batch_size'] == params['batch_size']) &
                (df['learning_rate'] == params['learning_rate'])
            )
            
            # 如果存在相同参数的预测，删除旧记录
            if existing_mask.any():
                df = df[~existing_mask]
            
            # 创建新的预测记录
            new_record = {
                'prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'target_period': target_period,
                'predicted_red_1': predicted_numbers[0],
                'predicted_red_2': predicted_numbers[1],
                'predicted_red_3': predicted_numbers[2],
                'predicted_red_4': predicted_numbers[3],
                'predicted_red_5': predicted_numbers[4],
                'predicted_red_6': predicted_numbers[5],
                'predicted_blue': predicted_numbers[6],
                'actual_red_1': None,
                'actual_red_2': None,
                'actual_red_3': None,
                'actual_red_4': None,
                'actual_red_5': None,
                'actual_red_6': None,
                'actual_blue': None,
                'red_hits': None,
                'blue_hit': None,
                'total_hits': None,
                'training_periods': params['training_periods'],
                'window_length': params['window_length'],
                'epochs': params['epochs'],
                'batch_size': params['batch_size'],
                'learning_rate': params['learning_rate']
            }
            
            # 添加新记录
            df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
            
            # 保存到文件
            df.to_csv(self.history_file, index=False, encoding='utf-8')
            
            st.success(f"预测结果已保存：第{target_period}期")
            
        except Exception as e:
            st.error(f"保存预测结果时出错: {e}")
    
    def update_actual_results(self, df_latest):
        """更新实际开奖结果"""
        try:
            # 读取预测历史
            history_df = pd.read_csv(self.history_file, encoding='utf-8')
            
            updated_count = 0
            
            for idx, row in history_df.iterrows():
                target_period = row['target_period']
                
                # 检查是否已有实际结果
                if pd.notna(row['actual_red_1']):
                    continue
                
                # 在最新数据中查找对应期数
                actual_data = df_latest[df_latest['Seq'] == target_period]
                
                if not actual_data.empty:
                    actual_row = actual_data.iloc[0]
                    
                    # 更新实际开奖号码
                    history_df.at[idx, 'actual_red_1'] = actual_row['red_1']
                    history_df.at[idx, 'actual_red_2'] = actual_row['red_2']
                    history_df.at[idx, 'actual_red_3'] = actual_row['red_3']
                    history_df.at[idx, 'actual_red_4'] = actual_row['red_4']
                    history_df.at[idx, 'actual_red_5'] = actual_row['red_5']
                    history_df.at[idx, 'actual_red_6'] = actual_row['red_6']
                    history_df.at[idx, 'actual_blue'] = actual_row['blue']
                    
                    # 计算命中数
                    predicted_red = [row[f'predicted_red_{i}'] for i in range(1, 7)]
                    actual_red = [actual_row[f'red_{i}'] for i in range(1, 7)]
                    
                    red_hits = len(set(predicted_red) & set(actual_red))
                    blue_hit = 1 if row['predicted_blue'] == actual_row['blue'] else 0
                    total_hits = red_hits + blue_hit
                    
                    history_df.at[idx, 'red_hits'] = red_hits
                    history_df.at[idx, 'blue_hit'] = blue_hit
                    history_df.at[idx, 'total_hits'] = total_hits
                    
                    updated_count += 1
            
            # 保存更新后的历史记录
            if updated_count > 0:
                history_df.to_csv(self.history_file, index=False, encoding='utf-8')
                st.success(f"已更新 {updated_count} 条预测记录的实际开奖结果")
            
        except Exception as e:
            st.error(f"更新实际开奖结果时出错: {e}")
    
    def get_history(self):
        """获取预测历史"""
        try:
            if os.path.exists(self.history_file):
                return pd.read_csv(self.history_file, encoding='utf-8')
            else:
                return pd.DataFrame()
        except Exception as e:
            st.error(f"读取预测历史时出错: {e}")
            return pd.DataFrame()
    
    def get_statistics(self):
        """获取预测统计信息"""
        history_df = self.get_history()
        
        if history_df.empty:
            return None
        
        # 只统计有实际结果的记录
        completed_df = history_df.dropna(subset=['actual_red_1'])
        
        if completed_df.empty:
            return None
        
        stats = {
            'total_predictions': len(completed_df),
            'avg_red_hits': completed_df['red_hits'].mean(),
            'avg_total_hits': completed_df['total_hits'].mean(),
            'blue_hit_rate': completed_df['blue_hit'].mean(),
            'max_hits': completed_df['total_hits'].max(),
            'hit_distribution': completed_df['total_hits'].value_counts().sort_index()
        }
        
        return stats