"""
特征选择模块
使用多种方法选择重要特征
"""
import pandas as pd
import numpy as np
from typing import List, Set, Optional
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import settings

try:
    from sklearn.feature_selection import (
        SelectKBest, mutual_info_regression, f_regression,
        RFE
    )
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class FeatureSelector:
    """特征选择器"""

    def __init__(self, df: pd.DataFrame, target_col: str = 'Target_Reg'):
        """
        参数:
            df: 包含特征和目标变量的DataFrame
            target_col: 目标变量列名
        """
        self.df = df.copy()
        self.target_col = target_col
        self.feature_cols = [col for col in df.columns if col not in [
            'Future_Return', 'Target_Class', 'Target_Reg',
            'open', 'high', 'low', 'close', 'volume'
        ]]

    def select_features(self, method: str = None, top_n: int = None) -> List[str]:
        """
        选择特征的主方法

        参数:
            method: 选择方法 ('correlation', 'mutual_info', 'tree'
            top_n: 选择前N个特征
        """
        if method is None:
            method = settings.FEATURE_SELECTION_METHOD
        if top_n is None:
            top_n = settings.TOP_N_FEATURES

        if method == 'correlation':
            return self.select_by_correlation(top_n)
        elif method == 'mutual_info':
            return self.select_by_mutual_info(top_n)
        elif method == 'tree':
            return self.select_by_tree_importance(top_n)
        else:
            print(f"未知方法: {method}，使用相关性方法")
            return self.select_by_correlation(top_n)

    def select_by_correlation(self, top_n: int) -> List[str]:
        """基于相关性选择特征"""
        df_clean = self.df[self.feature_cols + [self.target_col]].dropna()

        # 计算与目标变量的相关性
        correlations = df_clean.corr()[self.target_col].abs()
        correlations = correlations.drop(self.target_col)

        # 排序并选择前N个
        selected = correlations.sort_values(ascending=False).head(top_n).index.tolist()

        print(f"相关性选择: 选出{len(selected)}个特征")
        return selected

    def select_by_mutual_info(self, top_n: int) -> List[str]:
        """基于互信息选择特征"""
        if not SKLEARN_AVAILABLE:
            print("scikit-learn不可用，使用相关性方法")
            return self.select_by_correlation(top_n)

        df_clean = self.df[self.feature_cols + [self.target_col]].dropna()
        X = df_clean[self.feature_cols].fillna(0)
        y = df_clean[self.target_col]

        # 互信息回归
        mi = mutual_info_regression(X, y, random_state=settings.RANDOM_STATE)
        mi_series = pd.Series(mi, index=self.feature_cols)

        # 排序并选择前N个
        selected = mi_series.sort_values(ascending=False).head(top_n).index.tolist()

        print(f"互信息选择: 选出{len(selected)}个特征")
        return selected

    def select_by_tree_importance(self, top_n: int) -> List[str]:
        """基于树模型特征重要性选择"""
        if not SKLEARN_AVAILABLE:
            print("scikit-learn不可用，使用相关性方法")
            return self.select_by_correlation(top_n)

        df_clean = self.df[self.feature_cols + [self.target_col]].dropna()
        X = df_clean[self.feature_cols].fillna(0)
        y = df_clean[self.target_col]

        # 使用随机森林
        rf = RandomForestRegressor(n_estimators=100, random_state=settings.RANDOM_STATE)
        rf.fit(X, y)

        # 获取特征重要性
        importance = pd.Series(rf.feature_importances_, index=self.feature_cols)

        # 排序并选择前N个
        selected = importance.sort_values(ascending=False).head(top_n).index.tolist()

        print(f"树模型选择: 选出{len(selected)}个特征")
        return selected

    def remove_highly_correlated(self, feature_list: List[str], threshold: float = 0.95) -> List[str]:
        """
        移除高度相关的特征

        参数:
            feature_list: 候选特征列表
            threshold: 相关性阈值
        """
        df_clean = self.df[feature_list].dropna()

        # 计算相关矩阵
        corr_matrix = df_clean.corr().abs()

        # 选择上三角
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # 找出要删除的特征
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        # 保留特征
        selected = [col for col in feature_list if col not in to_drop]

        print(f"移除高度相关特征: 删除{len(to_drop)}个，保留{len(selected)}个")
        return selected

    def get_feature_importance(self, feature_list: List[str]) -> pd.DataFrame:
        """获取特征重要性排序"""
        df_clean = self.df[feature_list + [self.target_col]].dropna()

        # 计算相关性
        corr = df_clean.corr()[self.target_col].abs()
        corr = corr.drop(self.target_col)

        importance_df = pd.DataFrame({
            'Feature': corr.index,
            'Correlation': corr.values
        }).sort_values('Correlation', ascending=False)

        return importance_df

    def filter_features(self, df: pd.DataFrame, selected_features: List[str]) -> pd.DataFrame:
        """
        过滤DataFrame，只保留选中的特征和目标变量
        """
        # 保留原始数据列
        keep_cols = ['open', 'high', 'low', 'close', 'volume']
        target_cols = ['Future_Return', 'Target_Class', 'Target_Reg']

        final_cols = keep_cols + selected_features + target_cols
        final_cols = [col for col in final_cols if col in df.columns]

        return df[final_cols]
