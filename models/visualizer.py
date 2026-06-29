"""
完整的可视化模块
包含：预测图、训练曲线图、SHAP可视化、特征重要性图等
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（使用macOS实际可用的字体）
plt.rcParams['font.sans-serif'] = [
    'Heiti TC',
    'STFangsong',
    'STHeiti',
    'Arial Unicode MS',
    'Lucida Grande',
    'SimHei',
    'Source Han Sans SC',
    'Noto Sans CJK SC'
]
plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示

# 设置样式
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')


class ModelVisualizer:
    """模型可视化器"""

    def __init__(self, output_dir: str = "visualizations"):
        """
        初始化可视化器

        参数:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def plot_training_history(self, history: Dict[str, Any], show: bool = False):
        """
        绘制训练历史曲线

        参数:
            history: 训练历史字典
            show: 是否显示（False则仅保存）
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # 损失曲线
        if 'loss' in history:
            axes[0].plot(history['loss'], label='Training Loss')
            if 'val_loss' in history:
                axes[0].plot(history['val_loss'], label='Validation Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

        # 性能指标
        metric_keys = [k for k in history.keys() if not k.startswith('val_') and k != 'loss']
        if metric_keys:
            metric_name = metric_keys[0]
            axes[1].plot(history[metric_name], label=f'Training {metric_name}')
            if f'val_{metric_name}' in history:
                axes[1].plot(history[f'val_{metric_name}'], label=f'Validation {metric_name}')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel(metric_name)
            axes[1].set_title(f'Training & Validation {metric_name}', fontsize=14, fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / "training_history.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if not show:
            plt.close()
        print(f"已保存: {save_path}")

        return fig

    def plot_predictions_comparison(
        self,
        y_true: np.ndarray,
        y_pred_prob: np.ndarray,
        num_samples: int = 100,
        title: str = "预测结果对比",
        show: bool = False
    ):
        """
        绘制预测结果对比图（分类任务）

        参数:
            y_true: 真实标签
            y_pred_prob: 预测概率
            num_samples: 显示的样本数
            title: 图表标题
        """
        # 取部分样本
        n = min(num_samples, len(y_true))
        indices = np.arange(n)

        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

        # 1. 预测概率对比
        y_pred_class = (y_pred_prob.flatten() > 0.5).astype(int)

        ax1 = axes[0]
        ax1.plot(indices, y_true[:n], 'o-', label='True',
                markersize=4, linewidth=2, alpha=0.8)
        ax1.plot(indices, y_pred_class[:n], 'x-', label='Predicted',
                markersize=6, linewidth=1, alpha=0.8)
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Class (0=Down, 1=Up)')
        ax1.set_title('True vs Predicted Classes', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.1, 1.1)

        # 2. 预测概率分布
        ax2 = axes[1]
        ax2.plot(indices, y_pred_prob[:n].flatten(), 'o-',
                color='purple', linewidth=1, markersize=3, alpha=0.7)
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.6, label='Decision Threshold')
        ax2.fill_between(indices, 0.5, y_pred_prob[:n].flatten(), alpha=0.3,
                        where=(y_pred_prob[:n].flatten() >= 0.5), color='green', label='Predicted Up')
        ax2.fill_between(indices, y_pred_prob[:n].flatten(), 0.5, alpha=0.3,
                        where=(y_pred_prob[:n].flatten() < 0.5), color='red', label='Predicted Down')
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Predicted Probability of Up')
        ax2.set_title('Predicted Probabilities', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        save_path = self.output_dir / "predictions_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if not show:
            plt.close()
        print(f"已保存: {save_path}")

        return fig

    def plot_price_prediction_comparison(
        self,
        true_prices: np.ndarray,
        predicted_prices: np.ndarray,
        dates: Optional[np.ndarray] = None,
        num_samples: int = 100,
        title: str = "真实价格vs预测价格对比",
        show: bool = False
    ):
        """
        绘制价格预测对比折线图

        参数:
            true_prices: 真实价格数组
            predicted_prices: 预测价格数组
            dates: 日期数组（可选，用于x轴标签）
            num_samples: 显示的样本数
            title: 图表标题
            show: 是否显示
        """
        # 取部分样本
        n = min(num_samples, len(true_prices))

        fig, axes = plt.subplots(2, 1, figsize=(16, 10))

        # 1. 价格趋势对比
        ax1 = axes[0]
        x_indices = np.arange(n)

        ax1.plot(x_indices, true_prices[:n], 'b-', linewidth=2, label='真实价格', alpha=0.8)
        ax1.plot(x_indices, predicted_prices[:n], 'r--', linewidth=2, label='预测价格', alpha=0.8)

        ax1.set_xlabel('时间顺序', fontsize=12)
        ax1.set_ylabel('价格', fontsize=12)
        ax1.set_title('真实价格vs预测价格', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # 设置日期标签（如果有）
        if dates is not None:
            date_labels = dates[:n]
            if len(date_labels) > 20:
                # 如果日期太多，只显示一部分
                tick_positions = np.linspace(0, n-1, min(10, n), dtype=int)
                ax1.set_xticks(tick_positions)
                ax1.set_xticklabels([str(date_labels[i])[:10] for i in tick_positions], rotation=45, ha='right')

        # 2. 散点图 + 预测误差
        ax2 = axes[1]
        errors = predicted_prices[:n] - true_prices[:n]

        # 绘制散点图
        ax2.scatter(x_indices, true_prices[:n], color='blue', label='真实价格', alpha=0.6, s=30)
        ax2.scatter(x_indices, predicted_prices[:n], color='red', label='预测价格', alpha=0.6, s=30)

        # 绘制误差线
        for i in range(min(n, 50)):  # 避免太多线
            ax2.plot([i, i], [true_prices[i], predicted_prices[i]], 'k-', alpha=0.3, linewidth=0.5)

        ax2.set_xlabel('时间顺序', fontsize=12)
        ax2.set_ylabel('价格', fontsize=12)
        ax2.set_title('价格预测散点对比（含误差线）', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        if dates is not None:
            if len(date_labels) > 20:
                tick_positions = np.linspace(0, n-1, min(10, n), dtype=int)
                ax2.set_xticks(tick_positions)
                ax2.set_xticklabels([str(date_labels[i])[:10] for i in tick_positions], rotation=45, ha='right')

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        save_path = self.output_dir / "price_prediction_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if not show:
            plt.close()
        print(f"已保存: {save_path}")

        return fig

    def plot_price_prediction_detail(
        self,
        true_prices: np.ndarray,
        predicted_prices: np.ndarray,
        dates: Optional[np.ndarray] = None,
        show: bool = False
    ):
        """
        绘制详细的价格预测分析图

        参数:
            true_prices: 真实价格数组
            predicted_prices: 预测价格数组
            dates: 日期数组（可选）
            show: 是否显示
        """
        errors = predicted_prices - true_prices

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2)

        # 1. 价格预测对比
        ax1 = fig.add_subplot(gs[0, :])
        x_indices = np.arange(len(true_prices))
        ax1.plot(x_indices, true_prices, 'b-', linewidth=1.5, label='真实价格', alpha=0.8)
        ax1.plot(x_indices, predicted_prices, 'r--', linewidth=1.5, label='预测价格', alpha=0.8)
        ax1.set_xlabel('时间顺序', fontsize=12)
        ax1.set_ylabel('价格', fontsize=12)
        ax1.set_title('真实价格vs预测价格（完整时间序列）', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 误差分布
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist(errors, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
        ax2.set_xlabel('预测误差', fontsize=12)
        ax2.set_ylabel('频数', fontsize=12)
        ax2.set_title('预测误差分布', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. 散点图（真实vs预测）
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.scatter(true_prices, predicted_prices, alpha=0.6, s=30)
        min_val = min(true_prices.min(), predicted_prices.min())
        max_val = max(true_prices.max(), predicted_prices.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='理想预测')
        ax3.set_xlabel('真实价格', fontsize=12)
        ax3.set_ylabel('预测价格', fontsize=12)
        ax3.set_title('真实价格vs预测价格散点图', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = self.output_dir / "price_prediction_detail.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if not show:
            plt.close()
        print(f"已保存: {save_path}")

        return fig

    def plot_returns_prediction(
        self,
        true_returns: np.ndarray,
        predicted_returns: np.ndarray,
        num_samples: int = 100,
        show: bool = False
    ):
        """
        绘制收益率预测对比图

        参数:
            true_returns: 真实收益率数组
            predicted_returns: 预测收益率数组
            num_samples: 显示的样本数
            show: 是否显示
        """
        n = min(num_samples, len(true_returns))

        fig, axes = plt.subplots(2, 1, figsize=(16, 10))

        # 1. 收益率对比
        ax1 = axes[0]
        x_indices = np.arange(n)

        ax1.bar(x_indices, true_returns[:n], alpha=0.6, label='真实收益率', width=0.4, align='edge')
        ax1.bar(x_indices, predicted_returns[:n], alpha=0.6, label='预测收益率', width=0.4, align='center')
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        ax1.set_xlabel('时间顺序', fontsize=12)
        ax1.set_ylabel('收益率', fontsize=12)
        ax1.set_title('真实收益率vs预测收益率', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # 2. 累积收益率
        ax2 = axes[1]
        cum_true = np.cumprod(1 + true_returns[:n]) - 1
        cum_pred = np.cumprod(1 + predicted_returns[:n]) - 1

        ax2.plot(x_indices, cum_true, 'b-', linewidth=2, label='真实累积收益', alpha=0.8)
        ax2.plot(x_indices, cum_pred, 'r--', linewidth=2, label='预测累积收益', alpha=0.8)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        ax2.set_xlabel('时间顺序', fontsize=12)
        ax2.set_ylabel('累积收益率', fontsize=12)
        ax2.set_title('累积收益率对比', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = self.output_dir / "returns_prediction.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if not show:
            plt.close()
        print(f"已保存: {save_path}")

        return fig

    def plot_prediction_distribution(
        self,
        y_true: np.ndarray,
        y_pred_prob: np.ndarray,
        show: bool = False
    ):
        """
        绘制预测分布图

        参数:
            y_true: 真实标签
            y_pred_prob: 预测概率
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 1. 直方图
        ax1 = axes[0]
        y_pred_flat = y_pred_prob.flatten()
        ax1.hist(y_pred_flat, bins=30, alpha=0.7, edgecolor='black')
        ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('Predicted Probability of Up')
        ax1.set_ylabel('Count')
        ax1.set_title('Distribution of Predictions', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 2. 按真实类别分别展示
        ax2 = axes[1]
        mask_true_up = (y_true.flatten() == 1)
        mask_true_down = (y_true.flatten() == 0)

        ax2.hist(y_pred_flat[mask_true_up], bins=20, alpha=0.6, label='Actually Up', color='green')
        ax2.hist(y_pred_flat[mask_true_down], bins=20, alpha=0.6, label='Actually Down', color='red')
        ax2.axvline(x=0.5, color='black', linestyle='--', linewidth=2)
        ax2.set_xlabel('Predicted Probability of Up')
        ax2.set_ylabel('Count')
        ax2.set_title('Prediction Distribution by True Outcome', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.output_dir / "prediction_distribution.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if not show:
            plt.close()
        print(f"已保存: {save_path}")

        return fig

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        title: str = "Feature Importance",
        top_n: int = 15,
        show: bool = False
    ):
        """
        绘制特征重要性条形图

        参数:
            importance_df: 包含feature和importance列的DataFrame
            title: 图表标题
            top_n: 显示的特征数
        """
        df_plot = importance_df.head(top_n).copy()
        df_plot = df_plot.sort_values('importance', ascending=True)

        fig, ax = plt.subplots(figsize=(10, 8))

        colors = sns.color_palette('viridis', len(df_plot))
        bars = ax.barh(range(len(df_plot)), df_plot['importance'], color=colors)

        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + max(df_plot['importance']) * 0.01,
                    bar.get_y() + bar.get_height()/2,
                    f'{width:.6f}', ha='left', va='center')

        ax.set_yticks(range(len(df_plot)))
        ax.set_yticklabels(df_plot['feature'])
        ax.set_xlabel('Importance')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        save_path = self.output_dir / f"{title.lower().replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if not show:
            plt.close()
        print(f"已保存: {save_path}")

        return fig

    def plot_time_step_importance(
        self,
        time_df: pd.DataFrame,
        show: bool = False
    ):
        """
        绘制时间步重要性图

        参数:
            time_df: 包含time_step和importance的DataFrame
        """
        df_plot = time_df.sort_values('time_step', ascending=True)

        # 提取时间步数
        def extract_step(s):
            if isinstance(s, str) and s.startswith('t-'):
                return int(s.split('-')[1])
            return s

        df_plot['step_num'] = df_plot['time_step'].apply(extract_step)
        df_plot = df_plot.sort_values('step_num', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = plt.cm.Reds(np.linspace(0.3, 1, len(df_plot)))
        bars = ax.barh(range(len(df_plot)), df_plot['importance'], color=colors)

        ax.set_yticks(range(len(df_plot)))
        ax.set_yticklabels(df_plot['time_step'])
        ax.set_xlabel('Importance')
        ax.set_title('Time Step Importance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        save_path = self.output_dir / "time_step_importance.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if not show:
            plt.close()
        print(f"已保存: {save_path}")

        return fig

    def plot_model_performance(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        show: bool = False
    ):
        """
        绘制模型性能对比图

        参数:
            metrics_dict: 数据集 -> 指标 -> 数值 的字典
        """
        datasets = list(metrics_dict.keys())
        metrics = list(metrics_dict[datasets[0]].keys())

        fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 5))
        if len(metrics) == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            values = [metrics_dict[ds][metric] for ds in datasets]
            colors = ['skyblue' if ds == 'train' else 'orange' if ds == 'val' else 'green' for ds in datasets]
            bars = ax.bar(datasets, values, color=colors, alpha=0.7)
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} by Dataset', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.4f}',
                       ha='center', va='bottom')

        plt.tight_layout()
        save_path = self.output_dir / "model_performance.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if not show:
            plt.close()
        print(f"已保存: {save_path}")

        return fig

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred_prob: np.ndarray,
        threshold: float = 0.5,
        show: bool = False
    ):
        """
        绘制混淆矩阵

        参数:
            y_true: 真实标签
            y_pred_prob: 预测概率
            threshold: 分类阈值
        """
        from sklearn.metrics import confusion_matrix

        y_pred = (y_pred_prob.flatten() > threshold).astype(int)
        cm = confusion_matrix(y_true.flatten(), y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])

        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

        plt.tight_layout()
        save_path = self.output_dir / "confusion_matrix.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if not show:
            plt.close()
        print(f"已保存: {save_path}")

        return fig


class SHAPVisualizer:
    """SHAP可视化器"""

    def __init__(self, output_dir: str = "visualizations"):
        """
        初始化SHAP可视化器

        参数:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def plot_shap_summary(
        self,
        shap_values: np.ndarray,
        X_explain: np.ndarray,
        feature_names: list,
        top_n: int = 20,
        show: bool = False
    ):
        """
        绘制SHAP摘要图

        参数:
            shap_values: SHAP值数组
            X_explain: 解释数据
            feature_names: 特征名称列表
            top_n: 显示的特征数
        """
        try:
            import shap

            # 聚合特征（按基础特征名）
            base_features = []
            base_shap_values = []
            base_X_values = []

            # 分析特征名，提取基础特征
            feature_groups = {}
            for i, feat in enumerate(feature_names):
                base_name = feat.split('_t')[0] if '_t' in feat else feat
                if base_name not in feature_groups:
                    feature_groups[base_name] = {'indices': [], 'shap': [], 'X': []}
                feature_groups[base_name]['indices'].append(i)

            # 聚合
            for base_name, data in feature_groups.items():
                base_features.append(base_name)
                # 取该特征所有时间步绝对值的均值
                shap_agg = np.abs(shap_values[:, data['indices']]).mean(axis=1)
                base_shap_values.append(shap_agg)
                # 取该特征最新时间步的值用于颜色
                X_agg = X_explain[:, data['indices'][-1]]
                base_X_values.append(X_agg)

            base_shap_values = np.array(base_shap_values).T
            base_X_values = np.array(base_X_values).T

            # 计算重要性排序
            feat_importance = np.abs(base_shap_values).mean(axis=0)
            top_indices = feat_importance.argsort()[::-1][:top_n]

            plt.figure(figsize=(10, min(10, top_n*0.4+2)))

            shap.summary_plot(
                base_shap_values[:, top_indices],
                base_X_values[:, top_indices],
                feature_names=[base_features[i] for i in top_indices],
                plot_type='dot',
                show=False,
                max_display=top_n
            )

            plt.title('SHAP Summary Plot', fontsize=14, fontweight='bold')

            save_path = self.output_dir / "shap_summary_plot.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if not show:
                plt.close()
            print(f"已保存: {save_path}")

        except Exception as e:
            print(f"SHAP摘要图生成失败: {e}")

    def plot_shap_feature_importance(
        self,
        shap_importance_df: pd.DataFrame,
        top_n: int = 15,
        show: bool = False
    ):
        """
        绘制SHAP特征重要性条形图

        参数:
            shap_importance_df: SHAP特征重要性DataFrame
            top_n: 显示的特征数
        """
        df_plot = shap_importance_df.head(top_n).copy()
        df_plot = df_plot.sort_values('shap_importance', ascending=True)

        fig, ax = plt.subplots(figsize=(10, 8))

        colors = sns.color_palette('viridis', len(df_plot))
        bars = ax.barh(range(len(df_plot)), df_plot['shap_importance'], color=colors)

        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + max(df_plot['shap_importance']) * 0.01,
                    bar.get_y() + bar.get_height()/2,
                    f'{width:.2e}', ha='left', va='center')

        ax.set_yticks(range(len(df_plot)))
        ax.set_yticklabels(df_plot['feature'])
        ax.set_xlabel('SHAP Importance')
        ax.set_title('SHAP Feature Importance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        save_path = self.output_dir / "shap_feature_importance.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if not show:
            plt.close()
        print(f"已保存: {save_path}")

        return fig

    def plot_shap_force(
        self,
        shap_values: np.ndarray,
        X_explain: np.ndarray,
        feature_names: list,
        sample_idx: int = 0,
        show: bool = False
    ):
        """
        绘制单个样本的SHAP力图

        参数:
            shap_values: SHAP值数组
            X_explain: 解释数据
            feature_names: 特征名称列表
            sample_idx: 样本索引
        """
        try:
            import shap

            # 对该样本，取Top特征
            sample_shap = shap_values[sample_idx]
            top_indices = np.abs(sample_shap).argsort()[::-1][:10]

            plt.figure(figsize=(15, 4))

            try:
                # 尝试使用shap.force_plot
                explainer = None
                try:
                    # 创建一个简单的解释器
                    base_value = shap_values.mean()
                    shap.force_plot(
                        base_value,
                        sample_shap[top_indices],
                        X_explain[sample_idx, top_indices],
                        feature_names=[feature_names[i] for i in top_indices],
                        matplotlib=True,
                        show=False
                    )
                except:
                    # 备用的简单图
                    pass

            except Exception as e:
                print(f"标准SHAP力图生成失败，使用替代方案: {e}")

            # 使用简单的可视化作为替代
            fig, ax = plt.subplots(figsize=(12, 5))

            # 只显示Top特征
            top_features = [feature_names[i] for i in top_indices]
            top_shap = sample_shap[top_indices]

            colors = ['red' if v < 0 else 'green' for v in top_shap]
            bars = ax.barh(range(len(top_shap)), top_shap, color=colors)

            for i, bar in enumerate(bars):
                width = bar.get_width()
                align = 'right' if width < 0 else 'left'
                pos = width + (-0.01 if width < 0 else 0.01) * max(np.abs(top_shap))
                ax.text(pos, bar.get_y() + bar.get_height()/2,
                        f'{top_features[i]}: {width:.4f}', ha=align, va='center')

            ax.set_yticks([])
            ax.set_xlabel('SHAP Value')
            ax.set_title(f'SHAP Values for Sample {sample_idx}', fontsize=14, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax.grid(True, alpha=0.3)

            save_path = self.output_dir / f"shap_force_sample_{sample_idx}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if not show:
                plt.close()
            print(f"已保存: {save_path}")

        except Exception as e:
            print(f"SHAP力图生成失败: {e}")

    def plot_shap_dependence(
        self,
        shap_values: np.ndarray,
        X_explain: np.ndarray,
        feature_names: list,
        feature_name: str,
        interaction_feature: Optional[str] = None,
        show: bool = False
    ):
        """
        绘制SHAP依赖图

        参数:
            shap_values: SHAP值数组
            X_explain: 解释数据
            feature_names: 特征名称列表
            feature_name: 要分析的特征名
            interaction_feature: 交互特征名（可选）
        """
        try:
            import shap

            # 查找特征索引
            if feature_name in feature_names:
                feat_idx = feature_names.index(feature_name)
            else:
                # 查找包含该名称的特征
                matches = [i for i, name in enumerate(feature_names) if feature_name in name]
                if matches:
                    feat_idx = matches[-1]  # 使用最新时间步
                else:
                    print(f"未找到特征: {feature_name}")
                    return

            interaction_idx = None
            if interaction_feature:
                if interaction_feature in feature_names:
                    interaction_idx = feature_names.index(interaction_feature)
                else:
                    matches = [i for i, name in enumerate(feature_names) if interaction_feature in name]
                    if matches:
                        interaction_idx = matches[-1]

            fig, ax = plt.subplots(figsize=(10, 6))

            # 绘制散点图
            scatter = ax.scatter(
                X_explain[:, feat_idx],
                shap_values[:, feat_idx],
                c=X_explain[:, interaction_idx] if interaction_idx else X_explain[:, feat_idx],
                alpha=0.5,
                cmap='coolwarm'
            )

            # 添加趋势线
            z = np.polyfit(X_explain[:, feat_idx], shap_values[:, feat_idx], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(X_explain[:, feat_idx]), max(X_explain[:, feat_idx]), 100)
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)

            ax.set_xlabel(f'Feature Value: {feature_names[feat_idx]}')
            ax.set_ylabel('SHAP Value')
            ax.set_title(f'SHAP Dependence Plot: {feature_names[feat_idx]}', fontsize=14, fontweight='bold')
            if interaction_idx:
                plt.colorbar(scatter, label=f'Interaction Feature: {feature_names[interaction_idx]}')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            safe_feat_name = feature_name.replace('/', '_').replace('\\', '_')
            save_path = self.output_dir / f"shap_dependence_{safe_feat_name}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if not show:
                plt.close()
            print(f"已保存: {save_path}")

        except Exception as e:
            print(f"SHAP依赖图生成失败: {e}")


def create_complete_visualization_report(
    visualizer: ModelVisualizer,
    shap_visualizer: Optional[SHAPVisualizer],
    data_dict: dict,
    model,
    history: dict,
    importance_df: pd.DataFrame,
    time_df: pd.DataFrame,
    shap_data: Optional[dict] = None,
    price_data: Optional[dict] = None
):
    """
    创建完整的可视化报告

    参数:
        visualizer: 模型可视化器
        shap_visualizer: SHAP可视化器（可选）
        data_dict: 数据字典
        model: 训练好的模型
        history: 训练历史
        importance_df: 特征重要性
        time_df: 时间步重要性
        shap_data: SHAP数据字典（可选）
        price_data: 价格数据字典（可选，包含true_prices等）
    """
    print("\n" + "="*80)
    print("开始生成可视化报告")
    print("="*80)

    # 1. 训练历史图
    if history:
        print("\n1. 生成训练历史图...")
        visualizer.plot_training_history(history)

    # 2. 预测相关图
    print("\n2. 生成预测图...")
    y_pred_train = model.predict(data_dict['X_train'])
    y_pred_val = model.predict(data_dict['X_val'])
    y_pred_test = model.predict(data_dict['X_test'])

    visualizer.plot_predictions_comparison(
        data_dict['y_test'], y_pred_test, title="Test Set Predictions"
    )
    visualizer.plot_prediction_distribution(data_dict['y_test'], y_pred_test)
    visualizer.plot_confusion_matrix(data_dict['y_test'], y_pred_test)

    # 3. 价格预测相关图（新增）
    if price_data is not None:
        print("\n3. 生成价格预测图...")
        if 'true_prices' in price_data and 'predicted_prices' in price_data:
            visualizer.plot_price_prediction_comparison(
                price_data['true_prices'],
                price_data['predicted_prices'],
                price_data.get('dates', None),
                title="真实价格vs预测价格"
            )
            visualizer.plot_price_prediction_detail(
                price_data['true_prices'],
                price_data['predicted_prices'],
                price_data.get('dates', None)
            )
            if 'true_returns' in price_data and 'predicted_returns' in price_data:
                visualizer.plot_returns_prediction(
                    price_data['true_returns'],
                    price_data['predicted_returns']
                )

    # 4. 模型性能图
    print("\n4. 生成性能图...")
    train_metrics = model.evaluate(data_dict['X_train'], data_dict['y_train'])
    val_metrics = model.evaluate(data_dict['X_val'], data_dict['y_val'])
    test_metrics = model.evaluate(data_dict['X_test'], data_dict['y_test'])

    visualizer.plot_model_performance({
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics
    })

    # 5. 特征重要性图
    print("\n5. 生成特征重要性图...")
    visualizer.plot_feature_importance(importance_df, title="Feature Importance")
    if time_df is not None:
        visualizer.plot_time_step_importance(time_df)

    # 6. SHAP可视化
    if shap_data and shap_visualizer:
        print("\n6. 生成SHAP可视化...")
        if 'shap_importance_df' in shap_data:
            shap_visualizer.plot_shap_feature_importance(shap_data['shap_importance_df'])

        if all(k in shap_data for k in ['shap_values', 'X_explain', 'feature_names']):
            shap_visualizer.plot_shap_summary(
                shap_data['shap_values'],
                shap_data['X_explain'],
                shap_data['feature_names']
            )

            # 为Top3特征生成依赖图
            if 'shap_importance_df' in shap_data:
                top_features = shap_data['shap_importance_df']['feature'].head(3).tolist()
                for feat in top_features:
                    shap_visualizer.plot_shap_dependence(
                        shap_data['shap_values'],
                        shap_data['X_explain'],
                        shap_data['feature_names'],
                        feat
                    )

            # 为第一个样本生成力图
            shap_visualizer.plot_shap_force(
                shap_data['shap_values'],
                shap_data['X_explain'],
                shap_data['feature_names'],
                sample_idx=0
            )

    print("\n" + "="*80)
    print("可视化报告生成完成！")
    print("="*80)
