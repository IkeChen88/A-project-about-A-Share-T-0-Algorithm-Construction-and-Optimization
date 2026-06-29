"""
模型可解释性可视化模块
包含SHAP可视化、特征重要性可视化等功能
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

warnings.filterwarnings('ignore')

# 设置绘图风格
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class ExplainabilityVisualizer:
    """
    模型可解释性可视化类
    """

    def __init__(
        self,
        feature_names: List[str],
        sequence_length: int = 20,
        output_dir: str = "explainability_results"
    ):
        """
        初始化可视化器

        参数:
            feature_names: 特征名称列表
            sequence_length: 序列长度
            output_dir: 输出目录
        """
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self.sequence_length = sequence_length
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save_figure(self, filename: str, tight_layout: bool = True, dpi: int = 300):
        """
        保存图像

        参数:
            filename: 文件名
            tight_layout: 是否调整布局
            dpi: DPI
        """
        if tight_layout:
            plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=dpi, bbox_inches='tight')
        plt.close()

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        method_name: str = "Feature",
        top_n: int = 15,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        绘制特征重要性条形图

        参数:
            importance_df: 特征重要性DataFrame
            method_name: 方法名称
            top_n: 显示的特征数
            figsize: 图像尺寸
        """
        df_plot = importance_df.head(top_n).copy()

        fig, ax = plt.subplots(figsize=figsize)

        # 绘制条形图
        colors = sns.color_palette("viridis", len(df_plot))
        bars = ax.barh(range(len(df_plot)), df_plot['importance'], color=colors)

        # 添加数值标签
        for i, (bar, imp) in enumerate(zip(bars, df_plot['importance'])):
            if 'std' in df_plot.columns:
                label = f"{imp:.4f} (±{df_plot['std'].iloc[i]:.4f})"
            else:
                label = f"{imp:.4f}"
            ax.text(imp + max(df_plot['importance']) * 0.01, i, label, va='center')

        # 设置标签和标题
        ax.set_yticks(range(len(df_plot)))
        ax.set_yticklabels(df_plot['feature'])
        ax.set_xlabel('Importance')
        ax.set_title(f'{method_name} Feature Importance (Top {top_n})', fontsize=14, fontweight='bold')
        ax.invert_yaxis()  # 反转y轴，最重要的特征在顶部

        self._save_figure(f'{method_name.lower()}_feature_importance.png')
        print(f"Saved: {method_name} feature importance plot")

    def plot_comparative_importance(
        self,
        importances_dict: Dict[str, pd.DataFrame],
        top_n: int = 10,
        figsize: Tuple[int, int] = (12, 8)
    ):
        """
        绘制对比特征重要性

        参数:
            importances_dict: 包含多种方法的重要性字典
            top_n: 显示的特征数
            figsize: 图像尺寸
        """
        # 合并所有方法的重要性
        dfs = []
        for method, df in importances_dict.items():
            df = df[['feature', 'importance']].copy()
            df['method'] = method
            # 标准化重要性以便比较
            df['importance_norm'] = df['importance'] / df['importance'].max()
            dfs.append(df)

        combined_df = pd.concat(dfs, axis=0)

        # 获取在任一方法中排名靠前的特征
        top_features = set()
        for df in importances_dict.values():
            top_features.update(df['feature'].head(top_n).tolist())

        plot_df = combined_df[combined_df['feature'].isin(top_features)]

        # 绘制对比图
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(data=plot_df, x='importance_norm', y='feature', hue='method', ax=ax)

        ax.set_xlabel('Normalized Importance')
        ax.set_title(f'Comparative Feature Importance (Top {top_n} by each method)', fontsize=14, fontweight='bold')
        ax.legend(title='Method')

        self._save_figure('comparative_feature_importance.png')
        print("Saved: comparative feature importance plot")

    def plot_shap_summary(
        self,
        shap_values: np.ndarray,
        X_explain: np.ndarray,
        top_n: int = 15,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        绘制SHAP摘要图

        参数:
            shap_values: SHAP值数组
            X_explain: 解释用的数据
            top_n: 显示的特征数
            figsize: 图像尺寸
        """
        if not SHAP_AVAILABLE:
            print("SHAP not available for plotting")
            return

        # 展平数据以便SHAP绘图
        X_flat = X_explain.reshape(X_explain.shape[0], -1)

        # 展平SHAP值
        if len(shap_values.shape) == 3:
            shap_flat = shap_values.reshape(shap_values.shape[0], -1)
        else:
            shap_flat = np.array(shap_values)

        # 按特征聚合SHAP值并选择top_n
        feature_imp = np.abs(shap_flat).mean(axis=0)
        top_indices = feature_imp.argsort()[::-1][:top_n]

        # 创建展平的特征名
        feature_names_flat = [
            f"{feat}_t{t}"
            for t in range(self.sequence_length)
            for feat in self.feature_names
        ]
        top_feature_names = [feature_names_flat[i] for i in top_indices]

        # 绘制摘要图
        try:
            # 使用SHAP的摘要图
            plt.figure(figsize=figsize)
            shap.summary_plot(
                shap_flat[:, top_indices],
                X_flat[:, top_indices],
                feature_names=top_feature_names,
                plot_type="dot",
                max_display=top_n,
                show=False
            )
            plt.title(f'SHAP Summary Plot (Top {top_n} Features)', fontsize=14, fontweight='bold')
            self._save_figure('shap_summary_plot.png')
            print("Saved: SHAP summary plot")
        except Exception as e:
            print(f"SHAP summary plot failed: {e}")

    def plot_shap_force(
        self,
        explainer: object,
        shap_values: np.ndarray,
        X_explain: np.ndarray,
        sample_idx: int = 0,
        figsize: Tuple[int, int] = (12, 6)
    ):
        """
        绘制单个样本的SHAP力图

        参数:
            explainer: SHAP解释器
            shap_values: SHAP值数组
            X_explain: 解释用的数据
            sample_idx: 样本索引
            figsize: 图像尺寸
        """
        if not SHAP_AVAILABLE:
            print("SHAP not available for plotting")
            return

        try:
            X_flat = X_explain.reshape(X_explain.shape[0], -1)
            if len(shap_values.shape) == 3:
                shap_flat = shap_values.reshape(shap_values.shape[0], -1)
            else:
                shap_flat = np.array(shap_values)

            feature_names_flat = [
                f"{feat}_t{t}"
                for t in range(self.sequence_length)
                for feat in self.feature_names
            ]

            # 获取最重要的特征
            sample_shap = shap_flat[sample_idx]
            top_feat_indices = np.abs(sample_shap).argsort()[::-1][:10]

            plt.figure(figsize=figsize)
            shap.force_plot(
                explainer.expected_value if hasattr(explainer, 'expected_value') else np.mean(sample_shap),
                sample_shap[top_feat_indices],
                X_flat[sample_idx, top_feat_indices],
                feature_names=[feature_names_flat[i] for i in top_feat_indices],
                matplotlib=True,
                show=False
            )
            plt.title(f'SHAP Force Plot for Sample {sample_idx}', fontsize=14, fontweight='bold')
            self._save_figure(f'shap_force_sample_{sample_idx}.png')
            print(f"Saved: SHAP force plot for sample {sample_idx}")
        except Exception as e:
            print(f"SHAP force plot failed: {e}")

    def plot_shap_dependence(
        self,
        shap_values: np.ndarray,
        X_explain: np.ndarray,
        feature_name: str,
        interaction_feature: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        绘制SHAP依赖图

        参数:
            shap_values: SHAP值数组
            X_explain: 解释用的数据
            feature_name: 要分析的特征名
            interaction_feature: 交互特征名
            figsize: 图像尺寸
        """
        if not SHAP_AVAILABLE:
            print("SHAP not available for plotting")
            return

        try:
            X_flat = X_explain.reshape(X_explain.shape[0], -1)
            if len(shap_values.shape) == 3:
                shap_flat = shap_values.reshape(shap_values.shape[0], -1)
            else:
                shap_flat = np.array(shap_values)

            feature_names_flat = [
                f"{feat}_t{t}"
                for t in range(self.sequence_length)
                for feat in self.feature_names
            ]

            # 查找特征索引
            if feature_name in feature_names_flat:
                feat_idx = feature_names_flat.index(feature_name)
            else:
                # 尝试查找该特征的任意时间步
                matches = [i for i, name in enumerate(feature_names_flat) if feature_name in name]
                if matches:
                    feat_idx = matches[len(matches) // 2]  # 使用中间时间步
                else:
                    print(f"Feature {feature_name} not found")
                    return

            interaction_index = None
            if interaction_feature:
                if interaction_feature in feature_names_flat:
                    interaction_index = feature_names_flat.index(interaction_feature)
                else:
                    matches = [i for i, name in enumerate(feature_names_flat) if interaction_feature in name]
                    if matches:
                        interaction_index = matches[len(matches) // 2]

            plt.figure(figsize=figsize)
            shap.dependence_plot(
                feat_idx,
                shap_flat,
                X_flat,
                feature_names=feature_names_flat,
                interaction_index=interaction_index,
                show=False
            )
            plt.title(f'SHAP Dependence Plot: {feature_name}', fontsize=14, fontweight='bold')
            self._save_figure(f'shap_dependence_{feature_name}.png')
            print(f"Saved: SHAP dependence plot for {feature_name}")
        except Exception as e:
            print(f"SHAP dependence plot failed: {e}")

    def plot_feature_stability(
        self,
        stability_df: pd.DataFrame,
        top_n: int = 15,
        figsize: Tuple[int, int] = (12, 8)
    ):
        """
        绘制特征稳定性分析图

        参数:
            stability_df: 特征稳定性DataFrame
            top_n: 显示的特征数
            figsize: 图像尺寸
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # 绘制重要性排序
        df_plot = stability_df.head(top_n).copy()
        colors = sns.color_palette("viridis", len(df_plot))

        axes[0].barh(range(len(df_plot)), df_plot['mean_importance'], color=colors)
        axes[0].set_yticks(range(len(df_plot)))
        axes[0].set_yticklabels(df_plot['feature'])
        axes[0].set_xlabel('Mean Importance')
        axes[0].set_title(f'Feature Importance Rankings', fontsize=12, fontweight='bold')
        axes[0].invert_yaxis()

        # 绘制稳定性（变异系数）
        cv_df = stability_df.sort_values('cv_importance').head(top_n)
        axes[1].barh(range(len(cv_df)), cv_df['cv_importance'], color=sns.color_palette("RdYlGn_r", len(cv_df)))
        axes[1].set_yticks(range(len(cv_df)))
        axes[1].set_yticklabels(cv_df['feature'])
        axes[1].set_xlabel('Coefficient of Variation (lower = more stable)')
        axes[1].set_title(f'Feature Stability (lower CV = better)', fontsize=12, fontweight='bold')
        axes[1].invert_yaxis()

        plt.tight_layout()
        self._save_figure('feature_stability_analysis.png')
        print("Saved: feature stability analysis plot")

    def plot_temporal_heatmap(
        self,
        importance_df: pd.DataFrame,
        title: str = "Temporal Feature Importance",
        figsize: Tuple[int, int] = (14, 10)
    ):
        """
        绘制时间特征重要性热力图（按时间步和特征）

        参数:
            importance_df: 特征重要性（需要包含时间步的展开特征名）
            title: 图表标题
            figsize: 图像尺寸
        """
        # 解析特征名获取时间步和特征
        temp_data = []
        for _, row in importance_df.iterrows():
            feat_name = row['feature']
            if '_t' in feat_name:
                parts = feat_name.rsplit('_t', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    temp_data.append({
                        'feature': parts[0],
                        'time_step': int(parts[1]),
                        'importance': row['importance']
                    })

        if not temp_data:
            print("No temporal features found")
            return

        temp_df = pd.DataFrame(temp_data)

        # 创建热力图数据
        heatmap_data = temp_df.pivot(index='feature', columns='time_step', values='importance')

        plt.figure(figsize=figsize)
        sns.heatmap(heatmap_data, cmap='YlOrRd', annot=True, fmt='.4f', cbar_kws={'label': 'Importance'})
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Time Step')
        plt.ylabel('Feature')
        self._save_figure('temporal_importance_heatmap.png')
        print("Saved: temporal importance heatmap")

    def plot_summary_report(
        self,
        importances_dict: Dict[str, pd.DataFrame],
        stability_df: Optional[pd.DataFrame] = None,
        figsize: Tuple[int, int] = (16, 12)
    ):
        """
        绘制综合报告图

        参数:
            importances_dict: 多种方法的重要性字典
            stability_df: 特征稳定性数据（可选）
            figsize: 图像尺寸
        """
        n_plots = len(importances_dict) + (1 if stability_df is not None else 0)
        n_cols = 2
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig = plt.figure(figsize=figsize)

        plot_idx = 1
        for method, df in importances_dict.items():
            ax = fig.add_subplot(n_rows, n_cols, plot_idx)
            plot_df = df.head(10).copy()

            colors = sns.color_palette("viridis", len(plot_df))
            ax.barh(range(len(plot_df)), plot_df['importance'], color=colors)
            ax.set_yticks(range(len(plot_df)))
            ax.set_yticklabels(plot_df['feature'])
            ax.set_xlabel('Importance')
            ax.set_title(f'{method}', fontsize=11, fontweight='bold')
            ax.invert_yaxis()
            plot_idx += 1

        if stability_df is not None:
            ax = fig.add_subplot(n_rows, n_cols, plot_idx)
            plot_df = stability_df.head(10).copy()

            colors = sns.color_palette("RdYlGn_r", len(plot_df))
            bars = ax.barh(range(len(plot_df)), plot_df['mean_importance'],
                          yerr=plot_df['std_importance'], color=colors, capsize=5)
            ax.set_yticks(range(len(plot_df)))
            ax.set_yticklabels(plot_df['feature'])
            ax.set_xlabel('Mean Importance (±std)')
            ax.set_title('Stability Analysis', fontsize=11, fontweight='bold')
            ax.invert_yaxis()

        plt.tight_layout()
        plt.suptitle('Model Explainability Summary Report', fontsize=16, fontweight='bold', y=1.01)
        self._save_figure('explainability_summary_report.png')
        print("Saved: explainability summary report")

    def create_html_report(
        self,
        importances_dict: Dict[str, pd.DataFrame],
        stability_df: Optional[pd.DataFrame] = None
    ):
        """
        创建HTML格式的可解释性分析报告

        参数:
            importances_dict: 多种方法的重要性字典
            stability_df: 特征稳定性数据（可选）
        """
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Model Explainability Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #bdc3c7; padding-bottom: 8px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .figure {{ margin: 20px 0; text-align: center; }}
        .figure img {{ max-width: 100%; }}
    </style>
</head>
<body>
    <h1>Model Explainability Report</h1>

    <h2>Summary</h2>
    <p>Analysis performed on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p>Number of features: {self.n_features}</p>
    <p>Sequence length: {self.sequence_length}</p>
"""

        for method, df in importances_dict.items():
            html_content += f"""
    <h2>{method} Feature Importance</h2>
    <div class="figure">
        <img src="{method.lower()}_feature_importance.png" alt="{method} Feature Importance">
    </div>
    {df.head(20).to_html(index=False)}
"""

        if stability_df is not None:
            html_content += f"""
    <h2>Feature Stability Analysis</h2>
    <div class="figure">
        <img src="feature_stability_analysis.png" alt="Feature Stability Analysis">
    </div>
    {stability_df.head(20).to_html(index=False)}
"""

        html_content += """
</body>
</html>
"""

        with open(self.output_dir / 'explainability_report.html', 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"Saved: HTML report at {self.output_dir / 'explainability_report.html'}")
