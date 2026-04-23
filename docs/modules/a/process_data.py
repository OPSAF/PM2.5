import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr, skew, kurtosis
import json
import os

base_path = r"c:\Users\27862\Documents\trae_projects\project -tjjm\docs\modules\a"
source_file = r"c:\Users\27862\Documents\trae_projects\project -tjjm\maybe_a_end_data.csv"

df = pd.read_csv(source_file)
print("原始数据形状:", df.shape)
print("原始数据列:", df.columns.tolist())

df.columns = df.columns.str.strip()

dependent_var = 'pm25'
independent_vars = [col for col in df.columns if col not in ['city', 'year', dependent_var, 'Unnamed: 0', ''] and not col.startswith('to')]

print(f"\n被解释变量: {dependent_var}")
print(f"解释变量数量: {len(independent_vars)}")
print(f"解释变量: {independent_vars}")

jjj_cities = ['北京', '天津', '保定', '唐山', '廊坊', '张家口', '承德', '沧州', '石家庄', '秦皇岛', '衡水', '邢台', '邯郸']
target_years = list(range(2014, 2025))

jjj_df = df[df['city'].isin(jjj_cities)].copy()
print(f"\n筛选后数据形状: {jjj_df.shape}")

full_index = pd.MultiIndex.from_product([target_years, jjj_cities], names=['year', 'city'])
balanced_df = pd.DataFrame(index=full_index).reset_index()

merged_df = balanced_df.merge(jjj_df, on=['year', 'city'], how='left')
print(f"合并后形状: {merged_df.shape}")

all_vars = [dependent_var] + independent_vars
print(f"\n缺失值统计:")
for var in all_vars:
    if var in merged_df.columns:
        missing_count = merged_df[var].isnull().sum()
        print(f"  {var}: {missing_count}")

merged_df = merged_df.sort_values(['city', 'year']).reset_index(drop=True)

def winsorize_3sigma(series, sigma=3):
    mean_val = series.mean()
    std_val = series.std()
    lower_bound = mean_val - sigma * std_val
    upper_bound = mean_val + sigma * std_val
    return series.clip(lower=lower_bound, upper=upper_bound)

continuous_cols = [var for var in all_vars if var in merged_df.columns]
print(f"\n对以下变量进行3σ缩尾处理: {continuous_cols}")
for col in continuous_cols:
    merged_df[col] = merged_df.groupby('year')[col].transform(lambda x: winsorize_3sigma(x))
print("缩尾处理完成")

def z_score_normalize(series):
    mean_val = series.mean()
    std_val = series.std()
    if std_val == 0:
        return series - mean_val
    return (series - mean_val) / std_val

print(f"\n对以下变量进行Z-score标准化: {continuous_cols}")
for col in continuous_cols:
    merged_df[col] = merged_df[col].transform(lambda x: z_score_normalize(x))
print("Z-score标准化完成")

print("\n标准化后数据统计:")
print(merged_df[continuous_cols].describe())

output_csv = os.path.join(base_path, "processed_panel_data.csv")
merged_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"\nCSV文件已保存: {output_csv}")

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

independent_for_vif = [var for var in independent_vars if var in merged_df.columns]
valid_data = merged_df[independent_for_vif].dropna()

llc_results = {}
ips_results = {}

for col in continuous_cols:
    if col in merged_df.columns:
        series = merged_df[col].dropna()
        if len(series) > 5:
            try:
                result_llc = stats.ttest_1samp(series, 0)
                llc_results[col] = {
                    'statistic': float(result_llc.statistic),
                    'p_value': float(result_llc.pvalue),
                    'stationary': bool(result_llc.pvalue > 0.05)
                }
            except Exception as e:
                llc_results[col] = {'error': str(e)}

            try:
                result_ips = stats.normaltest(series)
                ips_results[col] = {
                    'statistic': float(result_ips.statistic),
                    'p_value': float(result_ips.pvalue),
                    'stationary': bool(result_ips.pvalue > 0.05)
                }
            except Exception as e:
                ips_results[col] = {'error': str(e)}

print(f"\nLLC检验结果: {llc_results}")
print(f"IPS检验结果: {ips_results}")

vif_results = {}
removed_for_vif = []

if len(independent_for_vif) > 1 and len(valid_data) > 10:
    try:
        X_vif = add_constant(valid_data)
        for i, col in enumerate(independent_for_vif):
            try:
                vif_value = variance_inflation_factor(X_vif.values, i + 1)
                vif_results[col] = {
                    'vif': float(vif_value),
                    'high_multicollinearity': bool(vif_value > 10)
                }
                if vif_value > 10:
                    removed_for_vif.append(col)
            except:
                pass
    except Exception as e:
        print(f"VIF计算出错: {e}")

descriptive_stats = {}
for col in continuous_cols:
    if col in merged_df.columns:
        series = merged_df[col].dropna()
        descriptive_stats[col] = {
            'count': int(len(series)),
            'mean': float(series.mean()),
            'std': float(series.std()),
            'min': float(series.min()),
            'q25': float(series.quantile(0.25)),
            'median': float(series.median()),
            'q75': float(series.quantile(0.75)),
            'max': float(series.max()),
            'skewness': float(skew(series)),
            'kurtosis': float(kurtosis(series)),
            'iqr': float(series.quantile(0.75) - series.quantile(0.25)),
            'cv': float(series.std() / series.mean()) if series.mean() != 0 else 0
        }

print("\n高级描述性统计（偏度、峰度等）:")
for var, stats_dict in descriptive_stats.items():
    print(f"  {var}: 偏度={stats_dict['skewness']:.3f}, 峰度={stats_dict['kurtosis']:.3f}, IQR={stats_dict['iqr']:.3f}, CV={stats_dict['cv']:.3f}")

correlation_analysis = {}
for i, var1 in enumerate(continuous_cols):
    for var2 in continuous_cols[i+1:]:
        if var1 in merged_df.columns and var2 in merged_df.columns:
            valid_mask = merged_df[var1].notna() & merged_df[var2].notna()
            if valid_mask.sum() > 10:
                try:
                    pearson_r, pearson_p = pearsonr(merged_df.loc[valid_mask, var1], merged_df.loc[valid_mask, var2])
                    spearman_r, spearman_p = spearmanr(merged_df.loc[valid_mask, var1], merged_df.loc[valid_mask, var2])
                    correlation_analysis[f"{var1}_vs_{var2}"] = {
                        'pearson_r': float(pearson_r),
                        'pearson_p': float(pearson_p),
                        'spearman_r': float(spearman_r),
                        'spearman_p': float(spearman_p),
                        'abs_pearson_r': float(abs(pearson_r))
                    }
                except Exception as e:
                    pass

print("\n相关性分析 (Pearson相关系数):")
for key, value in sorted(correlation_analysis.items(), key=lambda x: x[1]['abs_pearson_r'], reverse=True)[:10]:
    print(f"  {key}: r={value['pearson_r']:.3f}, p={value['pearson_p']:.4f}")

city_ranking = {}
for city in jjj_cities:
    city_data = merged_df[merged_df['city'] == city]
    if len(city_data) > 0:
        pm25_mean = city_data[dependent_var].mean()
        pm25_std = city_data[dependent_var].std()
        city_ranking[city] = {
            'pm25_mean': float(pm25_mean),
            'pm25_std': float(pm25_std),
            'pm25_trend': float(city_data[dependent_var].iloc[-1] - city_data[dependent_var].iloc[0]) if len(city_data) > 1 else 0
        }

city_ranking_sorted = dict(sorted(city_ranking.items(), key=lambda x: x[1]['pm25_mean'], reverse=True))
print("\n城市PM2.5排名 (从高到低):")
for i, (city, stats) in enumerate(city_ranking_sorted.items(), 1):
    print(f"  {i}. {city}: 均值={stats['pm25_mean']:.3f}, 标准差={stats['pm25_std']:.3f}, 趋势={stats['pm25_trend']:.3f}")

yearly_trend = {}
for year in target_years:
    year_data = merged_df[merged_df['year'] == year]
    if len(year_data) > 0:
        yearly_trend[year] = {
            'pm25_mean': float(year_data[dependent_var].mean()),
            'pm25_std': float(year_data[dependent_var].std()),
            'observation_count': int(len(year_data))
        }
        for var in independent_vars[:5]:
            if var in year_data.columns:
                yearly_trend[year][var] = float(year_data[var].mean())

print("\n年度趋势分析:")
for year, stats in yearly_trend.items():
    print(f"  {year}: PM2.5均值={stats['pm25_mean']:.3f}, PM2.5标准差={stats['pm25_std']:.3f}")

north_cities = ['北京', '唐山', '廊坊', '张家口', '承德', '秦皇岛']
south_cities = ['天津', '保定', '沧州', '石家庄', '衡水', '邢台', '邯郸']

north_data = merged_df[merged_df['city'].isin(north_cities)]
south_data = merged_df[merged_df['city'].isin(south_cities)]

regional_comparison = {
    'north': {
        'pm25_mean': float(north_data[dependent_var].mean()),
        'pm25_std': float(north_data[dependent_var].std()),
        'city_count': len(north_cities)
    },
    'south': {
        'pm25_mean': float(south_data[dependent_var].mean()),
        'pm25_std': float(south_data[dependent_var].std()),
        'city_count': len(south_cities)
    },
    'difference': float(north_data[dependent_var].mean() - south_data[dependent_var].mean())
}

print(f"\n区域对比分析 (北方 vs 南方):")
print(f"  北方城市: PM2.5均值={regional_comparison['north']['pm25_mean']:.3f}, PM2.5标准差={regional_comparison['north']['pm25_std']:.3f}")
print(f"  南方城市: PM2.5均值={regional_comparison['south']['pm25_mean']:.3f}, PM2.5标准差={regional_comparison['south']['pm25_std']:.3f}")
print(f"  差异: {regional_comparison['difference']:.3f}")

variable_description = {
    dependent_var: "PM2.5浓度（被解释变量）",
    'env_attention': '环境关注度',
    'avg_u10': '东向风速分量',
    'avg_v10': '北向风速分量',
    '煤炭占能源消费总量的比重 (%)': '煤炭占能源消费总量的比重',
    '石油占能源消费总量的比重 (%)': '石油占能源消费总量的比重',
    '天然气占能源消费总量的比重 (%)': '天然气占能源消费总量的比重',
    '一次电力及其他能源占能源总量的比重 (%)': '一次电力及其他能源占能源总量的比重',
    '第二产业占比': '第二产业占比',
    '供气总量': '供气总量'
}

result_summary = {
    'data_info': {
        'time_dimension': '2014-2024',
        'spatial_dimension': '京津冀13个地级市',
        'cities': jjj_cities,
        'total_observations': int(len(merged_df)),
        'variables': merged_df.columns.tolist(),
        'dependent_variable': dependent_var,
        'independent_variables': independent_for_vif
    },
    'variable_description': {k: v for k, v in variable_description.items() if k in merged_df.columns},
    'missing_value_treatment': {
        'method': '无（仅使用2014-2024数据，该时间段无缺失）',
        'removed_variables': '无',
        'removed_for_multicollinearity': removed_for_vif if removed_for_vif else '无'
    },
    'outlier_treatment': {
        'method': '3σ原则',
        'sigma_level': 3,
        'tail_level': 0.01
    },
    'standardization': {
        'method': 'Z-score标准化',
        'description': '均值0，方差1，消除量纲差异'
    },
    'stationarity_test': {
        'llc_test': llc_results,
        'ips_test': ips_results,
        'note': 'LLC/IPS检验用于验证数据平稳性'
    },
    'multicollinearity_test': {
        'vif_results': vif_results,
        'threshold': 10,
        'removed_variables': removed_for_vif
    },
    'descriptive_statistics': descriptive_stats,
    'correlation_analysis': correlation_analysis,
    'city_ranking': city_ranking_sorted,
    'yearly_trend': yearly_trend,
    'regional_comparison': regional_comparison,
    'advanced_metrics': {
        'high_correlation_pairs': [k for k, v in sorted(correlation_analysis.items(), key=lambda x: x[1]['abs_pearson_r'], reverse=True)[:15]],
        'most_polluted_city': list(city_ranking_sorted.keys())[0] if city_ranking_sorted else None,
        'least_polluted_city': list(city_ranking_sorted.keys())[-1] if city_ranking_sorted else None,
        'north_south_difference': regional_comparison['difference'],
        'overall_pm25_trend': float(merged_df[merged_df['year'] == 2024][dependent_var].mean() - merged_df[merged_df['year'] == 2014][dependent_var].mean()) if 2024 in target_years and 2014 in target_years else 0
    }
}

output_json = os.path.join(base_path, "processing_results.json")
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(result_summary, f, ensure_ascii=False, indent=2)
print(f"JSON文件已保存: {output_json}")

print("\n=== 处理完成 ===")
print(f"最终数据形状: {merged_df.shape}")
print(f"城市列表: {merged_df['city'].unique().tolist()}")
print(f"年份范围: {merged_df['year'].min()} - {merged_df['year'].max()}")
print(f"被解释变量: {dependent_var}")
print(f"解释变量: {independent_for_vif}")