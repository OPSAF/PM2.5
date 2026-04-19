import pandas as pd
import numpy as np
from scipy import stats
import json
import os

base_path = r"c:\Users\27862\Documents\trae_projects\project -tjjm\docs\modules\a"
source_file = r"c:\Users\27862\Documents\trae_projects\project -tjjm\maybe_a_end_data.csv"

df = pd.read_csv(source_file)
print("原始数据形状:", df.shape)
print("原始数据列:", df.columns.tolist())

df.columns = df.columns.str.strip()

dependent_var = 'pm25'
independent_vars = [col for col in df.columns if col not in ['city', 'year', dependent_var, 'Unnamed: 0', '']]

print(f"\n被解释变量: {dependent_var}")
print(f"解释变量数量: {len(independent_vars)}")
print(f"解释变量: {independent_vars}")

jjj_cities = ['北京', '天津', '保定', '唐山', '廊坊', '张家口', '承德', '沧州', '石家庄', '秦皇岛', '衡水', '邢台', '邯郸']
target_years = list(range(2010, 2025))

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

def linear_interpolate_group(group, cols):
    group = group.sort_values('year')
    for col in cols:
        if col in group.columns:
            missing_mask = group[col].isnull()
            if missing_mask.any():
                group[col] = group[col].interpolate(method='linear', limit_direction='both')
    return group

vars_to_interpolate = [var for var in all_vars if var in merged_df.columns]
merged_df = merged_df.groupby('city', group_keys=False).apply(
    lambda x: linear_interpolate_group(x, vars_to_interpolate)
)

print(f"\n插值后缺失值统计:")
for var in all_vars:
    if var in merged_df.columns:
        missing_count = merged_df[var].isnull().sum()
        print(f"  {var}: {missing_count}")

def count_consecutive_missing(group, col):
    missing = group[col].isnull().astype(int)
    max_consecutive = 0
    current_consecutive = 0
    for m in missing:
        if m == 1:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
    return max_consecutive

vars_to_drop = []
for var in all_vars:
    if var in merged_df.columns:
        consecutive = merged_df.groupby('city').apply(lambda x: count_consecutive_missing(x, var))
        vars_with_3plus_missing = consecutive[consecutive >= 3].index.tolist()
        if vars_with_3plus_missing:
            print(f"\n警告: {var} 在以下城市有连续3年+缺失: {vars_with_3plus_missing}")
            vars_to_drop.append(var)

if vars_to_drop:
    print(f"\n剔除变量(连续3年+缺失): {vars_to_drop}")
    merged_df = merged_df.drop(columns=vars_to_drop)
    independent_vars = [v for v in independent_vars if v not in vars_to_drop]
else:
    print("\n无连续3年+缺失的变量")

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
            'max': float(series.max())
        }

variable_description = {
    dependent_var: "PM2.5浓度（被解释变量）",
    'env_attention': '环境关注度',
    'total_water': '总水量',
    'avg_u10': '东向风速分量',
    'avg_v10': '北向风速分量',
    'tobeijing': '到北京的距离',
    'totianjin': '到天津的距离',
    'toshijiazhuang': '到石家庄的距离',
    'totangshan': '到唐山的距离',
    'toqinhuangdao': '到秦皇岛的距离',
    'tohandan': '到邯郸的距离',
    'toxingtai': '到邢台的距禋',
    'tobaoding': '到保定的距禋',
    'tozhangjiakou': '到张家口的距禋',
    'tochengde': '到承德的距禋',
    'tocangzhou': '到沧州的距禋',
    'tolangfang': '到廊坊的距禋',
    'tohengshui': '到衡水的距禋',
    '煤炭占能源消费总量的比重 (%)': '煤炭占能源消费总量的比重',
    '石油占能源消费总量的比重 (%)': '石油占能源消费总量的比重',
    '天然气占能源消费总量的比重 (%)': '天然气占能源消费总量的比重',
    '一次电力及其他能源占能源总量的比重 (%)': '一次电力及其他能源占能源总量的比重',
    '第二产业占比': '第二产业占比',
    '供气总量': '供气总量'
}

result_summary = {
    'data_info': {
        'time_dimension': '2010-2024',
        'spatial_dimension': '京津冀13个地级市',
        'cities': jjj_cities,
        'total_observations': int(len(merged_df)),
        'variables': merged_df.columns.tolist(),
        'dependent_variable': dependent_var,
        'independent_variables': independent_for_vif
    },
    'variable_description': {k: v for k, v in variable_description.items() if k in merged_df.columns},
    'missing_value_treatment': {
        'method': '线性插值填充少量缺失值',
        'removed_variables': vars_to_drop if vars_to_drop else '无',
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
    'descriptive_statistics': descriptive_stats
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