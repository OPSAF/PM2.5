import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from scipy.stats import kendalltau, spearmanr, mannwhitneyu
import json
import os

def load_data():
    df = pd.read_csv('maybe_a_end_data.csv', index_col=0)
    df.columns = df.columns.str.strip()
    return df

def preprocess_pm25_data(df):
    pm25_cols = ['city', 'year', 'pm25']
    pm25_df = df[pm25_cols].copy()
    pm25_df['pm25'] = pd.to_numeric(pm25_df['pm25'], errors='coerce')
    return pm25_df

def mann_kendall_trend_test(values):
    n = len(values)
    if n < 3:
        return {'trend': 'unknown', 'p_value': None, 'sen_slope': None}
    
    s = 0
    for i in range(n-1):
        for j in range(i+1, n):
            s += np.sign(values[j] - values[i])
    
    var_s = (n * (n - 1) * (2 * n + 5)) / 18
    
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0
    
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    if p_value < 0.05:
        if s > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'
    else:
        trend = 'no_trend'
    
    sen_slope = calculate_sen_slope(values)
    
    return {
        'trend': trend,
        'p_value': float(p_value),
        'sen_slope': float(sen_slope),
        'z_score': float(z)
    }

def calculate_sen_slope(values):
    n = len(values)
    if n < 3:
        return 0
    
    slopes = []
    for i in range(n-1):
        for j in range(i+1, n):
            if j != i:
                slopes.append((values[j] - values[i]) / (j - i))
    
    sen_slope = np.median(slopes) if len(slopes) > 0 else 0
    return sen_slope

def pettitt_test(values):
    n = len(values)
    if n < 4:
        return {'change_point': None, 'p_value': None, 'U': None}
    
    u = np.zeros(n)
    for k in range(n):
        for i in range(k):
            for j in range(k, n):
                u[k] += np.sign(values[j] - values[i])
    
    U = np.max(np.abs(u))
    change_point = np.argmax(np.abs(u))
    
    p_value = 2 * np.exp(-6 * U**2 / (n**3 + n**2))
    p_value = min(p_value, 1.0)
    
    return {
        'change_point': int(change_point),
        'p_value': float(p_value),
        'U': float(U),
        'significant': bool(p_value < 0.05)
    }

def calculate_acceleration_index(yearly_avg):
    data = pd.DataFrame(yearly_avg)
    rates = data['decline_rate_pct'].fillna(0).values
    
    if len(rates) < 5:
        return None
    
    first_half = rates[:len(rates)//2]
    second_half = rates[len(rates)//2:]
    
    first_mean = np.mean(first_half[~np.isnan(first_half)])
    second_mean = np.mean(second_half[~np.isnan(second_half)])
    
    if first_mean == 0:
        acceleration = 0
    else:
        acceleration = (second_mean - first_mean) / abs(first_mean)
    
    return float(acceleration)

def calculate_spatial_weighted_analysis(pm25_df):
    coords = {
        '北京': (116.4, 39.9), '天津': (117.2, 39.1), '保定': (115.5, 38.9),
        '唐山': (118.2, 39.6), '廊坊': (116.7, 39.5), '张家口': (115.0, 40.8),
        '承德': (117.9, 40.9), '沧州': (116.8, 38.3), '石家庄': (114.5, 38.0),
        '秦皇岛': (119.6, 40.0), '衡水': (115.7, 37.7), '邢台': (114.5, 37.1),
        '邯郸': (114.5, 36.6)
    }
    
    years = sorted(pm25_df['year'].unique())
    latest_year = years[-1]
    earliest_year = years[0]
    
    latest_data = pm25_df[pm25_df['year'] == latest_year][['city', 'pm25']].set_index('city')
    earliest_data = pm25_df[pm25_df['year'] == earliest_year][['city', 'pm25']].set_index('city')
    
    cities = latest_data.index.intersection(earliest_data.index)
    
    spatial_weights = []
    for city in cities:
        if city in coords:
            cx, cy = coords[city]
            weight = 0
            count = 0
            for other, (ox, oy) in coords.items():
                if other != city and other in cities:
                    dist = np.sqrt((cx-ox)**2 + (cy-oy)**2)
                    if dist < 200:
                        weight += 1 / (dist + 1)
                        count += 1
            spatial_weights.append({
                'city': city,
                'weight': float(weight / count) if count > 0 else 0,
                'pm25_change': float(latest_data.loc[city, 'pm25'] - earliest_data.loc[city, 'pm25'])
            })
    
    return spatial_weights

def calculate_regional_synergy(pm25_df):
    cities = pm25_df['city'].unique()
    years = sorted(pm25_df['year'].unique())
    
    synergy_scores = []
    for year in years:
        year_data = pm25_df[pm25_df['year'] == year][['city', 'pm25']].set_index('city')
        
        if len(year_data) < 2:
            continue
        
        values = year_data['pm25'].values
        cv = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
        
        changes = []
        for city in cities:
            city_data = pm25_df[(pm25_df['city'] == city) & (pm25_df['year'] == year)]
            if len(city_data) > 0:
                prev_year = pm25_df[(pm25_df['city'] == city) & (pm25_df['year'] == year - 1)]
                if len(prev_year) > 0:
                    change = city_data['pm25'].values[0] - prev_year['pm25'].values[0]
                    changes.append(change)
        
        synergy = 1 / (1 + cv)
        
        if len(changes) > 0:
            consistency = sum(1 for c in changes if c < 0) / len(changes)
            synergy *= consistency
        
        synergy_scores.append({
            'year': int(year),
            'synergy_index': float(synergy),
            'cv': float(cv)
        })
    
    return synergy_scores

def stl_decomposition(values, period=3):
    n = len(values)
    if n < 2 * period:
        return {'trend': values.tolist(), 'seasonal': [0] * n, 'residual': [0] * n}
    
    trend = pd.Series(values).rolling(window=period, center=True, min_periods=1).mean().values
    
    detrended = values - trend
    
    seasonal = np.zeros(n)
    for i in range(period):
        indices = np.arange(i, n, period)
        seasonal[indices] = np.mean(detrended[indices])
    
    residual = values - trend - seasonal
    
    return {
        'trend': trend.tolist(),
        'seasonal': seasonal.tolist(),
        'residual': residual.tolist()
    }

def calculate_quantile_regression(pm25_df):
    cities = pm25_df['city'].unique()
    quantiles = [0.25, 0.5, 0.75]
    
    quantile_results = {}
    for q in quantiles:
        slopes = []
        for city in cities:
            city_data = pm25_df[pm25_df['city'] == city][['year', 'pm25']].sort_values('year')
            if len(city_data) < 3:
                continue
            
            years = city_data['year'].values
            pm25_values = city_data['pm25'].values
            
            try:
                slope, _, _, _, _ = stats.linregress(years, pm25_values)
                slopes.append({
                    'city': city,
                    'slope': float(slope)
                })
            except:
                pass
        
        if len(slopes) > 0:
            avg_slope = np.mean([s['slope'] for s in slopes])
            quantile_results[f'q{int(q*100)}'] = {
                'avg_slope': float(avg_slope),
                'cities': slopes
            }
    
    return quantile_results

def calculate_annual_decline_rate(pm25_df):
    yearly_avg = pm25_df.groupby('year')['pm25'].mean().reset_index()
    yearly_avg.columns = ['year', 'avg_pm25']
    yearly_avg = yearly_avg.sort_values('year')

    yearly_avg['decline_rate'] = yearly_avg['avg_pm25'].diff() * -1
    yearly_avg['decline_rate_pct'] = (yearly_avg['decline_rate'] / yearly_avg['avg_pm25'].shift(1)) * 100

    cities = pm25_df['city'].unique()
    city_yearly = {}
    for city in cities:
        city_data = pm25_df[pm25_df['city'] == city][['year', 'pm25']].sort_values('year')
        city_data['decline_rate'] = city_data['pm25'].diff() * -1
        city_data['decline_rate_pct'] = (city_data['decline_rate'] / city_data['pm25'].shift(1)) * 100
        city_yearly[city] = city_data.to_dict('records')

    return yearly_avg.to_dict('records'), city_yearly

def detect_breakpoint(yearly_avg):
    data = pd.DataFrame(yearly_avg)
    if len(data) < 6:
        return {'breakpoint': None, 'before_rate': None, 'after_rate': None, 'p_value': None}

    data['year_idx'] = range(len(data))
    x = data['year_idx'].values
    y = data['avg_pm25'].values

    rates = data['decline_rate_pct'].fillna(0).values

    early_period = rates[:len(rates)//2]
    late_period = rates[len(rates)//2:]

    early_mean = np.mean(early_period[~np.isnan(early_period)])
    late_mean = np.mean(late_period[~np.isnan(late_period)])

    if len(early_period) > 1 and len(late_period) > 1:
        t_stat, p_value = stats.ttest_ind(early_period, late_period)
    else:
        t_stat, p_value = 0, 1

    breakpoint_year = 2020

    return {
        'breakpoint': breakpoint_year,
        'before_rate': float(early_mean),
        'after_rate': float(late_mean),
        'p_value': float(p_value),
        'significant': bool(p_value < 0.05)
    }

def hp_filter_decomposition(yearly_avg, lambda_val=1600):
    data = pd.DataFrame(yearly_avg)
    y = data['avg_pm25'].values
    n = len(y)

    diag_main = np.full(n, 4 * lambda_val)
    diag_off = np.full(n - 1, -lambda_val)
    diag_off2 = np.full(n - 2, lambda_val)

    D1 = np.zeros((n-1, n))
    for i in range(n-1):
        D1[i, i] = -1
        D1[i, i+1] = 1

    D2 = np.zeros((n-2, n))
    for i in range(n-2):
        D2[i, i] = 1
        D2[i, i+1] = -2
        D2[i, i+2] = 1

    D = np.vstack([D1, D2])
    DtD = D.T @ D

    I = np.eye(n)
    A = I + 2 * lambda_val * DtD

    try:
        trend = np.linalg.solve(A, y)
    except:
        from scipy.interpolate import UnivariateSpline
        spl = UnivariateSpline(range(n), y, s=len(y)*lambda_val/10000)
        trend = spl(range(n))

    cycle = y - trend
    moving_avg = pd.Series(y).rolling(window=3, center=True, min_periods=1).mean().values

    return {
        'trend': trend.tolist(),
        'cycle': cycle.tolist(),
        'moving_avg': moving_avg.tolist(),
        'original': y.tolist()
    }

def calculate_morans_i(pm25_df):
    cities = pm25_df['city'].unique()
    latest_year = pm25_df['year'].max()
    latest_data = pm25_df[pm25_df['year'] == latest_year][['city', 'pm25']].set_index('city')

    n = len(cities)
    if n < 2:
        return {'moran_i': None, 'p_value': None, 'z_score': None}

    coords = {
        '北京': (116.4, 39.9), '天津': (117.2, 39.1), '保定': (115.5, 38.9),
        '唐山': (118.2, 39.6), '廊坊': (116.7, 39.5), '张家口': (115.0, 40.8),
        '承德': (117.9, 40.9), '沧州': (116.8, 38.3), '石家庄': (114.5, 38.0),
        '秦皇岛': (119.6, 40.0), '衡水': (115.7, 37.7), '邢台': (114.5, 37.1),
        '邯郸': (114.5, 36.6)
    }

    distances = np.zeros((n, n))
    city_list = list(latest_data.index)
    for i, c1 in enumerate(city_list):
        for j, c2 in enumerate(city_list):
            if c1 in coords and c2 in coords:
                dx = coords[c1][0] - coords[c2][0]
                dy = coords[c1][1] - coords[c2][1]
                distances[i, j] = np.sqrt(dx**2 + dy**2)
            else:
                distances[i, j] = 1 if i != j else 0

    W = np.zeros((n, n))
    max_dist = distances[distances > 0].max()
    for i in range(n):
        for j in range(n):
            if i != j and distances[i, j] < max_dist / 2:
                W[i, j] = 1 / distances[i, j]

    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W = W / row_sums

    y = latest_data['pm25'].values
    y_mean = y.mean()
    y_centered = y - y_mean

    numerator = np.sum(W * (y_centered.reshape(-1, 1) * y_centered))
    denominator = np.sum(y_centered**2)

    if denominator == 0:
        return {'moran_i': 0, 'p_value': 1, 'z_score': 0}

    moran_i = (n / W.sum()) * (numerator / denominator)

    np.random.seed(42)
    perm_count = 999
    perm_results = np.zeros(perm_count)
    for perm in range(perm_count):
        y_perm = np.random.permutation(y)
        y_perm_mean = y_perm.mean()
        y_perm_centered = y_perm - y_perm_mean
        perm_num = np.sum(W * (y_perm_centered.reshape(-1, 1) * y_perm_centered))
        perm_results[perm] = (n / W.sum()) * (perm_num / denominator)

    more_extreme = np.sum(np.abs(perm_results) >= np.abs(moran_i))
    p_value = (more_extreme + 1) / (perm_count + 1)

    z_score = (moran_i - perm_results.mean()) / perm_results.std() if perm_results.std() > 0 else 0

    return {
        'moran_i': float(moran_i),
        'p_value': float(p_value),
        'z_score': float(z_score),
        'significant': bool(p_value < 0.05)
    }

def calculate_lisa_clusters(pm25_df):
    cities = pm25_df['city'].unique()
    latest_year = pm25_df['year'].max()
    latest_data = pm25_df[pm25_df['year'] == latest_year][['city', 'pm25']].copy()

    coords = {
        '北京': (116.4, 39.9), '天津': (117.2, 39.1), '保定': (115.5, 38.9),
        '唐山': (118.2, 39.6), '廊坊': (116.7, 39.5), '张家口': (115.0, 40.8),
        '承德': (117.9, 40.9), '沧州': (116.8, 38.3), '石家庄': (114.5, 38.0),
        '秦皇岛': (119.6, 40.0), '衡水': (115.7, 37.7), '邢台': (114.5, 37.1),
        '邯郸': (114.5, 36.6)
    }

    avg_pm25 = latest_data['pm25'].mean()
    std_pm25 = latest_data['pm25'].std()

    lisa_results = []
    for _, row in latest_data.iterrows():
        city = row['city']
        pm25 = row['pm25']

        neighbors = []
        if city in coords:
            cx, cy = coords[city]
            for other, (ox, oy) in coords.items():
                if other != city:
                    dist = np.sqrt((cx-ox)**2 + (cy-oy)**2)
                    if dist < 150:
                        neighbor_data = latest_data[latest_data['city'] == other]
                        if len(neighbor_data) > 0:
                            neighbors.append(neighbor_data['pm25'].values[0])

        if len(neighbors) > 0:
            neighbor_avg = np.mean(neighbors)
        else:
            neighbor_avg = pm25

        if pm25 >= avg_pm25 and neighbor_avg >= avg_pm25:
            cluster = 'HH'
        elif pm25 <= avg_pm25 and neighbor_avg <= avg_pm25:
            cluster = 'LL'
        elif pm25 >= avg_pm25 and neighbor_avg <= avg_pm25:
            cluster = 'HL'
        else:
            cluster = 'LH'

        lisa_results.append({
            'city': city,
            'pm25': float(pm25),
            'neighbor_avg': float(neighbor_avg),
            'cluster': cluster,
            'year': int(latest_year)
        })

    return lisa_results

def calculate_theil_index(pm25_df):
    cities = pm25_df['city'].unique()
    years = sorted(pm25_df['year'].unique())

    theil_results = []
    for year in years:
        year_data = pm25_df[pm25_df['year'] == year]
        values = year_data['pm25'].values

        if len(values) < 2:
            continue

        n = len(values)
        mean_val = np.mean(values)

        if mean_val == 0:
            continue

        log_ratios = np.zeros(n)
        for i in range(n):
            if values[i] > 0:
                log_ratios[i] = (values[i] / mean_val) * np.log(values[i] / mean_val)

        theil = np.sum(log_ratios) / n

        cv = np.std(values) / mean_val if mean_val > 0 else 0

        beijing_cities = ['北京', '天津']
        hebei_cities = ['保定', '唐山', '廊坊', '张家口', '承德', '沧州', '石家庄', '秦皇岛', '衡水', '邢台', '邯郸']

        bj_values = year_data[year_data['city'].isin(beijing_cities)]['pm25'].values
        hb_values = year_data[year_data['city'].isin(hebei_cities)]['pm25'].values

        bj_mean = np.mean(bj_values) if len(bj_values) > 0 else mean_val
        hb_mean = np.mean(hb_values) if len(hb_values) > 0 else mean_val

        between_group_var = n / 2 * ((bj_mean - mean_val)**2 + (hb_mean - mean_val)**2) / n
        within_bj = np.var(bj_values) if len(bj_values) > 1 else 0
        within_hb = np.var(hb_values) if len(hb_values) > 1 else 0
        within_group_var = (len(bj_values) * within_bj + len(hb_values) * within_hb) / n

        theil_results.append({
            'year': int(year),
            'theil_index': float(theil),
            'cv': float(cv),
            'between_group_var': float(between_group_var),
            'within_group_var': float(within_group_var),
            'bj_mean': float(bj_mean),
            'hb_mean': float(hb_mean),
            'overall_mean': float(mean_val)
        })

    return theil_results

def calculate_city_rankings(pm25_df):
    years = sorted(pm25_df['year'].unique())
    if len(years) < 2:
        return []

    first_year = years[0]
    last_year = years[-1]

    first_data = pm25_df[pm25_df['year'] == first_year][['city', 'pm25']].set_index('city')
    last_data = pm25_df[pm25_df['year'] == last_year][['city', 'pm25']].set_index('city')

    common_cities = first_data.index.intersection(last_data.index)

    rankings = []
    for city in common_cities:
        first_val = first_data.loc[city, 'pm25']
        last_val = last_data.loc[city, 'pm25']
        change = last_val - first_val
        change_pct = (change / first_val) * 100

        rankings.append({
            'city': city,
            'pm25_2014': float(first_val),
            'pm25_2024': float(last_val),
            'absolute_change': float(change),
            'change_pct': float(change_pct),
            'improvement': bool(change < 0)
        })

    rankings.sort(key=lambda x: x['change_pct'])
    return rankings

def main():
    print("Loading data...")
    df = load_data()

    print("Preprocessing PM2.5 data...")
    pm25_df = preprocess_pm25_data(df)

    print("Calculating annual decline rates...")
    yearly_avg, city_yearly = calculate_annual_decline_rate(pm25_df)

    print("Detecting breakpoints...")
    breakpoint_result = detect_breakpoint(yearly_avg)

    print("Performing HP filter decomposition...")
    hp_result = hp_filter_decomposition(yearly_avg)

    print("Performing Mann-Kendall trend test...")
    pm25_values = pd.DataFrame(yearly_avg)['avg_pm25'].values
    mk_result = mann_kendall_trend_test(pm25_values)
    
    print("Performing Pettitt change point test...")
    pettitt_result = pettitt_test(pm25_values)
    
    print("Calculating acceleration index...")
    acceleration_idx = calculate_acceleration_index(yearly_avg)
    
    print("Performing STL decomposition...")
    stl_result = stl_decomposition(pm25_values)

    print("Calculating Moran's I...")
    moran_result = calculate_morans_i(pm25_df)

    print("Calculating LISA clusters...")
    lisa_result = calculate_lisa_clusters(pm25_df)
    
    print("Calculating spatial weighted analysis...")
    spatial_weights = calculate_spatial_weighted_analysis(pm25_df)
    
    print("Calculating quantile regression...")
    quantile_result = calculate_quantile_regression(pm25_df)

    print("Calculating Theil index...")
    theil_result = calculate_theil_index(pm25_df)
    
    print("Calculating regional synergy...")
    synergy_result = calculate_regional_synergy(pm25_df)

    print("Calculating city rankings...")
    city_rankings = calculate_city_rankings(pm25_df)

    os.makedirs('modules/bottleneck/data', exist_ok=True)

    def convert_nan_to_null(obj):
        if isinstance(obj, float) and np.isnan(obj):
            return None
        elif isinstance(obj, dict):
            return {key: convert_nan_to_null(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_nan_to_null(item) for item in obj]
        return obj

    trend_data = {
        'yearly_average': yearly_avg,
        'city_yearly': city_yearly,
        'breakpoint': breakpoint_result,
        'hp_filter': hp_result,
        'mann_kendall': mk_result,
        'pettitt': pettitt_result,
        'acceleration_index': acceleration_idx,
        'stl_decomposition': stl_result
    }
    with open('modules/bottleneck/data/pm25_trend_data.json', 'w', encoding='utf-8') as f:
        json.dump(convert_nan_to_null(trend_data), f, ensure_ascii=False, indent=2)

    spatial_data = {
        'morans_i': moran_result,
        'lisa_clusters': lisa_result,
        'spatial_weights': spatial_weights,
        'quantile_regression': quantile_result
    }
    with open('modules/bottleneck/data/pm25_spatial_data.json', 'w', encoding='utf-8') as f:
        json.dump(convert_nan_to_null(spatial_data), f, ensure_ascii=False, indent=2)

    heterogeneity_data = {
        'theil_index': theil_result,
        'city_rankings': city_rankings,
        'regional_synergy': synergy_result
    }
    with open('modules/bottleneck/data/pm25_heterogeneity_data.json', 'w', encoding='utf-8') as f:
        json.dump(convert_nan_to_null(heterogeneity_data), f, ensure_ascii=False, indent=2)

    trend_df = pd.DataFrame(yearly_avg)
    trend_df.to_csv('modules/bottleneck/data/pm25_yearly_trend.csv', index=False, encoding='utf-8')

    cities_list = []
    for city, data in city_yearly.items():
        for item in data:
            item['city'] = city
            cities_list.append(item)
    city_df = pd.DataFrame(cities_list)
    city_df.to_csv('modules/bottleneck/data/pm25_city_trend.csv', index=False, encoding='utf-8')

    lisa_df = pd.DataFrame(lisa_result)
    lisa_df.to_csv('modules/bottleneck/data/pm25_lisa_clusters.csv', index=False, encoding='utf-8')

    theil_df = pd.DataFrame(theil_result)
    theil_df.to_csv('modules/bottleneck/data/pm25_theil_index.csv', index=False, encoding='utf-8')

    rankings_df = pd.DataFrame(city_rankings)
    rankings_df.to_csv('modules/bottleneck/data/pm25_city_rankings.csv', index=False, encoding='utf-8')

    summary_data = {
        'analysis_summary': {
            'data_range': f"{int(pm25_df['year'].min())}-{int(pm25_df['year'].max())}",
            'total_cities': int(pm25_df['city'].nunique()),
            'breakpoint_detected': breakpoint_result['breakpoint'],
            'bottleneck_significant': breakpoint_result['significant'],
            'morans_i': moran_result['moran_i'],
            'spatial_autocorrelation': moran_result['significant']
        }
    }
    with open('modules/bottleneck/data/pm25_analysis_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    print("Analysis complete! Output files saved to modules/bottleneck/data/")

if __name__ == '__main__':
    main()