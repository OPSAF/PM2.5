import pandas as pd
import numpy as np
from scipy import stats
from scipy.interpolate import griddata, Rbf
from scipy.spatial.distance import cdist
import json
import os
import math

def load_data():
    df = pd.read_csv('maybe_a_end_data.csv', index_col=0)
    df.columns = df.columns.str.strip()
    return df

def preprocess_pm25_data(df):
    pm25_cols = ['city', 'year', 'pm25']
    pm25_df = df[pm25_cols].copy()
    pm25_df['pm25'] = pd.to_numeric(pm25_df['pm25'], errors='coerce')
    return pm25_df

ORIGIN_LON = 115.5
ORIGIN_LAT = 38.8

def lonlat_to_cartesian(lon, lat, origin_lon=ORIGIN_LON, origin_lat=ORIGIN_LAT):
    R = 6371.0
    lon_rad = math.radians(lon)
    lat_rad = math.radians(lat)
    origin_lon_rad = math.radians(origin_lon)
    origin_lat_rad = math.radians(origin_lat)
    x = R * (lon_rad - origin_lon_rad) * math.cos(origin_lat_rad)
    y = R * (lat_rad - origin_lat_rad)
    return x, y

def get_city_coordinates():
    coords = {
        '北京': (116.4, 39.9),
        '天津': (117.2, 39.1),
        '保定': (115.5, 38.9),
        '唐山': (118.2, 39.6),
        '廊坊': (116.7, 39.5),
        '张家口': (115.0, 40.8),
        '承德': (117.9, 40.9),
        '沧州': (116.8, 38.3),
        '石家庄': (114.5, 38.0),
        '秦皇岛': (119.6, 40.0),
        '衡水': (115.7, 37.7),
        '邢台': (114.5, 37.1),
        '邯郸': (114.5, 36.6)
    }
    return coords

def calculate_city_cartesian_coords():
    coords = get_city_coordinates()
    cartesian_coords = {}
    min_x, min_y = float('inf'), float('inf')
    for city, (lon, lat) in coords.items():
        x, y = lonlat_to_cartesian(lon, lat)
        cartesian_coords[city] = {'x': x, 'y': y, 'lon': lon, 'lat': lat}
        min_x = min(min_x, x)
        min_y = min(min_y, y)
    for city in cartesian_coords:
        cartesian_coords[city]['x'] -= min_x
        cartesian_coords[city]['y'] -= min_y
    return cartesian_coords

def create_city_coordinates_csv():
    cartesian_coords = calculate_city_cartesian_coords()
    records = []
    for city, data in cartesian_coords.items():
        records.append({
            'city': city,
            'lon': data['lon'],
            'lat': data['lat'],
            'x_km': round(data['x'], 3),
            'y_km': round(data['y'], 3)
        })
    df = pd.DataFrame(records)
    df.to_csv('modules/map/data/city_coordinates.csv', index=False, encoding='utf-8')
    return df

def merge_coordinates_with_pm25(pm25_df):
    cartesian_coords = calculate_city_cartesian_coords()
    records = []
    for _, row in pm25_df.iterrows():
        city = row['city']
        if city in cartesian_coords:
            records.append({
                'city': city,
                'year': row['year'],
                'pm25': row['pm25'],
                'x': cartesian_coords[city]['x'],
                'y': cartesian_coords[city]['y']
            })
    return pd.DataFrame(records)

def generate_grid_data(x, y, z, grid_size=50):
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_margin = x_range * 0.1
    y_margin = y_range * 0.1
    xi = np.linspace(x_min - x_margin, x_max + x_margin, grid_size)
    yi = np.linspace(y_min - y_margin, y_max + y_margin, grid_size)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x, y), z, (Xi, Yi), method='cubic')
    return Xi, Yi, Zi

def kriging_interpolation(x, y, z, Xi, Yi, lag=50):
    variogram = []
    max_dist = np.sqrt((x.max() - x.min())**2 + (y.max() - y.min())**2) / 2
    for h in np.linspace(0, max_dist, 20):
        pairs = []
        for i in range(len(x)):
            for j in range(i+1, len(x)):
                dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if abs(dist - h) < lag:
                    pairs.append((z[i] - z[j])**2)
        if len(pairs) > 0:
            variogram.append((h, np.mean(pairs)))
    if len(variogram) < 3:
        return griddata((x, y), z, (Xi, Yi), method='linear')
    distances = np.array([v[0] for v in variogram])
    semivariances = np.array([v[1] for v in variogram])
    try:
        slope, intercept, _, _, _ = stats.linregress(distances, semivariances)
        if slope <= 0:
            return griddata((x, y), z, (Xi, Yi), method='linear')
        sill = np.var(z)
        range_param = sill / slope if slope > 0 else max_dist
        nugget = 0
        Zi = np.zeros_like(Xi)
        for i in range(Xi.shape[0]):
            for j in range(Xi.shape[1]):
                distances_to_points = np.sqrt((x - Xi[i,j])**2 + (y - Yi[i,j])**2)
                weights = np.zeros(len(x))
                for k in range(len(x)):
                    d = distances_to_points[k]
                    if d == 0:
                        weights[k] = 1
                    else:
                        h = d / range_param
                        weights[k] = 1 - (3*h/2)**2 + (h/2)**3 if h <= 1 else 0
                if weights.sum() > 0:
                    weights /= weights.sum()
                else:
                    weights = np.ones(len(x)) / len(x)
                Zi[i,j] = np.sum(weights * z)
        return Zi
    except:
        return griddata((x, y), z, (Xi, Yi), method='linear')

def generate_heatmap_json(pm25_df, year):
    year_data = pm25_df[pm25_df['year'] == year]
    if len(year_data) < 3:
        return None
    x = year_data['x'].values
    y = year_data['y'].values
    z = year_data['pm25'].values
    Xi, Yi, Zi = generate_grid_data(x, y, z, grid_size=60)
    Zi_filled = kriging_interpolation(x, y, z, Xi, Yi)
    heatmap_data = []
    for i in range(Xi.shape[0]):
        for j in range(Xi.shape[1]):
            if not np.isnan(Zi_filled[i,j]):
                heatmap_data.append([float(Xi[i,j]), float(Yi[i,j]), float(Zi_filled[i,j])])
    return heatmap_data

def coordinate_regression_model(pm25_df):
    cities = pm25_df['city'].unique()
    years = sorted(pm25_df['year'].unique())
    results = []
    for year in years:
        year_data = pm25_df[pm25_df['year'] == year]
        if len(year_data) < 3:
            continue
        x = year_data['x'].values
        y = year_data['y'].values
        z = year_data['pm25'].values
        X_design = np.column_stack([np.ones(len(x)), x, y, x*y])
        try:
            coeffs = np.linalg.lstsq(X_design, z, rcond=None)[0]
            y_pred = X_design @ coeffs
            ss_res = np.sum((z - y_pred)**2)
            ss_tot = np.sum((z - np.mean(z))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            n = len(z)
            p = X_design.shape[1]
            if n > p:
                mse = ss_res / (n - p)
                var_coeffs = mse * np.linalg.inv(X_design.T @ X_design)
                se_coeffs = np.sqrt(np.diag(var_coeffs))
                t_stats = coeffs / se_coeffs
                p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n-p))
            else:
                se_coeffs = np.zeros_like(coeffs)
                t_stats = np.zeros_like(coeffs)
                p_values = np.ones_like(coeffs)
            results.append({
                'year': int(year),
                'alpha0': float(coeffs[0]),
                'beta1': float(coeffs[1]),
                'beta2': float(coeffs[2]),
                'beta3': float(coeffs[3]),
                'r_squared': float(r_squared),
                'n_cities': int(n),
                't_alpha0': float(t_stats[0]),
                't_beta1': float(t_stats[1]),
                't_beta2': float(t_stats[2]),
                't_beta3': float(t_stats[3]),
                'p_alpha0': float(p_values[0]),
                'p_beta1': float(p_values[1]),
                'p_beta2': float(p_values[2]),
                'p_beta3': float(p_values[3])
            })
        except Exception as e:
            continue
    return results

def calculate_spatial_autocorrelation(pm25_df):
    cities = pm25_df['city'].unique()
    years = sorted(pm25_df['year'].unique())
    results = []
    for year in years:
        year_data = pm25_df[pm25_df['year'] == year].set_index('city')
        common_cities = [c for c in cities if c in year_data.index]
        n = len(common_cities)
        if n < 2:
            continue
        coords = calculate_city_cartesian_coords()
        distances = np.zeros((n, n))
        for i, c1 in enumerate(common_cities):
            for j, c2 in enumerate(common_cities):
                if c1 in coords and c2 in coords:
                    dx = coords[c1]['x'] - coords[c2]['x']
                    dy = coords[c1]['y'] - coords[c2]['y']
                    distances[i, j] = np.sqrt(dx**2 + dy**2)
                else:
                    distances[i, j] = 1 if i != j else 0
        max_dist = distances[distances > 0].max()
        W = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j and distances[i, j] < max_dist / 2:
                    W[i, j] = 1 / distances[i, j]
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        W = W / row_sums
        pm25_values = np.array([year_data.loc[c, 'pm25'] for c in common_cities])
        y_mean = pm25_values.mean()
        y_centered = pm25_values - y_mean
        numerator = np.sum(W * (y_centered.reshape(-1, 1) * y_centered))
        denominator = np.sum(y_centered**2)
        if denominator == 0:
            moran_i = 0
        else:
            moran_i = (n / W.sum()) * (numerator / denominator)
        results.append({
            'year': int(year),
            'moran_i': float(moran_i)
        })
    return results

def export_html_data(pm25_df):
    cartesian_coords = calculate_city_cartesian_coords()
    years = sorted(pm25_df['year'].unique())
    city_list = [{'city': city, 'x': data['x'], 'y': data['y']} for city, data in cartesian_coords.items()]
    city_pm25_data = {}
    for city in cartesian_coords.keys():
        city_pm25_data[city] = []
        for year in years:
            city_year_data = pm25_df[(pm25_df['city'] == city) & (pm25_df['year'] == year)]
            if len(city_year_data) > 0:
                city_pm25_data[city].append({
                    'year': int(year),
                    'pm25': float(city_year_data['pm25'].values[0])
                })
    base_data = {
        'cities': city_list,
        'city_pm25': city_pm25_data,
        'years': [int(y) for y in years],
        'year_range': [int(min(years)), int(max(years))]
    }
    with open('modules/map/data/base_data.json', 'w', encoding='utf-8') as f:
        json.dump(base_data, f, ensure_ascii=False, indent=2)
    for year in years:
        year_data = pm25_df[pm25_df['year'] == year]
        if len(year_data) >= 3:
            heatmap = generate_heatmap_json(pm25_df, year)
            if heatmap:
                with open(f'modules/map/data/heatmap_{int(year)}.json', 'w', encoding='utf-8') as f:
                    json.dump(heatmap, f, ensure_ascii=False)
    scatter_data = []
    for _, row in pm25_df.iterrows():
        city = row['city']
        if city in cartesian_coords:
            scatter_data.append({
                'city': city,
                'year': int(row['year']),
                'x': float(cartesian_coords[city]['x']),
                'y': float(cartesian_coords[city]['y']),
                'pm25': float(row['pm25'])
            })
    with open('modules/map/data/scatter_data.json', 'w', encoding='utf-8') as f:
        json.dump(scatter_data, f, ensure_ascii=False, indent=2)
    print(f"Exported base data and {len(years)} yearly heatmaps")

def main():
    print("Loading data...")
    df = load_data()
    print("Preprocessing PM2.5 data...")
    pm25_df = preprocess_pm25_data(df)
    print("Calculating city Cartesian coordinates...")
    coords_df = create_city_coordinates_csv()
    print(f"Created coordinates for {len(coords_df)} cities")
    print("Merging coordinates with PM2.5 data...")
    pm25_with_coords = merge_coordinates_with_pm25(pm25_df)
    print("Exporting HTML visualization data...")
    export_html_data(pm25_with_coords)
    print("Running coordinate-PM25 regression model...")
    regression_results = coordinate_regression_model(pm25_with_coords)
    if regression_results:
        reg_df = pd.DataFrame(regression_results)
        reg_df.to_csv('modules/map/data/regression_results.csv', index=False, encoding='utf-8')
        print(f"Regression results: {len(regression_results)} years analyzed")
    print("Calculating spatial autocorrelation (Moran's I)...")
    moran_results = calculate_spatial_autocorrelation(pm25_with_coords)
    if moran_results:
        moran_df = pd.DataFrame(moran_results)
        moran_df.to_csv('modules/map/data/morans_i_timeseries.csv', index=False, encoding='utf-8')
        print(f"Moran's I calculated for {len(moran_results)} years")
    print("Spatial analysis complete!")

if __name__ == '__main__':
    main()