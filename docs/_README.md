# PM2.5 下降瓶颈分析模块

## 模块介绍

本模块专注于京津冀地区PM2.5浓度下降瓶颈的存在性验证分析，通过统计方法严谨证明"京津冀PM2.5浓度下降进入瓶颈期"的现象，为相关研究提供数据支持。

## 核心功能

### 1. 时间趋势断点分析
- **方法**：断点回归（Breakpoint Regression） + HP 滤波分解
- **功能**：
  - 绘制2010-2024年京津冀整体、分省市PM2.5浓度时间趋势图
  - 计算逐年下降速率
  - 识别下降速率的统计显著突变节点
  - 验证"2013-2020年快速下降、2021-2024年下降速率显著放缓"的瓶颈特征
  - 使用HP滤波分解趋势项与波动项，量化长期减排动能的衰减幅度

### 2. 空间格局演变分析
- **方法**：全局莫兰指数（Global Moran's I） + LISA 集聚图
- **功能**：
  - 计算全局莫兰指数，验证PM2.5浓度的空间自相关性
  - 证明区域污染溢出效应的存在
  - 识别京津冀"高-高污染集聚"的瓶颈区域、污染传输核心通道

### 3. 区域异质性分析
- **方法**：泰尔指数分解 + 变异系数
- **功能**：
  - 分解京津冀区域内PM2.5浓度差异的来源
  - 识别哪些城市/区域拖了整体减排的后腿
  - 定位区域协同瓶颈

## 文件结构

```
project/
├── modules/
│   ├── pm25_bottleneck_analysis.py      # 主分析脚本
│   ├── pm25_bottleneck_visualization.html  # 可视化页面
│   ├── pm25_analysis_summary.json       # 分析摘要数据
│   ├── pm25_trend_data.json             # 趋势分析数据
│   ├── pm25_spatial_data.json           # 空间分析数据
│   ├── pm25_heterogeneity_data.json     # 异质性分析数据
│   ├── pm25_yearly_trend.csv            # 年度趋势数据
│   ├── pm25_city_trend.csv              # 城市趋势数据
│   ├── pm25_lisa_clusters.csv           # LISA聚类数据
│   ├── pm25_theil_index.csv             # 泰尔指数数据
│   └── pm25_city_rankings.csv           # 城市排名数据
└── start_server.bat                     # 启动服务器脚本
```

## 如何运行

### 方法一：使用启动脚本（推荐）
1. 双击 `start_server.bat` 文件
2. 系统会自动：
   - 启动本地web服务器
   - 打开浏览器访问可视化页面
   - 显示运行状态
3. 查看完分析结果后，按任意键关闭服务器

### 方法二：手动运行
1. **运行数据分析**：
   ```bash
   python modules/pm25_bottleneck_analysis.py
   ```
2. **启动本地服务器**：
   ```bash
   python -m http.server 8000
   ```
3. **访问可视化页面**：
   在浏览器中打开 `http://localhost:8000/modules/pm25_bottleneck_visualization.html`

## 数据说明

### 输入数据
- 模块使用 `maybe_a_end_data` 中的PM2.5数据
- 数据时间范围：2014-2024年
- 覆盖京津冀地区主要城市

### 输出数据
1. **JSON数据文件**：
   - `pm25_analysis_summary.json`：分析摘要信息
   - `pm25_trend_data.json`：趋势分析详细数据
   - `pm25_spatial_data.json`：空间分析数据
   - `pm25_heterogeneity_data.json`：异质性分析数据

2. **CSV数据文件**：
   - `pm25_yearly_trend.csv`：年度PM2.5浓度趋势
   - `pm25_city_trend.csv`：各城市PM2.5浓度趋势
   - `pm25_lisa_clusters.csv`：LISA聚类结果
   - `pm25_theil_index.csv`：泰尔指数分解结果
   - `pm25_city_rankings.csv`：城市改善排名

## 可视化说明

### 主要图表
1. **趋势分析**：
   - 京津冀整体PM2.5浓度时间趋势图
   - HP滤波分解图
   - 年度下降速率图
   - 主要城市趋势对比图

2. **空间分析**：
   - LISA集聚图
   - 城市PM2.5浓度空间分布

3. **异质性分析**：
   - 泰尔指数时间变化
   - 区域对比图
   - 城市排名图
   - 变异系数分解图

4. **数据表格**：
   - LISA聚类结果表
   - 城市排名表

5. **分析结论**：
   - 瓶颈期确认
   - 空间溢出效应
   - 区域失衡分析
   - 关键发现

## 技术实现

### 数据分析
- **语言**：Python
- **核心库**：pandas, numpy, scipy, statsmodels
- **空间分析**：使用统计方法计算莫兰指数和LISA聚类

### 可视化
- **前端框架**：HTML5 + JavaScript
- **图表库**：Chart.js
- **样式库**：Tailwind CSS
- **图标库**：Font Awesome

## 常见问题解决

### 1. 数据加载失败
- **原因**：浏览器安全限制，直接打开HTML文件时fetch请求被阻止
- **解决方案**：使用 `start_server.bat` 启动本地web服务器访问

### 2. 服务器启动失败
- **检查**：确保Python已正确安装
- **替代方案**：使用其他本地服务器软件（如XAMPP、WAMP等）

### 3. 图表显示异常
- **检查**：浏览器控制台是否有错误信息
- **解决方案**：刷新页面或清除浏览器缓存

## 系统要求

- **操作系统**：Windows 7+ / macOS / Linux
- **Python**：3.6+
- **浏览器**：Chrome、Firefox、Edge等现代浏览器

## 注意事项

1. 本模块仅用于数据分析和可视化，不涉及实时数据采集
2. 分析结果基于历史数据，仅供参考
3. 若数据文件结构发生变化，可能需要调整脚本

## 联系方式

如有问题或建议，请联系相关技术人员。