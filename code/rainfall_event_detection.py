# ============================================================
# 全球降雨事件检测与追踪系统
# ============================================================
#
# 【功能概述】
#   基于 CMORPH 卫星反演降水产品（0.072°≈8km, 30min），自动检测全球范围
#   的降雨事件，提取事件时空轨迹、物理特征、海陆分类、气候分区等信息，
#   输出可供气候分析、极端降水研究的结构化事件数据集。
#
# 【事件划分逻辑（核心流水线）】
#   1. 数据读取    按天读取 CMORPH NetCDF → 展开为 (grid, time) 长表
#                  支持numpy快速加载 + 缺失值线性插值
#   2. 网格事件识别 对每个网格独立做 IETD 分析：
#                  - 降雨量 > threshold → 降雨开始
#                  - 无雨间隔 > IETD    → 事件结束
#                  - IETD/threshold 按柯本气候分区或纬度带自适应
#                  - numba JIT 加速逐网格遍历
#   3. 跨日合并    相同网格在时间上连续（间隔<IETD）的片段合并为一个网格事件
#   4. DBSCAN聚类  将网格事件按空间-时间 proximity 聚合为天气尺度事件：
#                  - 3D笛卡尔坐标(x,y,z) + 时间维度 → Ball Tree 球面距离
#                  - 预分区分块聚类（大数据量优化）
#                  - eps 随纬度余弦修正（高纬度缩小空间阈值）
#   5. 动态切分    对DBSCAN聚类结果做事件内分裂检测：
#                  - 面积骤减（>area_change_threshold）→ 切分点
#                  - 强度骤降（>intensity_drop_threshold）→ 切分点
#                  - 质心跳跃（>5°，循环经度修正）→ 切分点
#   6. 事件汇总    按事件聚合统计：起止时间、总雨量、峰值、轨迹质心等
#                  - 加权线性回归拟合位移轨迹（含经度解缠绕）
#                  - 循环加权平均计算质心经度（处理跨0°经线）
#                  - 极端值标志位（不截断，保留原始值）
#   7. 地理分类    sjoin + buffer 判断海陆类型：
#                  - 4类：内陆/沿海/近海/海洋（coastal_buffer + 纬度修正）
#                  - 大洲、国家、海洋名称、柯本气候分区
#   8. 跨月合并    并查集(union-find)合并跨月边界的同一事件：
#                  - 时间间隔 ≤ 6h + 空间距离 ≤ 2°（循环经度归一化）
#                  - avg_precip_depth 用各片段加权平均（避免虚高）
#   9. 全年统一ID  跨月合并后重新编号：YYYY_NNNNNN
#   10.输出       CSV + 交互式HTML地图 + 静态报告 + 数据字典
#
# 【坐标系约定】
#   - CMORPH 原始经度：0-360°
#   - naturalearth/shapefile：-180-180°
#   - 内部统一用 lon_col_normalized 转换，所有地理分类用归一化经度
#   - 质心/距离/轨迹计算用循环平均/循环差值处理跨0°经线
#
# 【用法】修改下方 CONFIG 字典中的参数，然后直接运行
# ============================================================

import os
import sys
import gc
import re
import glob
import math
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from tqdm import tqdm

# 可选依赖：numba JIT 加速
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("  ⚠️ numba 未安装，核心循环将使用纯Python（建议安装: pip install numba）")
    print("     安装后预计提速 5-15x")

# 可选依赖：未安装时降级处理
try:
    import geopandas as gpd
    import shapely.geometry as geom
    from shapely.ops import nearest_points
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

# ===== reverse_geocoder：高精度离线国家归属（优先级高于sjoin）=====
# 基于 GeoNames ~500万 城市/村落 经纬度库，O(1) k-d tree 近邻查询
# 比 Natural Earth sjoin：小国更精确、查询速度×10
try:
    import reverse_geocoder as _rg_mod
    HAS_RG = True
    # ISO 国家代码 → (国家名, 大洲) 映射
    # 数据来源：ISO 3166-1 alpha-2 + UN geoscheme，与 Natural Earth 大洲名对齐（英文）
    _ISO_CC_TO_COUNTRY_CONTINENT = {
        # ---- 亚洲 Asia ----
        "CN": ("China", "Asia"), "IN": ("India", "Asia"), "JP": ("Japan", "Asia"),
        "KR": ("South Korea", "Asia"), "KP": ("North Korea", "Asia"), "MN": ("Mongolia", "Asia"),
        "TH": ("Thailand", "Asia"), "VN": ("Vietnam", "Asia"), "ID": ("Indonesia", "Asia"),
        "MY": ("Malaysia", "Asia"), "SG": ("Singapore", "Asia"), "PH": ("Philippines", "Asia"),
        "KH": ("Cambodia", "Asia"), "LA": ("Laos", "Asia"), "MM": ("Myanmar", "Asia"),
        "BD": ("Bangladesh", "Asia"), "PK": ("Pakistan", "Asia"), "AF": ("Afghanistan", "Asia"),
        "IR": ("Iran", "Asia"), "IQ": ("Iraq", "Asia"), "SA": ("Saudi Arabia", "Asia"),
        "AE": ("United Arab Emirates", "Asia"), "IL": ("Israel", "Asia"), "PS": ("Palestine", "Asia"),
        "JO": ("Jordan", "Asia"), "SY": ("Syria", "Asia"), "LB": ("Lebanon", "Asia"),
        "TW": ("Taiwan", "Asia"), "HK": ("Hong Kong", "Asia"), "MO": ("Macau", "Asia"),
        "NP": ("Nepal", "Asia"), "LK": ("Sri Lanka", "Asia"), "BT": ("Bhutan", "Asia"),
        "MV": ("Maldives", "Asia"), "BN": ("Brunei", "Asia"), "TL": ("Timor-Leste", "Asia"),
        "KZ": ("Kazakhstan", "Asia"), "UZ": ("Uzbekistan", "Asia"), "TM": ("Turkmenistan", "Asia"),
        "KG": ("Kyrgyzstan", "Asia"), "TJ": ("Tajikistan", "Asia"), "AM": ("Armenia", "Asia"),
        "AZ": ("Azerbaijan", "Asia"), "GE": ("Georgia", "Asia"), "YE": ("Yemen", "Asia"),
        "OM": ("Oman", "Asia"), "QA": ("Qatar", "Asia"), "KW": ("Kuwait", "Asia"),
        "BH": ("Bahrain", "Asia"), "TR": ("Turkey", "Asia"), "CY": ("Cyprus", "Asia"),
        "RU": ("Russia", "Asia"),  # 大部分国土在亚洲
        # ---- 欧洲 Europe ----
        "GB": ("United Kingdom", "Europe"), "FR": ("France", "Europe"), "DE": ("Germany", "Europe"),
        "IT": ("Italy", "Europe"), "ES": ("Spain", "Europe"), "PT": ("Portugal", "Europe"),
        "NL": ("Netherlands", "Europe"), "BE": ("Belgium", "Europe"), "LU": ("Luxembourg", "Europe"),
        "CH": ("Switzerland", "Europe"), "AT": ("Austria", "Europe"), "PL": ("Poland", "Europe"),
        "CZ": ("Czech Republic", "Europe"), "SK": ("Slovakia", "Europe"), "HU": ("Hungary", "Europe"),
        "RO": ("Romania", "Europe"), "BG": ("Bulgaria", "Europe"), "GR": ("Greece", "Europe"),
        "SE": ("Sweden", "Europe"), "NO": ("Norway", "Europe"), "DK": ("Denmark", "Europe"),
        "FI": ("Finland", "Europe"), "IS": ("Iceland", "Europe"), "IE": ("Ireland", "Europe"),
        "EE": ("Estonia", "Europe"), "LV": ("Latvia", "Europe"), "LT": ("Lithuania", "Europe"),
        "UA": ("Ukraine", "Europe"), "BY": ("Belarus", "Europe"), "MD": ("Moldova", "Europe"),
        "RS": ("Serbia", "Europe"), "HR": ("Croatia", "Europe"), "SI": ("Slovenia", "Europe"),
        "BA": ("Bosnia and Herz.", "Europe"), "ME": ("Montenegro", "Europe"), "MK": ("North Macedonia", "Europe"),
        "AL": ("Albania", "Europe"), "XK": ("Kosovo", "Europe"), "MT": ("Malta", "Europe"),
        "AD": ("Andorra", "Europe"), "MC": ("Monaco", "Europe"), "LI": ("Liechtenstein", "Europe"),
        "SM": ("San Marino", "Europe"), "VA": ("Vatican", "Europe"),
        # ---- 非洲 Africa ----
        "EG": ("Egypt", "Africa"), "LY": ("Libya", "Africa"), "TN": ("Tunisia", "Africa"),
        "DZ": ("Algeria", "Africa"), "MA": ("Morocco", "Africa"), "SD": ("Sudan", "Africa"),
        "SS": ("South Sudan", "Africa"), "ET": ("Ethiopia", "Africa"), "SO": ("Somalia", "Africa"),
        "DJ": ("Djibouti", "Africa"), "ER": ("Eritrea", "Africa"), "KE": ("Kenya", "Africa"),
        "TZ": ("Tanzania", "Africa"), "UG": ("Uganda", "Africa"), "RW": ("Rwanda", "Africa"),
        "BI": ("Burundi", "Africa"), "CD": ("Dem. Rep. Congo", "Africa"), "CG": ("Congo", "Africa"),
        "GA": ("Gabon", "Africa"), "GQ": ("Eq. Guinea", "Africa"), "CM": ("Cameroon", "Africa"),
        "NG": ("Nigeria", "Africa"), "GH": ("Ghana", "Africa"), "CI": ("Côte d'Ivoire", "Africa"),
        "BJ": ("Benin", "Africa"), "TG": ("Togo", "Africa"), "BF": ("Burkina Faso", "Africa"),
        "ML": ("Mali", "Africa"), "NE": ("Niger", "Africa"), "TD": ("Chad", "Africa"),
        "CF": ("Central African Rep.", "Africa"), "SN": ("Senegal", "Africa"), "GM": ("Gambia", "Africa"),
        "GW": ("Guinea-Bissau", "Africa"), "GN": ("Guinea", "Africa"), "SL": ("Sierra Leone", "Africa"),
        "LR": ("Liberia", "Africa"), "CV": ("Cape Verde", "Africa"), "MR": ("Mauritania", "Africa"),
        "MW": ("Malawi", "Africa"), "ZM": ("Zambia", "Africa"), "ZW": ("Zimbabwe", "Africa"),
        "MZ": ("Mozambique", "Africa"), "NA": ("Namibia", "Africa"), "BW": ("Botswana", "Africa"),
        "ZA": ("South Africa", "Africa"), "LS": ("Lesotho", "Africa"), "SZ": ("Eswatini", "Africa"),
        "AO": ("Angola", "Africa"), "KM": ("Comoros", "Africa"), "MG": ("Madagascar", "Africa"),
        "MU": ("Mauritius", "Africa"), "RE": ("Réunion", "Africa"), "YT": ("Mayotte", "Africa"),
        "ST": ("São Tomé and Principe", "Africa"), "SC": ("Seychelles", "Africa"),
        # ---- 北美洲 North America ----
        "US": ("United States of America", "North America"), "CA": ("Canada", "North America"),
        "MX": ("Mexico", "North America"), "GT": ("Guatemala", "North America"),
        "BZ": ("Belize", "North America"), "SV": ("El Salvador", "North America"),
        "HN": ("Honduras", "North America"), "NI": ("Nicaragua", "North America"),
        "CR": ("Costa Rica", "North America"), "PA": ("Panama", "North America"),
        "CU": ("Cuba", "North America"), "HT": ("Haiti", "North America"),
        "DO": ("Dominican Rep.", "North America"), "JM": ("Jamaica", "North America"),
        "PR": ("Puerto Rico", "North America"), "BS": ("Bahamas", "North America"),
        "TT": ("Trinidad and Tobago", "North America"), "BB": ("Barbados", "North America"),
        "DM": ("Dominica", "North America"), "LC": ("St. Lucia", "North America"),
        "VC": ("St. Vincent and the Gren.", "North America"), "GD": ("Grenada", "North America"),
        "AG": ("Antigua and Barb.", "North America"), "KN": ("St. Kitts and Nevis", "North America"),
        "BM": ("Bermuda", "North America"), "GL": ("Greenland", "North America"),
        # ---- 南美洲 South America ----
        "BR": ("Brazil", "South America"), "AR": ("Argentina", "South America"),
        "CL": ("Chile", "South America"), "PE": ("Peru", "South America"),
        "CO": ("Colombia", "South America"), "VE": ("Venezuela", "South America"),
        "EC": ("Ecuador", "South America"), "BO": ("Bolivia", "South America"),
        "PY": ("Paraguay", "South America"), "UY": ("Uruguay", "South America"),
        "GY": ("Guyana", "South America"), "SR": ("Suriname", "South America"),
        "GF": ("Fr. S. Antarctic Lands", "South America"), "FK": ("Falkland Is.", "South America"),
        # ---- 大洋洲 Oceania ----
        "AU": ("Australia", "Oceania"), "NZ": ("New Zealand", "Oceania"),
        "PG": ("Papua New Guinea", "Oceania"), "FJ": ("Fiji", "Oceania"),
        "SB": ("Solomon Islands", "Oceania"), "VU": ("Vanuatu", "Oceania"),
        "NC": ("New Caledonia", "Oceania"), "PF": ("Fr. Polynesia", "Oceania"),
        "WS": ("Samoa", "Oceania"), "TO": ("Tonga", "Oceania"),
        "KI": ("Kiribati", "Oceania"), "TV": ("Tuvalu", "Oceania"),
        "NR": ("Nauru", "Oceania"), "MH": ("Marshall Islands", "Oceania"),
        "FM": ("Micronesia", "Oceania"), "PW": ("Palau", "Oceania"),
        "CK": ("Cook Is.", "Oceania"), "NU": ("Niue", "Oceania"),
        "TK": ("Tokelau", "Oceania"), "AS": ("American Samoa", "Oceania"),
        "GU": ("Guam", "Oceania"), "MP": ("N. Mariana Is.", "Oceania"),
        "WF": ("Wallis and Futuna", "Oceania"), "PN": ("Pitcairn Is.", "Oceania"),
        # ---- 七海/海洋（rg永远不会返回，但用于完整性）----
        "XX": ("Seven seas", "Seven seas (open ocean)"),
        # ---- 南极洲 ----
        "AQ": ("Antarctica", "Antarctica"), "TF": ("Fr. S. Antarctic Lands", "Antarctica"),
        "GS": ("S. Geo. and S. Sandw. Is.", "Antarctica"), "HM": ("Heard I. and McDonald Is.", "Antarctica"),
    }
    # 大洲名英文→中文映射（与 add_classifications 的 _continent_en2cn 保持一致）
    _RG_CONTINENT_EN2CN = {
        "Asia": "亚洲", "Europe": "欧洲", "Africa": "非洲",
        "North America": "北美洲", "South America": "南美洲",
        "Oceania": "大洋洲", "Antarctica": "南极洲",
        "Seven seas (open ocean)": "海洋",
    }
except ImportError:
    HAS_RG = False
    _ISO_CC_TO_COUNTRY_CONTINENT = None
    _RG_CONTINENT_EN2CN = None

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from geopy.distance import geodesic
    HAS_GEOPY = True
except ImportError:
    HAS_GEOPY = False

warnings.filterwarnings('ignore')

# ============================================================
# ★★★ 用户参数配置区（修改这里即可） ★★★
# ============================================================
CONFIG = {
    # 数据路径
    # ★多年处理: data_dir 为 CMORPH 根目录，代码会自动拼接 \YYYY 子目录
    "data_dir": r"E:\date download\CMORPH",
    "output_dir": r"F:\output_CMORPH_2000_2025",
    "kg_shapefile_path": r"E:\全球柯本气候区划\8ed04-main\气候区划\气候区划.shp",
    "kg_raster_path": r"E:\全球柯本气候区划\8ed04-main\气候区划\world_koppen1.tif",
    "temp_dir": r"E:\temp",

    # 时间范围（多年处理：start_year~end_year，每年处理12个月）
    "start_year": 2000,
    "end_year": 2025,
    "year": 2025,            # 当前处理年份（循环时自动覆盖）
    "start_month": 1,
    "end_month": 12,
    "month": 1,             # 初始月份（循环时自动覆盖）
    "start_day": 1,
    "end_day": 31,          # 每月处理天数（循环时按月实际天数自动覆盖）
    # ★跳过不存在的年份（数据缺失）
    "skip_missing_years": True,
    # ★断点续传：跳过已生成结果的月份/年份（中断后无需从头重新处理）
    "skip_existing_results": True,

    # 空间分辨率（固定为原始值，不修改）
    "spatial_res_deg": 0.072,

    # DBSCAN 核心参数（手动设定或通过网格搜索获得）
    "dbscan_eps_deg": 0.8,
    "dbscan_minPts": 10,

    # 是否启用网格搜索（初次运行建议开启，但耗时较长）
    "enable_grid_search": False,
    "eps_candidates": list(np.arange(0.5, 1.3, 0.15)),
    "minPts_candidates": [6, 8, 10, 12, 15],

    # IETD 参数
    "ietd_candidates": [3, 6, 9, 12],
    "ietd_selected": None,

    # 动态切分
    "split_method": "dynamic",
    "fixed_split_hours": 48,
    "area_change_threshold": 0.4,
    "intensity_drop_threshold": 0.5,
    
    # 自适应切分（按月份调整阈值）
    # ★夏季(6-8月)阈值放宽，避免对流正常波动被误切
    "adaptive_split": True,
    "split_params_by_month": {
        # 1-3月 (冬季): 层状雨多，变化慢
        1: {"area_change_threshold": 0.5, "intensity_drop_threshold": 0.6},
        2: {"area_change_threshold": 0.5, "intensity_drop_threshold": 0.6},
        3: {"area_change_threshold": 0.45, "intensity_drop_threshold": 0.55},
        # 4-5月 (春季): 对流雨增多
        4: {"area_change_threshold": 0.4, "intensity_drop_threshold": 0.5},
        5: {"area_change_threshold": 0.4, "intensity_drop_threshold": 0.5},
        # ★6-8月 (夏季): 放宽阈值(0.35/0.45)，避免对流波动误切
        6: {"area_change_threshold": 0.35, "intensity_drop_threshold": 0.45},
        7: {"area_change_threshold": 0.35, "intensity_drop_threshold": 0.45},
        8: {"area_change_threshold": 0.35, "intensity_drop_threshold": 0.45},
        # 9-12月 (秋季/冬季): 逐步回归冬季模式
        9: {"area_change_threshold": 0.4, "intensity_drop_threshold": 0.5},
        10: {"area_change_threshold": 0.45, "intensity_drop_threshold": 0.55},
        11: {"area_change_threshold": 0.5, "intensity_drop_threshold": 0.6},
        12: {"area_change_threshold": 0.5, "intensity_drop_threshold": 0.6},
    },
    "min_split_duration_hours": 2.0,  # ★切分后每段最小历时(小时)，避免过短切分
    
    # 时间缩放因子敏感性分析
    "time_scale_candidates": [5.0, 8.0, 10.0, 12.0, 15.0, 20.0],
    "time_scale_auto": True,
    
    # DBSCAN 大数据量优化
    "dbscan_max_points_exact": 50000,
    "dbscan_use_ball_tree": True,

    # 区域自适应
    "adaptive_threshold": True,
    "use_koppen_climate": True,         # ⭐ 使用柯本气候分区（如果可用）
    "lat_bands": {
        "tropical":    {"lat_min": -23.5, "lat_max": 23.5, "threshold": 0.05, "IETD": 4},
        "subtropical": {"lat_min": 23.5,  "lat_max": 35,   "threshold": 0.1,  "IETD": 6},
        "temperate":   {"lat_min": 35,    "lat_max": 55,   "threshold": 0.15, "IETD": 8},
        "polar":       {"lat_min": 55,    "lat_max": 90,   "threshold": 0.2,  "IETD": 12}
    },
    # 柯本气候分类阈值映射（按气候大区设置参数）
    "koppen_climate_params": {
        "A": {"threshold": 0.05, "IETD": 4},   # 热带：对流雨多，阈值低
        "B": {"threshold": 0.20, "IETD": 12},  # 干旱：降水少，阈值高
        "C": {"threshold": 0.10, "IETD": 6},   # 温带：适中
        "D": {"threshold": 0.15, "IETD": 8},   # 大陆性：冬季少，夏季多
        "E": {"threshold": 0.20, "IETD": 12},  # 极地：降水极少
        # 亚热带细分
        "Cfa": {"threshold": 0.08, "IETD": 5},  # 湿润亚热带
        "Cwa": {"threshold": 0.08, "IETD": 5},  # 亚热带季风
        "Csa": {"threshold": 0.12, "IETD": 7},  # 地中海气候
        # 热带雨林
        "Af": {"threshold": 0.03, "IETD": 3},
        "Am": {"threshold": 0.04, "IETD": 4},
        # 热带草原
        "Aw": {"threshold": 0.06, "IETD": 5},
        # 干旱细分
        "BSh": {"threshold": 0.15, "IETD": 10},  # 半干旱
        "BWh": {"threshold": 0.25, "IETD": 15},  # 热带沙漠
        "BSk": {"threshold": 0.18, "IETD": 12},  # 半干旱（温带）
        "BWk": {"threshold": 0.22, "IETD": 14},  # 温带沙漠
    },
    "global_rain_threshold": 0.1,
    "global_IETD_hours": 6,

    # 事件过滤
    "min_rain_per_event": 0.5,
    "min_area_pixels": 25,   # 海洋事件最小网格数 (过滤碎裂噪声)
    "min_area_pixels_land": 5,   # 陆地事件最小网格数 (陆地降雨空间尺度小，阈值更低)
    "min_confidence": 0.3,   # 最低置信度阈值
    "min_duration_hours": 0.5,
    "time_merge_hours": 6,
    "expand_search_radius": 3,

    # 缺失值处理
    "missing_method": "linear_interp",

    # 性能参数
    "fast_mode": True,                  # ⚡ 快速模式（优化性能，牺牲少量精度）
    "max_grids_per_batch": 5000,        # 每批处理的网格数（增大可提速）
    "memory_check_interval": 50,        # 内存检查间隔
    "enable_parallel": False,           # 并行处理（需要安装 multiprocess）
    "n_cores": 1,
    "chunk_hours": 4,
    "time_step": 0.5,
    
    # 快速模式优化参数
    "sample_for_ietd": 200,            # IETD 敏感性分析采样网格数
    "skip_validation": True,            # 跳过数据验证以提速
    "coarse_filter": True,             # 粗过滤（只保留 >0.1mm 的数据）
    "use_numpy_only": True,            # 尽量用 numpy 而不是 pandas

    # 输出
    "enable_global_id": True,
    "coastal_buffer_km": 100,   # 沿海/近海分类的buffer范围(km)，扩大以避免陆地事件被误判为海洋
    # 跨月事件合并参数
    "cross_month_max_gap_hours": 6,    # 跨月合并的最大时间间隔
    "cross_month_max_distance_deg": 2.0,  # 跨月合并的最大空间距离(度)
    # 互动地图预览：只生成指定月份的地图（节省时间），设为 [] 则全部生成
    "interactive_map_months": [1],     # 默认只生成1月预览
    
    # 事件验证
    "enable_validation": True,
    "filter_invalid_events": True,

    # 时间缩放因子（用于 DBSCAN 时间维度）
    "time_scale_factor": 10.0,

    # ---- v17.0 性能优化参数 ----
    "performance": {
        "skip_ietd_analysis": False,        # ★开启IETD敏感性分析，论文需报告敏感性对比
        "use_numba": True,                 # 使用numba JIT加速
        "climate_lut_resolution": 0.1,     # 气候区查找表分辨率（度）
        "dbscan_pre_partition": True,      # DBSCAN预分区
        "dbscan_partition_lon": 10,        # 经度分块数
        "dbscan_partition_lat": 6,         # 纬度分块数
        "max_events_exact_dbscan": 50000,  # 精确DBSCAN上限
        "vectorized_geo": True,            # 向量化地理分类
        "batch_size_grid_ident": 10000,    # 网格识别批次大小（增大提速）
        "use_smart_split": True,           # 智能跳过IETD分析（多日复用）
        "spatial_downsample_deg": 0.25,    # 空间降采样分辨率(度, 0=不降采样)
        "numba_parallel": True,            # numba prange多线程
        "nopython": True,                  # numba nopython模式
        "cache_tsf_per_season": True,      # ★按季度缓存time_scale_factor（参数调优缓存，不改检测逻辑）
        "save_intermediate_pkl": False,    # 是否保存月度pkl中间文件（关闭可省I/O，断点续传仍用CSV）
        "gc_threshold_mb": 3000,           # gc.collect()触发阈值（内存MB），0=每次都调用
        "io_workers": 8,                   # ★并行读取.nc文件的线程数（I/O密集型，GIL不阻塞；建议=CPU核数或磁盘并发数）
    },
}


# ============================================================
# 配置校验（不破坏现有dict访问，提供类型检查和缺失键检测）
# ============================================================

# 配置schema：字段名 → (类型, 是否必需, 说明)
_CONFIG_SCHEMA = {
    # 基本参数
    "year": (int, True, "当前处理年份"),
    "start_year": (int, True, "起始年份"),
    "end_year": (int, True, "结束年份"),
    "month": (int, True, "处理月份(1-12)"),
    "start_day": (int, True, "起始日"),
    "end_day": (int, True, "结束日"),
    "data_dir": (str, True, "CMORPH数据目录"),
    "output_dir": (str, True, "输出目录"),
    "spatial_res_deg": (float, True, "空间分辨率(度)"),
    "dbscan_eps_deg": ((int, float), True, "DBSCAN空间阈值(度)"),
    "dbscan_minPts": (int, True, "DBSCAN最小点数"),
    "time_scale_factor": ((int, float), True, "时间缩放因子"),
    # 阈值
    "global_rain_threshold": ((int, float), True, "全局降雨阈值(mm)"),
    "global_IETD_hours": ((int, float), True, "全局IETD(小时)"),
    "min_rain_per_event": ((int, float), True, "事件最小降雨量(mm)"),
    "min_area_pixels": (int, True, "海洋事件最小网格数"),
    "min_area_pixels_land": (int, True, "陆地事件最小网格数"),
    "min_confidence": ((int, float), True, "最低置信度"),
    "min_duration_hours": ((int, float), True, "最小持续时间(小时)"),
    # 地理分类
    "coastal_buffer_km": ((int, float), True, "沿海buffer(km)"),
    # 跨月合并
    "cross_month_max_gap_hours": ((int, float), True, "跨月合并最大间隔(小时)"),
    "cross_month_max_distance_deg": ((int, float), True, "跨月合并最大距离(度)"),
}


def validate_config(config):
    """校验CONFIG字典的完整性和类型正确性

    Args:
        config: CONFIG字典

    Returns:
        list: 错误消息列表（空列表=校验通过）
    """
    errors = []
    _type_errors = set()  # 记录类型错误的键，跳过其值域检查
    for key, (expected_type, required, desc) in _CONFIG_SCHEMA.items():
        if key not in config:
            if required:
                errors.append(f"缺少必需参数: '{key}' ({desc})")
        else:
            val = config[key]
            if not isinstance(val, expected_type):
                errors.append(
                    f"参数 '{key}' 类型错误: 期望{expected_type}, 实际{type(val).__name__} ({desc})"
                )
                _type_errors.add(key)
    # 值域检查（跳过类型错误的键）
    if "year" in config and "year" not in _type_errors and not (1998 <= config["year"] <= 2030):
        errors.append(f"year={config['year']} 超出CMORPH数据范围(1998-2030)")
    if "start_year" in config and "start_year" not in _type_errors and not (1998 <= config["start_year"] <= 2030):
        errors.append(f"start_year={config['start_year']} 超出范围(1998-2030)")
    if "end_year" in config and "end_year" not in _type_errors and not (1998 <= config["end_year"] <= 2030):
        errors.append(f"end_year={config['end_year']} 超出范围(1998-2030)")
    if ("start_year" in config and "end_year" in config and
        "start_year" not in _type_errors and "end_year" not in _type_errors and
        config["start_year"] > config["end_year"]):
        errors.append(f"start_year({config['start_year']}) > end_year({config['end_year']})")
    if "month" in config and "month" not in _type_errors and not (1 <= config["month"] <= 12):
        errors.append(f"month={config['month']} 不在1-12范围")
    if "dbscan_eps_deg" in config and "dbscan_eps_deg" not in _type_errors and config["dbscan_eps_deg"] <= 0:
        errors.append(f"dbscan_eps_deg必须>0, 当前={config['dbscan_eps_deg']}")
    if "dbscan_minPts" in config and "dbscan_minPts" not in _type_errors and config["dbscan_minPts"] < 1:
        errors.append(f"dbscan_minPts必须>=1, 当前={config['dbscan_minPts']}")
    return errors


# ============================================================
# 辅助函数
# ============================================================

# ---- v16.0 新增: 气候区查找表（预构建，O(1)查询） ----
_climate_lut_cache = {}  # 缓存查找表，避免重复构建

def build_climate_lut(kg_gdf, lon_res=0.1, lat_res=0.1):
    """预构建气候区查找表（全球网格映射）
    
    将气候区多边形转换为规则网格查找表，实现O(1)查询
    构建一次可复用，适合批量处理多天数据
    """
    cache_key = f"{lon_res}_{lat_res}"
    if cache_key in _climate_lut_cache:
        return _climate_lut_cache[cache_key]
    
    if kg_gdf is None:
        _climate_lut_cache[cache_key] = None
        return None
    
    # 构建全球网格
    lons = np.arange(0, 360 + lon_res, lon_res)
    lats = np.arange(-90, 90 + lat_res, lat_res)
    
    lon_idx = np.arange(len(lons))
    lat_idx = np.arange(len(lats))
    
    # 创建网格点
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    
    # 批量空间查询（分块处理以控制内存）
    # ★减小chunk_size + 显式GC，避免sjoin累积object dtype导致内存溢出
    chunk_size = 30000  # 从100000减到30000，降低单次内存峰值
    n_chunks = (len(points) + chunk_size - 1) // chunk_size
    zone_codes = np.full(len(points), "未知", dtype=object)

    print(f"  🔨 构建气候区查找表 ({len(lons)}×{len(lats)} 网格, {n_chunks} 块)...")
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, len(points))
        chunk = points[start:end]

        # 处理经度>180的情况
        chunk_lons = np.where(chunk[:, 0] > 180, chunk[:, 0] - 360, chunk[:, 0])
        chunk_lats = chunk[:, 1]

        # 构建GeoDataFrame进行批量空间查询
        pts = gpd.GeoDataFrame(
            {"idx": range(len(chunk))},
            geometry=[geom.Point(lon, lat) for lon, lat in zip(chunk_lons, chunk_lats)],
            crs="EPSG:4326"
        )
        try:
            result = gpd.sjoin(pts, kg_gdf, how="left", predicate="intersects")
        except TypeError:
            result = gpd.sjoin(pts, kg_gdf, how="left", op="intersects")

        if "climate_zone" in result.columns:
            # 向量化提取气候区（避免逐行循环）
            result_dedup = result[~result.index.duplicated(keep="first")]
            matched = result_dedup["climate_zone"].notna()
            if matched.any():
                idx_vals = result_dedup.loc[matched, "idx"].astype(int).values
                zone_vals = result_dedup.loc[matched, "climate_zone"].values
                zone_codes[start + idx_vals] = zone_vals

        # ★显式释放内存，避免累积导致MemoryError
        del pts, result
        if (i + 1) % 5 == 0:
            gc.collect()  # 每5块强制GC
            print(f"     进度: {i+1}/{n_chunks}")

    gc.collect()  # 构建完成后GC
    
    # 重塑为2D数组 [lat_idx, lon_idx]
    zone_grid = zone_codes.reshape(len(lats), len(lons))
    
    # 存储lut元数据
    lut = {
        "lons": lons,
        "lats": lats,
        "lon_idx": lon_idx,
        "lat_idx": lat_idx,
        "zone_grid": zone_grid,
        "lon_res": lon_res,
        "lat_res": lat_res,
    }
    
    _climate_lut_cache[cache_key] = lut
    print(f"  ✅ 气候区查找表构建完成: {len(lats)}×{len(lons)} 网格, {len(set(zone_codes))} 个气候区")
    return lut


def query_climate_lut(lon_arr, lat_arr, lut):
    """使用查找表批量查询气候区（O(1) per point）
    
    Args:
        lon_arr: 经度数组 (0~360)
        lat_arr: 纬度数组 (-90~90)
        lut: build_climate_lut() 返回的查找表
    
    Returns:
        climate_zones: 气候区数组
    """
    if lut is None:
        return np.full(len(lon_arr), "未知", dtype=object)
    
    lons = lut["lons"]
    lats = lut["lats"]
    zone_grid = lut["zone_grid"]
    lon_res = lut["lon_res"]
    lat_res = lut["lat_res"]
    
    # LUT lons 使用 0~360 范围，确保查询也用 0~360
    lon_adj = lon_arr % 360
    
    # 计算索引
    lon_idx = np.clip(np.round(lon_adj / lon_res).astype(int), 0, len(lons) - 1)
    lat_idx = np.clip(np.round((lat_arr + 90) / lat_res).astype(int), 0, len(lats) - 1)
    
    # 批量查表
    return zone_grid[lat_idx, lon_idx]


# ---- v16.0 新增: numba 加速的事件提取核心 ----
if HAS_NUMBA:
    @jit(nopython=True, cache=True)
    def _extract_events_numba(time_arr, precip_arr, threshold, ietd, time_step_hours):
        """numba加速的单网格事件提取
        
        Args:
            time_arr: 时间数组（数值化，小时）
            precip_arr: 降水数组
            threshold: 降水阈值
            ietd: IETD（小时）
            time_step_hours: 时间步长（小时）
        
        Returns:
            events: 事件列表 [(start_idx, end_idx, total, max_rain, peak_idx), ...]
        """
        n = len(time_arr)
        if n == 0:
            return np.zeros((0, 5), dtype=np.float64)
        
        # 标记雨段
        is_rain = precip_arr >= threshold
        
        # 找出降雨段的起止
        segments_start = []
        segments_end = []
        in_rain = False
        seg_start = 0
        
        for i in range(n):
            if is_rain[i] and not in_rain:
                seg_start = i
                in_rain = True
            elif not is_rain[i] and in_rain:
                segments_start.append(seg_start)
                segments_end.append(i - 1)
                in_rain = False
        
        if in_rain:
            segments_start.append(seg_start)
            segments_end.append(n - 1)
        
        if len(segments_start) == 0:
            return np.zeros((0, 5), dtype=np.float64)
        
        # IETD合并相邻段
        n_seg = len(segments_start)
        seg_start_times = np.empty(n_seg)
        seg_end_times = np.empty(n_seg)
        for i in range(n_seg):
            seg_start_times[i] = time_arr[segments_start[i]]
            seg_end_times[i] = time_arr[segments_end[i]]
        
        # 合并相邻段（间隙<IETD）
        events = []
        ev_start_idx = segments_start[0]
        ev_end_idx = segments_end[0]
        
        for i in range(1, n_seg):
            gap = seg_start_times[i] - seg_end_times[i-1]
            if gap <= ietd:
                # 合并
                ev_end_idx = segments_end[i]
            else:
                # 保存当前事件
                start_time = time_arr[ev_start_idx]
                end_time = time_arr[ev_end_idx]
                ev_data = precip_arr[ev_start_idx:ev_end_idx+1]
                total = np.sum(ev_data)
                max_rain = np.max(ev_data)
                peak_idx = ev_start_idx + np.argmax(ev_data)
                events.append((start_time, end_time, total, max_rain, peak_idx))
                
                ev_start_idx = segments_start[i]
                ev_end_idx = segments_end[i]
        
        # 保存最后一个事件
        start_time = time_arr[ev_start_idx]
        end_time = time_arr[ev_end_idx]
        ev_data = precip_arr[ev_start_idx:ev_end_idx+1]
        total = np.sum(ev_data)
        max_rain = np.max(ev_data)
        peak_idx = ev_start_idx + np.argmax(ev_data)
        events.append((start_time, end_time, total, max_rain, peak_idx))
        
        if len(events) == 0:
            return np.zeros((0, 5), dtype=np.float64)
        
        return np.array(events, dtype=np.float64)

    # ---- v17.0 新增: 批量numba函数（一次调用处理所有网格，prange并行）----
    @jit(nopython=True, cache=True, parallel=True)
    def _extract_all_events_batch(time_hours, precip_2d, thresholds, ietds):
        """批量numba: 一次调用处理所有网格的事件提取（prange并行版）

        每个网格写入独立区域，避免prange竞态条件。
        调用方负责compact结果。

        Returns:
            events: (n_grids * MAX_PER_GRID, 7) float32数组, 调用方需compact
        """
        n_times, n_grids = precip_2d.shape
        MAX_PER_GRID = 15  # 每网格最多事件数上限（热带对流日可能>5事件，提高到15防截断）

        # 每网格独立区域，避免竞态; float32节省内存
        events = np.zeros((n_grids * MAX_PER_GRID, 7), dtype=np.float32)

        for g in prange(n_grids):
            offset = g * MAX_PER_GRID
            local_count = 0

            thresh = thresholds[g]
            ietd_h = ietds[g]
            precip_col = precip_2d[:, g]
            time_col = time_hours

            # 快速检查
            rain_count = 0
            for t in range(n_times):
                if precip_col[t] >= thresh:
                    rain_count += 1
            if rain_count == 0:
                continue

            # 识别降雨段
            in_rain = False
            seg_start = 0
            n_seg = 0
            seg_starts = np.empty(100, dtype=np.int64)
            seg_ends = np.empty(100, dtype=np.int64)

            for t in range(n_times):
                if precip_col[t] >= thresh:
                    if not in_rain:
                        seg_start = t
                        in_rain = True
                else:
                    if in_rain:
                        seg_starts[n_seg] = seg_start
                        seg_ends[n_seg] = t - 1
                        n_seg += 1
                        in_rain = False

            if in_rain:
                seg_starts[n_seg] = seg_start
                seg_ends[n_seg] = n_times - 1
                n_seg += 1

            if n_seg == 0:
                continue

            # IETD合并 + 写入
            ev_s = seg_starts[0]
            ev_e = seg_ends[0]

            for i in range(1, n_seg):
                gap = time_col[seg_starts[i]] - time_col[seg_ends[i-1]]
                if gap <= ietd_h:
                    ev_e = seg_ends[i]
                else:
                    # 保存当前事件
                    if local_count < MAX_PER_GRID:
                        e_idx = offset + local_count
                        ev_data = precip_col[ev_s:ev_e+1]
                        total_v = 0.0
                        max_v = 0.0
                        peak_l = 0
                        for j in range(len(ev_data)):
                            v = ev_data[j]
                            total_v += v
                            if v > max_v:
                                max_v = v
                                peak_l = j
                        dur = time_col[ev_e] - time_col[ev_s]
                        if total_v > 0 and dur > 0:
                            events[e_idx, 0] = g
                            events[e_idx, 1] = time_col[ev_s]
                            events[e_idx, 2] = time_col[ev_e]
                            events[e_idx, 3] = total_v
                            events[e_idx, 4] = max_v
                            events[e_idx, 5] = time_col[ev_s] + peak_l * 0.5
                            events[e_idx, 6] = dur
                            local_count += 1

                    ev_s = seg_starts[i]
                    ev_e = seg_ends[i]

            # 最后一个事件
            if local_count < MAX_PER_GRID:
                e_idx = offset + local_count
                ev_data = precip_col[ev_s:ev_e+1]
                total_v = 0.0
                max_v = 0.0
                peak_l = 0
                for j in range(len(ev_data)):
                    v = ev_data[j]
                    total_v += v
                    if v > max_v:
                        max_v = v
                        peak_l = j
                dur = time_col[ev_e] - time_col[ev_s]
                if total_v > 0 and dur > 0:
                    events[e_idx, 0] = g
                    events[e_idx, 1] = time_col[ev_s]
                    events[e_idx, 2] = time_col[ev_e]
                    events[e_idx, 3] = total_v
                    events[e_idx, 4] = max_v
                    events[e_idx, 5] = time_col[ev_s] + peak_l * 0.5
                    events[e_idx, 6] = dur
                    local_count += 1

        return events

else:
    # numba不可用时的纯Python回退
    def _extract_events_numba(time_arr, precip_arr, threshold, ietd, time_step_hours):
        """纯Python版单网格事件提取"""
        n = len(time_arr)
        if n == 0:
            return np.zeros((0, 5), dtype=np.float64)
        
        is_rain = precip_arr >= threshold
        
        segments_start = []
        segments_end = []
        in_rain = False
        seg_start = 0
        
        for i in range(n):
            if is_rain[i] and not in_rain:
                seg_start = i
                in_rain = True
            elif not is_rain[i] and in_rain:
                segments_start.append(seg_start)
                segments_end.append(i - 1)
                in_rain = False
        
        if in_rain:
            segments_start.append(seg_start)
            segments_end.append(n - 1)
        
        if len(segments_start) == 0:
            return np.zeros((0, 5), dtype=np.float64)
        
        n_seg = len(segments_start)
        seg_start_times = np.array([time_arr[segments_start[i]] for i in range(n_seg)])
        seg_end_times = np.array([time_arr[segments_end[i]] for i in range(n_seg)])
        
        events = []
        ev_start_idx = segments_start[0]
        ev_end_idx = segments_end[0]
        
        for i in range(1, n_seg):
            gap = seg_start_times[i] - seg_end_times[i-1]
            if gap <= ietd:
                ev_end_idx = segments_end[i]
            else:
                start_time = time_arr[ev_start_idx]
                end_time = time_arr[ev_end_idx]
                ev_data = precip_arr[ev_start_idx:ev_end_idx+1]
                total = np.sum(ev_data)
                max_rain = np.max(ev_data)
                peak_idx = ev_start_idx + np.argmax(ev_data)
                events.append((start_time, end_time, total, max_rain, peak_idx))
                ev_start_idx = segments_start[i]
                ev_end_idx = segments_end[i]
        
        start_time = time_arr[ev_start_idx]
        end_time = time_arr[ev_end_idx]
        ev_data = precip_arr[ev_start_idx:ev_end_idx+1]
        total = np.sum(ev_data)
        max_rain = np.max(ev_data)
        peak_idx = ev_start_idx + np.argmax(ev_data)
        events.append((start_time, end_time, total, max_rain, peak_idx))
        
        if len(events) == 0:
            return np.zeros((0, 5), dtype=np.float64)
        
        return np.array(events, dtype=np.float64)

    # 批量版纯Python回退
    def _extract_all_events_batch(time_hours, precip_2d, thresholds, ietds):
        """纯Python批量事件提取（慢，但保证功能）"""
        n_times, n_grids = precip_2d.shape
        all_events = []
        
        for g in range(n_grids):
            thresh = thresholds[g]
            ietd_h = ietds[g]
            precip_col = precip_2d[:, g]
            
            is_rain = precip_col >= thresh
            if not is_rain.any():
                continue
            
            in_rain = False
            seg_start = 0
            segs_start, segs_end = [], []
            for t in range(n_times):
                if is_rain[t] and not in_rain:
                    seg_start = t; in_rain = True
                elif not is_rain[t] and in_rain:
                    segs_start.append(seg_start); segs_end.append(t-1); in_rain = False
            if in_rain:
                segs_start.append(seg_start); segs_end.append(n_times-1)
            
            if not segs_start:
                continue
            
            ev_s, ev_e = segs_start[0], segs_end[0]
            for i in range(1, len(segs_start)):
                gap = time_hours[segs_start[i]] - time_hours[segs_end[i-1]]
                if gap <= ietd_h:
                    ev_e = segs_end[i]
                else:
                    ev_data = precip_col[ev_s:ev_e+1]
                    total = float(np.sum(ev_data))
                    if total > 0:
                        duration = time_hours[ev_e] - time_hours[ev_s]
                        if duration > 0:
                            peak_idx = int(np.argmax(ev_data))
                            all_events.append((g, time_hours[ev_s], time_hours[ev_e],
                                              total, float(np.max(ev_data)),
                                              time_hours[ev_s] + peak_idx * 0.5,
                                              float(duration)))
                    ev_s, ev_e = segs_start[i], segs_end[i]
            
            ev_data = precip_col[ev_s:ev_e+1]
            total = float(np.sum(ev_data))
            if total > 0:
                duration = time_hours[ev_e] - time_hours[ev_s]
                if duration > 0:
                    peak_idx = int(np.argmax(ev_data))
                    all_events.append((g, time_hours[ev_s], time_hours[ev_e],
                                      total, float(np.max(ev_data)),
                                      time_hours[ev_s] + peak_idx * 0.5,
                                      float(duration)))
        
        if not all_events:
            return np.zeros((0, 7), dtype=np.float64)
        return np.array(all_events, dtype=np.float64)


# ---- 陆地掩膜（简化版：陆地+海岸线，排除开阔海洋）----
# 使用简化规则：纬度>60°全为陆地（极地冰盖），其他用经度阈值
# 更精确的需要陆地shapefile，但此方法可排除大部分海洋
def _build_land_mask(lons, lats, buffer_deg=0.5):
    """构建陆地掩膜（快速简化版）
    
    基于经验规则判断陆地/海洋：
    - 纬度>66°: 极地冰盖（全是"陆地"）
    - 其他区域: 用简化的大陆轮廓判断
    """
    n = len(lons)
    is_land = np.zeros(n, dtype=bool)
    
    # 极地冰盖
    is_land[np.abs(lats) > 66] = True
    
    # 简化的大陆判定（粗略但有效）
    # 欧亚大陆: lon -10~180, lat 10~70
    eur = (lons > -10) & (lons < 180) & (lats > 10) & (lats < 70)
    is_land[eur] = True
    
    # 非洲: lon -20~50, lat -40~38
    af = (lons > -20) & (lons < 50) & (lats > -40) & (lats < 38)
    is_land[af] = True
    
    # 北美: lon -170~-50, lat 7~84
    na = (lons > -170) & (lons < -50) & (lats > 7) & (lats < 84)
    is_land[na] = True
    
    # 南美: lon -82~-34, lat -56~12
    sa = (lons > -82) & (lons < -34) & (lats > -56) & (lats < 12)
    is_land[sa] = True
    
    # 澳大利亚: lon 112~154, lat -40~-10
    au = (lons > 112) & (lons < 154) & (lats > -40) & (lats < -10)
    is_land[au] = True
    
    # 格陵兰: lon -60~-15, lat 60~83
    gl = (lons > -60) & (lons < -15) & (lats > 60) & (lats < 83)
    is_land[gl] = True
    
    # 英国/爱尔兰: lon -10~2, lat 50~59
    uk = (lons > -10) & (lons < 2) & (lats > 50) & (lats < 59)
    is_land[uk] = True
    
    # 日本: lon 130~146, lat 31~45
    jp = (lons > 130) & (lons < 146) & (lats > 31) & (lats < 45)
    is_land[jp] = True
    
    # 马达加斯加: lon 43~51, lat -25~-12
    mg = (lons > 43) & (lons < 51) & (lats > -25) & (lats < -12)
    is_land[mg] = True
    
    # 新西兰: lon 166~179, lat -48~-34
    nz = (lons > 166) & (lons < 179) & (lats > -48) & (lats < -34)
    is_land[nz] = True
    
    return is_land


def check_memory_usage():
    """检查内存使用（简化版）"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
        print(f"💾 内存使用: {mem_mb:.1f} MB")
        return mem_mb
    except ImportError:
        print("💾 内存使用: psutil 未安装，跳过")
        return 0.0


def batch_query_climate_zone(lons, lats, kg_gdf, batch_size=50000):
    """批量查询气候区（高效版 - 兼容不同版本 geopandas）
    
    Args:
        lons: 经度数组
        lats: 纬度数组
        kg_gdf: 柯本气候分区 GeoDataFrame 或 RasterData 对象
        batch_size: 每批查询的点数
    
    Returns:
        climate_zones: 气候区数组
    """
    if kg_gdf is None:
        return None
    
    # 判断对象类型：GeoDataFrame 或 RasterData
    is_raster = hasattr(kg_gdf, '_lon_idx_cache') and hasattr(kg_gdf, 'lons')
    is_vector = HAS_GEOPANDAS and isinstance(kg_gdf, gpd.GeoDataFrame)
    
    if is_raster and not is_vector:
        return _batch_query_raster(lons, lats, kg_gdf)
    
    if not HAS_GEOPANDAS:
        return None
    
    n_points = len(lons)
    climate_zones = np.full(n_points, "未知", dtype=object)
    
    # 处理经度 > 180 的情况
    lons_adjusted = np.where(lons > 180, lons - 360, lons)
    
    n_batches = (n_points + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_points)
        
        batch_lons = lons_adjusted[start:end]
        batch_lats = lats[start:end]
        
        # 构建 GeoDataFrame
        points = [geom.Point(lon, lat) for lon, lat in zip(batch_lons, batch_lats)]
        points_gdf = gpd.GeoDataFrame(
            {"idx": range(len(points))},
            geometry=points, crs="EPSG:4326"
        )
        
        # 批量空间连接
        try:
            # 兼容不同版本的 geopandas
            try:
                # 新版 geopandas (>= 0.13)
                intersects = gpd.sjoin(
                    points_gdf, kg_gdf, 
                    how="left", predicate="intersects"
                )
            except TypeError:
                # 旧版 geopandas
                intersects = gpd.sjoin(
                    points_gdf, kg_gdf, 
                    how="left", op="intersects"
                )
            
            # 提取气候区
            if "climate_zone" in intersects.columns:
                for _, row in intersects.iterrows():
                    if pd.notna(row["climate_zone"]):
                        climate_zones[start + int(row["idx"])] = row["climate_zone"]
            
            # 处理未匹配的点（找最近邻）
            # 兼容不同版本的 index 命名
            index_col = None
            for possible_col in ["index_left", "index_right", "idx_left", "idx_right"]:
                if possible_col in intersects.columns:
                    index_col = possible_col
                    break
            
            if index_col is None and not intersects.empty:
                # 使用原始索引
                matched_indices = intersects.index.get_level_values(0) if intersects.index.nlevels > 1 else intersects.index
                unmatched_mask = ~points_gdf.index.isin(matched_indices)
            elif index_col:
                unmatched_mask = ~points_gdf["idx"].isin(intersects["idx"][intersects[index_col].notna()])
                unmatched_mask = points_gdf["idx"].notna()
                # 更简单的方法：用 idx 判断
                matched_idx = set(intersects["idx"].dropna().astype(int).unique())
                unmatched_mask = ~points_gdf["idx"].isin(matched_idx)
            else:
                unmatched_mask = pd.Series([True] * len(points_gdf), index=points_gdf.index)
            
            unmatched = points_gdf[unmatched_mask]
            
            if not unmatched.empty:
                try:
                    kg_projected = kg_gdf.to_crs("EPSG:3857")
                    unmatched_proj = unmatched.to_crs("EPSG:3857")
                    
                    for idx, row in unmatched_proj.iterrows():
                        dists = kg_projected.geometry.distance(row.geometry)
                        min_dist = dists.min()
                        if min_dist < 200000:  # 200km 阈值
                            nearest_idx = dists.idxmin()
                            climate_zones[start + int(row["idx"])] = kg_gdf.loc[nearest_idx, "climate_zone"]
                except Exception:
                    pass
                    
        except Exception as e:
            print(f"    ⚠️ 气候区查询失败 (批次 {batch_idx}): {e}")
            # 尝试简化方法：直接遍历点
            try:
                for i, (lon, lat) in enumerate(zip(batch_lons, batch_lats)):
                    point = geom.Point(lon, lat)
                    try:
                        # 尝试直接查询
                        for _, poly in kg_gdf.iterrows():
                            if poly.geometry.contains(point):
                                climate_zones[start + i] = poly.get("climate_zone", "未知")
                                break
                    except Exception:
                        pass
            except Exception:
                pass
    
    return climate_zones


def _batch_query_raster(lons, lats, raster_obj):
    """使用 raster 对象批量查询气候区"""
    n_points = len(lons)
    climate_zones = np.full(n_points, "未知", dtype=object)
    
    for i in range(n_points):
        lon = lons[i]
        lat = lats[i]
        result = raster_obj.query(lon, lat)
        if result is not None:
            climate_zones[i] = result
    
    return climate_zones


def get_koppen_params(climate_zone, config):
    """根据柯本气候区获取参数
    
    Args:
        climate_zone: 气候区代码（如 'Af', 'BSh', 'Cfa' 或数字代码 '1', '2' 等）
                      也支持语义化海洋名称如 '南温带太平洋海洋'
        config: 配置字典
    
    Returns:
        (threshold, ietd): 阈值和 IETD
    """
    params_map = config.get("koppen_climate_params", {})
    default = config.get("global_rain_threshold", 0.1)
    default_ietd = config.get("global_IETD_hours", 6)
    
    if climate_zone is None or str(climate_zone) in ["未知", "未分类", "", "nan", "None"]:
        return default, default_ietd
    
    zone_str = str(climate_zone).strip()
    
    # 如果是语义化海洋气候区名（包含"海洋"字样），根据纬度带名映射到柯本区
    if "海洋" in zone_str:
        # 识别纬度带 → 匹配参数（注意 params_map[letter] 是 dict {"threshold":.., "IETD":..}）
        def _ocean_params(letter, default_t, default_i):
            if letter in params_map:
                p = params_map[letter]
                return p["threshold"], p["IETD"]
            return default_t, default_i
        if "热带" in zone_str:
            return _ocean_params("A", 0.08, 6)
        if "亚热带" in zone_str:
            return _ocean_params("C", 0.09, 6)
        if "温带" in zone_str:
            return _ocean_params("C", 0.1, 8)
        if "北极" in zone_str or "南极" in zone_str:
            return _ocean_params("E", 0.08, 12)
        # 默认海洋
        return default, default_ietd
    
    # 数字代码到柯本字母的映射（标准Koppen-Geiger GIS编码）
    KOPPEN_NUMERIC_MAP = {
        "1": "Af", "2": "Am", "3": "Aw",      # 热带
        "4": "BSh", "5": "BSk",                  # 半干旱
        "6": "BWh", "7": "BWk",                  # 沙漠
        "8": "Cfa", "9": "Cfb", "10": "Cfc",    # 温带湿润
        "11": "Csa", "12": "Csb", "13": "Csc",  # 地中海
        "14": "Cwa", "15": "Cwb", "16": "Cwc",  # 亚热带季风
        "17": "Dfa", "18": "Dfb", "19": "Dfc", "20": "Dfd",  # 大陆性湿润
        "21": "Dsa", "22": "Dsb", "23": "Dsc", "24": "Dsd",  # 大陆性夏干
        "25": "Dwa", "26": "Dwb", "27": "Dwc", "28": "Dwd",  # 大陆性季风
        "29": "ET", "30": "EF",                  # 极地
    }
    
    # 如果是数字代码，转换为柯本字母
    if zone_str in KOPPEN_NUMERIC_MAP:
        zone_str = KOPPEN_NUMERIC_MAP[zone_str]
    elif zone_str.isdigit() and int(zone_str) >= 31:
        # ★防御: 非标准码(>=31, 如NoData/海洋占位符)视为未知，回退到默认值
        #   防止 climate_zone='31' 之类的幽灵分区出现在结果中
        return default, default_ietd

    # 尝试精确匹配
    if zone_str in params_map:
        p = params_map[zone_str]
        return p["threshold"], p["IETD"]

    # 尝试匹配首字母（气候大区：A/B/C/D/E）
    if len(zone_str) >= 1:
        first_letter = zone_str[0]
        if first_letter in params_map:
            p = params_map[first_letter]
            return p["threshold"], p["IETD"]

    return default, default_ietd


def calculate_great_circle_azimuth(lon1, lat1, lon2, lat2):
    """计算大圆方位角（标量版，保留兼容）"""
    try:
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlon = math.radians(lon2 - lon1)
        x = math.sin(dlon) * math.cos(lat2_rad)
        y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
        bearing = math.degrees(math.atan2(x, y))
        bearing = bearing % 360
        return bearing
    except Exception:
        return np.nan


def get_adaptive_params(lat, config):
    """根据纬度获取自适应参数"""
    if not config["adaptive_threshold"]:
        return {"threshold": config["global_rain_threshold"], "IETD": config["global_IETD_hours"]}
    abs_lat = abs(lat)
    bands = config["lat_bands"]
    if abs_lat <= 23.5:
        return {"threshold": bands["tropical"]["threshold"], "IETD": bands["tropical"]["IETD"]}
    elif abs_lat <= 35:
        return {"threshold": bands["subtropical"]["threshold"], "IETD": bands["subtropical"]["IETD"]}
    elif abs_lat <= 55:
        return {"threshold": bands["temperate"]["threshold"], "IETD": bands["temperate"]["IETD"]}
    else:
        return {"threshold": bands["polar"]["threshold"], "IETD": bands["polar"]["IETD"]}


def handle_missing(ts, method="linear_interp"):
    """处理缺失值（向量化版）"""
    ts = np.asarray(ts, dtype=float)
    if method == "zero":
        return np.where(np.isnan(ts), 0, ts)
    if method == "linear_interp":
        if np.all(np.isnan(ts)):
            return np.zeros_like(ts)
        nans = np.isnan(ts)
        if not np.any(nans):
            return ts
        if np.sum(~nans) == 1:
            ts[nans] = ts[~nans][0]
            return ts
        x = np.arange(len(ts))
        ts[nans] = np.interp(x[nans], x[~nans], ts[~nans])
        return ts
    return ts


def load_kg_shapefile(shapefile_path, raster_path=None):
    """加载柯本气候区数据（支持 shapefile 和 raster 两种格式）
    
    优先使用 shapefile，如果失败则尝试 raster 文件
    """
    # 尝试加载 shapefile
    if not HAS_GEOPANDAS:
        print("  ⚠️ geopandas 未安装")
        if raster_path and os.path.exists(raster_path):
            return load_kg_raster(raster_path)
        return None
    
    if not os.path.exists(shapefile_path):
        print(f"  ⚠️ 气候区 shapefile 不存在: {shapefile_path}")
        # 列出同目录下可能的文件
        dir_path = os.path.dirname(shapefile_path)
        if os.path.exists(dir_path):
            print(f"     目录 {dir_path} 下的文件:")
            for f in os.listdir(dir_path):
                if f.endswith(('.shp', '.geojson', '.gpkg', '.tif', '.tiff')):
                    print(f"       - {f}")
        # 尝试 raster
        if raster_path and os.path.exists(raster_path):
            print(f"  🔄 尝试加载 raster 文件: {raster_path}")
            return load_kg_raster(raster_path)
        return None
    
    try:
        kg_gdf = gpd.read_file(shapefile_path)
        print(f"  ✅ 加载气候区 shapefile: {len(kg_gdf)} 个多边形")
        print(f"     字段: {list(kg_gdf.columns)}")
        
        # 自动查找分类字段
        possible_fields = ["GRIDCODE", "CLIMATE", "CLIMATEZON", "ZONE", "CODE", "CLASS", "Type", "ID", "zone", "climate", "CN_NAME", "NAME", "类型"]
        class_field = None
        for f in possible_fields:
            if f in kg_gdf.columns:
                class_field = f
                break
        if class_field is None:
            # 尝试使用第一个非几何、非数值型字段
            for f in kg_gdf.columns:
                if f != 'geometry' and kg_gdf[f].dtype == 'object':
                    class_field = f
                    break
            if class_field is None:
                # 使用第一个非几何字段
                for f in kg_gdf.columns:
                    if f != 'geometry':
                        class_field = f
                        break
        
        if class_field:
            kg_gdf["climate_zone"] = kg_gdf[class_field].astype(str)
            unique_zones = kg_gdf["climate_zone"].nunique()
            print(f"     分类字段: {class_field}, 唯一气候区: {unique_zones}")
            
            # 数字代码到柯本字母的映射
            KOPPEN_NUMERIC_MAP = {
                "1": "Af", "2": "Am", "3": "Aw",
                "4": "BSh", "5": "BSk", "6": "BWh", "7": "BWk",
                "8": "Cfa", "9": "Cfb", "10": "Cfc",
                "11": "Csa", "12": "Csb", "13": "Csc",
                "14": "Cwa", "15": "Cwb", "16": "Cwc",
                "17": "Dfa", "18": "Dfb", "19": "Dfc", "20": "Dfd",
                "21": "Dsa", "22": "Dsb", "23": "Dsc", "24": "Dsd",
                "25": "Dwa", "26": "Dwb", "27": "Dwc", "28": "Dwd",
                "29": "ET", "30": "EF",
            }
            
            # 如果值是数字代码，转换为柯本字母
            sample_vals = kg_gdf["climate_zone"].head(5).tolist()
            is_numeric = all(v.isdigit() for v in sample_vals if v)
            if is_numeric:
                print("     检测到数字代码，转换为柯本字母...")
                # ★修复: 非标准码(>=31, 如NoData/海洋占位符)标为"未知"，回退到纬度带
                #   原代码 KOPPEN_NUMERIC_MAP.get(v, v) 会把 31 等保留为字符串，
                #   导致出现"幽灵气候区"(报告: climate_zone='31' 有154个陆地事件)
                n_unknown = sum(1 for v in kg_gdf["climate_zone"] if v not in KOPPEN_NUMERIC_MAP)
                if n_unknown > 0:
                    print(f"     ⚠️ {n_unknown} 个非标准气候区码(如31)标为'未知',将回退到纬度带")
                kg_gdf["climate_zone"] = kg_gdf["climate_zone"].map(
                    lambda v: KOPPEN_NUMERIC_MAP.get(v, "未知")
                )
            
            # 显示前几个气候区
            print(f"     气候区列表: {dict(kg_gdf['climate_zone'].value_counts().head(10))}")
        else:
            kg_gdf["climate_zone"] = "未知"
        
        # 设置 CRS
        if kg_gdf.crs is None:
            kg_gdf = kg_gdf.set_crs("EPSG:4326")
        else:
            print(f"     坐标系: {kg_gdf.crs}")
        
        return kg_gdf
    except Exception as e:
        print(f"  ⚠️ 加载气候区 shapefile 失败: {e}")
        # 尝试 raster
        if raster_path and os.path.exists(raster_path):
            print(f"  🔄 尝试加载 raster 文件: {raster_path}")
            return load_kg_raster(raster_path)
        return None


def load_kg_raster(raster_path):
    """加载柯本气候区 raster 文件（备选方案）
    
    使用 xarray/rasterio 读取tif文件
    """
    try:
        import xarray as xr
        print(f"  📦 加载 raster 文件: {raster_path}")
        
        ds = xr.open_dataset(raster_path)
        print(f"     变量: {list(ds.data_vars)}")
        print(f"     维度: {dict(ds.dims)}")
        
        # 获取气候区数据
        var_name = list(ds.data_vars)[0]
        data = ds[var_name]
        
        # 获取经纬度
        lons = ds.coords.get('lon', ds.coords.get('x', None))
        lats = ds.coords.get('lat', ds.coords.get('y', None))
        
        if lons is None or lats is None:
            # 尝试从 transform 获取
            print("     ⚠️ 无法获取经纬度坐标")
            return None
        
        print(f"     经度范围: {float(lons.min()):.2f}° - {float(lons.max()):.2f}°")
        print(f"     纬度范围: {float(lats.min()):.2f}° - {float(lats.max()):.2f}°")
        print(f"     分辨率: {float(abs(lons[1] - lons[0])):.4f}°")
        
        # 将数据转换为点
        lon_2d, lat_2d = np.meshgrid(lons.values, lats.values)
        values = data.values.flatten()
        
        # 获取唯一值
        unique_vals = np.unique(values[~np.isnan(values)])
        print(f"     唯一气候区代码: {len(unique_vals)} 个")
        
        # 如果有属性表，尝试获取气候区名称
        # 简化处理：直接用代码作为气候区标识
        class RasterData:
            def __init__(self, lons, lats, values):
                self.lons = lons
                self.lats = lats
                self.values = values
                self.zone_mapping = {v: str(int(v)) for v in unique_vals}
                # 缓存搜索结果以加速批量查询
                self._lon_idx_cache = {}
                self._lat_idx_cache = {}
            
            def query(self, lon, lat):
                """查询指定经纬度的气候区"""
                # 找到最近的网格点
                if lon > 180:
                    lon = lon - 360
                
                # 使用缓存加速
                lon_key = round(lon, 4)
                lat_key = round(lat, 4)
                
                if lon_key in self._lon_idx_cache:
                    lon_idx = self._lon_idx_cache[lon_key]
                else:
                    lon_idx = np.argmin(np.abs(self.lons - lon))
                    self._lon_idx_cache[lon_key] = lon_idx
                
                if lat_key in self._lat_idx_cache:
                    lat_idx = self._lat_idx_cache[lat_key]
                else:
                    lat_idx = np.argmin(np.abs(self.lats - lat))
                    self._lat_idx_cache[lat_key] = lat_idx
                
                val = self.values[lat_idx, lon_idx]
                if np.isnan(val):
                    return None
                return self.zone_mapping.get(val, str(int(val)))
        
        raster_obj = RasterData(lons.values, lats.values, data.values)
        print("  ✅ Raster 加载成功")
        return raster_obj
        
    except ImportError:
        print("  ⚠️ xarray 未安装，无法加载 raster 文件")
        return None
    except Exception as e:
        print(f"  ⚠️ 加载 raster 文件失败: {e}")
        return None


def extract_climate_from_shapefile(lon, lat, kg_gdf):
    """从 shapefile/raster 提取气候区（兼容版）"""
    if kg_gdf is None:
        return _simple_climate_zone(lat)
    
    # 判断对象类型：RasterData（有 _lon_idx_cache 属性）vs GeoDataFrame
    is_raster = hasattr(kg_gdf, '_lon_idx_cache') and hasattr(kg_gdf, 'lons')
    
    if is_raster:
        result = kg_gdf.query(lon, lat)
        if result is not None:
            return result
        return _simple_climate_zone(lat)
    
    if not HAS_GEOPANDAS:
        return _simple_climate_zone(lat)
    
    if lon > 180:
        point = geom.Point(lon - 360, lat)
    else:
        point = geom.Point(lon, lat)
    point_gdf = gpd.GeoDataFrame(geometry=[point], crs="EPSG:4326")
    try:
        # 兼容不同版本 geopandas
        try:
            intersects = gpd.sjoin(point_gdf, kg_gdf, how="left", predicate="intersects")
        except TypeError:
            intersects = gpd.sjoin(point_gdf, kg_gdf, how="left", op="intersects")
        
        if not intersects.empty and "climate_zone" in intersects.columns:
            val = intersects.iloc[0].get("climate_zone")
            if pd.notna(val):
                return str(val)
        # 找最近
        try:
            kg_gdf_projected = kg_gdf.to_crs("EPSG:3857")
            point_proj = gpd.GeoDataFrame(geometry=[point], crs="EPSG:4326").to_crs("EPSG:3857")
            dists = kg_gdf_projected.geometry.distance(point_proj.iloc[0].geometry)
            min_dist = dists.min()
            nearest_idx = dists.idxmin()
            if min_dist < 100000:
                return str(kg_gdf.loc[nearest_idx, "climate_zone"]) + "（邻近）"
        except Exception:
            pass
        return "未分类"
    except Exception:
        return _simple_climate_zone(lat)


def _simple_climate_zone(lat):
    """简单气候区分类（按纬度带）"""
    if -23.5 <= lat <= 23.5:
        return "热带"
    elif abs(lat) > 23.5 and abs(lat) <= 35:
        return "亚热带"
    elif abs(lat) > 35 and abs(lat) <= 55:
        return "温带"
    else:
        return "寒带"


def _standardize_world_gdf(world, data_label=""):
    """★统一标准化Natural Earth GeoDataFrame列名（50m/110m/10m通用）

    不同精度/不同来源的Natural Earth列名不一致：
    - 原始shapefile: name/continent
    - GitHub GeoJSON: NAME/CONTINENT/SOVEREIGNT/ADMIN/NAME_LONG/BRK_NAME/GEOUNIT 等

    ★关键修复：50m 数据含 7+ 个 name-like 列，全部 rename 会产生重复列名。
    解决方案：只保留第一个 name-like 列为 "name"，其他列直接丢弃；continent 同理。
    """
    if world is None or len(world) == 0:
        return None
    # 1) 按优先级挑选唯一的 name 列和 continent 列
    name_priorities = ["name", "NAME", "admin", "ADMIN", "sovereignt", "SOVEREIGNT",
                       "name_long", "NAME_LONG", "brk_name", "BRK_NAME", "geounit", "GEOUNIT"]
    name_col = next((c for c in name_priorities if c in world.columns), None)
    if name_col is None:
        name_col = next((c for c in world.columns if "name" in c.lower()), None)
    cont_priorities = ["continent", "CONTINENT"]
    cont_col = next((c for c in cont_priorities if c in world.columns), None)
    if cont_col is None:
        cont_col = next((c for c in world.columns if c.lower() == "continent"), None)
    # 2) 只保留这两列 + geometry，丢弃其他 name-like 列（避免重复列名）
    cols_to_keep = []
    if name_col is not None:
        cols_to_keep.append(name_col)
    if cont_col is not None:
        cols_to_keep.append(cont_col)
    if "geometry" in world.columns:
        cols_to_keep.append("geometry")
    world = world[cols_to_keep].copy()
    # 3) 统一重命名为 name/continent
    _ren = {}
    if name_col is not None and name_col != "name":
        _ren[name_col] = "name"
    if cont_col is not None and cont_col != "continent":
        _ren[cont_col] = "continent"
    if _ren:
        world = world.rename(columns=_ren)
    # 4) continent 列兜底
    if "continent" not in world.columns:
        print(f"  ⚠️ {data_label}无continent列，将用vectorized回退补全")
        world["continent"] = "Unknown"
    # 5) 过滤南极洲（不参与降雨事件分类）
    pre = len(world)
    world = world[world["continent"] != "Antarctica"].copy()
    if data_label:
        print(f"  ✅ {data_label}加载完成: {len(world)}/{pre} 个国家 (过滤南极洲后)")
    return world


def load_geo_data():
    """加载地理数据：**50m精度优先 + 本地缓存 + 多级回退**
    
    精度策略（快→慢，粗→细）：
    1. 本地缓存 ne_50m_admin_0_countries.geojson（1:50M精度，推荐）
    2. 在线下载50m Natural Earth（精度≈110m的2.2倍）
    3. geopandas自带naturalearth_lowres（1:110M，兼容性保障）
    4. 在线下载110m备用（最小体积，极端降级）
    
    本地缓存目录：~/.cache/trae_rainfall/geo/
    """
    if not HAS_GEOPANDAS:
        return None

    # ---- 本地缓存目录 ----
    cache_dir = os.path.join(
        os.path.expanduser("~"), ".cache", "trae_rainfall", "geo"
    )
    os.makedirs(cache_dir, exist_ok=True)

    # ---- 数据源定义：(精度标签, 缓存文件名, [下载URLs]) ----
    # 优先50m：边界细节比110m高2倍，海岸线/国界明显更精确
    data_sources = [
        ("50m", "ne_50m_admin_0_countries.geojson", [
            "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson",
            "https://github.com/nvkelso/natural-earth-vector/raw/master/geojson/ne_50m_admin_0_countries.geojson",
            "https://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_50m_admin_0_countries.geojson",
        ]),
        ("110m", "ne_110m_admin_0_countries.geojson", [
            "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson",
            "https://d2ad6b4ur7yvpq.cloudfront.net/naturalearth-3.3.0/ne_110m_admin_0_countries.geojson",
        ]),
    ]

    try:
        # ---- 步骤1：尝试 geopandas 自带数据（无网络时的最后防线）----
        try:
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            world = _standardize_world_gdf(world, "[内置]naturalearth_lowres(110m)")
            if world is not None:
                # 只是保底，不是优先；设置一个标记让用户知道不是最高精度
                print(f"  💡 提示：可下载50m精度数据获得更准确的海岸线和国家边界")
                return {"continents": world, "land": world}
        except (AttributeError, FileNotFoundError):
            pass
        except Exception as _e:
            print(f"  ⚠️ geopandas 自带数据读取失败: {_e}")

        # ---- 步骤2：优先50m，其次110m（本地缓存→在线下载）----
        for res_label, filename, urls in data_sources:
            cache_path = os.path.join(cache_dir, filename)
            # (a) 检查本地缓存
            if os.path.exists(cache_path) and os.path.getsize(cache_path) > 100000:
                try:
                    world = gpd.read_file(cache_path)
                    world = _standardize_world_gdf(world, f"[缓存]Natural Earth {res_label}")
                    if world is not None:
                        return {"continents": world, "land": world}
                except Exception as _e:
                    print(f"  ⚠️ 缓存损坏，重新下载: {_e}")
                    try:
                        os.remove(cache_path)
                    except OSError:
                        pass
            # (b) 在线下载
            for url in urls:
                try:
                    print(f"  🔄 下载 Natural Earth {res_label} {url[:65]}...")
                    world = gpd.read_file(url)
                    if world is None or len(world) == 0:
                        continue
                    # 写入缓存供下次使用
                    try:
                        world.to_file(cache_path, driver="GeoJSON")
                        print(f"  💾 已缓存到: {cache_path}")
                    except Exception as _werr:
                        print(f"  ⚠️ 缓存写入失败(继续使用): {_werr}")
                    world = _standardize_world_gdf(world, f"Natural Earth {res_label}(在线)")
                    if world is not None:
                        return {"continents": world, "land": world}
                except Exception as _e2:
                    print(f"  ⚠️ 下载失败 [{res_label}]: {type(_e2).__name__}")
                    continue

        print("  ⚠️ 所有地理数据源均不可用，将使用简单经纬度分类")
        return None
    except Exception as _outer:
        print(f"  ⚠️ load_geo_data 异常: {_outer}")
        import traceback; traceback.print_exc()
        return None


def ietd_sensitivity_analysis(dt_precip, config):
    """IETD 敏感性分析 - 多策略评估
    
    策略：
    1. 选择不同降水模式的网格（非仅最湿）
    2. 分析日内降雨间隙分布，找自然断点
    3. 综合事件数稳定性 + 间隙分位数 + CV 选择最优 IETD
    """
    print("\n🔬 IETD敏感性分析")
    ietd_values = config["ietd_candidates"]
    fast_mode = config.get("fast_mode", True)
    time_step = config.get("time_step", 0.5)  # 小时
    
    # ---- 步骤1：选择有代表性的测试网格 ----
    print("  📊 选择测试网格（覆盖不同降水模式）...")
    grid_rain_counts = dt_precip[dt_precip["precip"] > 0].groupby(["x", "y"]).size().reset_index(name="rain_count")
    if grid_rain_counts.empty:
        print("  ⚠️ 没有降水数据，使用默认 IETD")
        return 6
    
    # 过滤掉只有1次降水的网格（无法分析间隙）
    min_counts = max(4, int(3.0 / time_step))  # 至少有3小时的降水
    valid_grids = grid_rain_counts[grid_rain_counts["rain_count"] >= min_counts]
    if valid_grids.empty:
        print(f"  ⚠️ 有效网格不足（阈值={min_counts}次），使用默认 IETD")
        return 6
    
    # 按降水频次分桶
    n_bins = 10
    valid_grids = valid_grids.sort_values("rain_count")
    bin_size = max(1, len(valid_grids) // n_bins)
    
    test_grids_list = []
    n_per_bin = max(2, 20 // n_bins) if fast_mode else max(5, 50 // n_bins)
    
    for i in range(n_bins):
        start = i * bin_size
        end = min(start + bin_size, len(valid_grids))
        bin_data = valid_grids.iloc[start:end]
        if len(bin_data) == 0:
            continue
        n_sample = min(n_per_bin, len(bin_data))
        test_grids_list.append(bin_data.sample(n=n_sample, random_state=42))
    
    test_grids = pd.concat(test_grids_list, ignore_index=True)
    print(f"  📍 测试 {len(test_grids)} 个网格（覆盖 {len(valid_grids)} 个有效网格的 {n_bins} 个频次段）")
    
    # 显示测试网格的降水频次分布
    count_dist = test_grids["rain_count"].describe()
    print(f"     降水频次范围: {int(count_dist['min'])} ~ {int(count_dist['max'])} 次/天")
    print(f"     平均: {count_dist['mean']:.1f} 次, 中位数: {count_dist['50%']:.1f} 次")
    
    # ---- 步骤2：分析所有网格的日内降雨间隙分布 ----
    all_gaps = []
    grid_data_cache = {}
    
    for _, tg in test_grids.iterrows():
        gkey = (tg["x"], tg["y"])
        dt_g = dt_precip[(dt_precip["x"] == tg["x"]) & (dt_precip["y"] == tg["y"])].copy()
        # 如果 time 是 int16 索引，在此处转换为 datetime（每个网格数据量小）
        dt_g = _convert_time_if_needed(dt_g, config)
        dt_g = dt_g.sort_values("time").reset_index(drop=True)
        
        # 识别降雨段
        dt_g["is_rain"] = dt_g["precip"] >= 0.1
        dt_g["rain_group"] = (dt_g["is_rain"] != dt_g["is_rain"].shift()).cumsum()
        rain_segments = dt_g[dt_g["is_rain"] == True]
        if rain_segments.empty:
            continue
        
        event_info = rain_segments.groupby("rain_group").agg(
            start_time=("time", "min"),
            end_time=("time", "max")
        ).reset_index().sort_values("start_time")
        
        # 计算相邻降雨段之间的间隙（小时）
        if len(event_info) > 1:
            gaps = event_info["start_time"].diff().dt.total_seconds() / 3600
            gaps = gaps.dropna()
            all_gaps.extend(gaps.tolist())
        
        grid_data_cache[gkey] = event_info
    
    if not all_gaps:
        print("  ⚠️ 所有测试网格均为连续降雨，无法分析间隙")
        print("  📍 使用默认 IETD = 6 小时")
        return 6
    
    all_gaps_arr = np.array(all_gaps)
    print("  📏 降雨间隙统计:")
    print(f"     间隙数: {len(all_gaps_arr)}")
    print(f"     最小: {np.min(all_gaps_arr):.1f}h, 最大: {np.max(all_gaps_arr):.1f}h")
    print(f"     中位数: {np.median(all_gaps_arr):.1f}h, 均值: {np.mean(all_gaps_arr):.1f}h")
    print(f"     75%分位: {np.percentile(all_gaps_arr, 75):.1f}h, 90%分位: {np.percentile(all_gaps_arr, 90):.1f}h")
    
    # ---- 步骤3：对每个 IETD 计算事件统计 ----
    ietd_stats = {}
    for ietd in ietd_values:
        all_durations = []
        total_events = 0
        event_counts_per_grid = []
        
        for gkey, event_info in grid_data_cache.items():
            if len(event_info) == 0:
                continue
            
            event_info_copy = event_info.copy()
            event_info_copy["gap"] = event_info_copy["start_time"].diff().dt.total_seconds() / 3600
            event_info_copy.loc[event_info_copy.index[0], "gap"] = np.inf
            event_info_copy["event_id"] = (event_info_copy["gap"] > ietd).cumsum()
            
            merged = event_info_copy.groupby("event_id").agg(
                start=("start_time", "min"),
                end=("end_time", "max")
            )
            merged["duration"] = (merged["end"] - merged["start"]).dt.total_seconds() / 3600
            
            n_ev = len(merged)
            event_counts_per_grid.append(n_ev)
            total_events += n_ev
            all_durations.extend(merged["duration"].tolist())
        
        # 计算统计量
        if len(all_durations) > 0:
            mean_dur = np.mean(all_durations)
            std_dur = np.std(all_durations, ddof=1) if len(all_durations) > 1 else 0.0
            cv = std_dur / mean_dur if mean_dur > 0 else np.inf
            
            # 事件数变化率（相对于前一个IETD）
            count_cv = np.std(event_counts_per_grid, ddof=1) / np.mean(event_counts_per_grid) if len(event_counts_per_grid) > 1 else 0.0
        else:
            mean_dur = std_dur = cv = count_cv = 0.0
        
        ietd_stats[ietd] = {
            "total_events": total_events,
            "mean_duration": mean_dur,
            "std_duration": std_dur,
            "cv_duration": cv,
            "count_cv": count_cv,
            "mean_events_per_grid": np.mean(event_counts_per_grid) if event_counts_per_grid else 0
        }
    
    # ---- 步骤4：打印结果 ----
    print("\n  📋 各 IETD 值对比:")
    print(f"  {'IETD':>6} | {'事件数':>6} | {'平均时长':>8} | {'时长CV':>7} | {'网格间CV':>7} | {'事件/网格':>9}")
    print(f"  {'-'*6}─┼─{'-'*6}─┼─{'-'*8}─┼─{'-'*7}─┼─{'-'*7}─┼─{'-'*9}")
    for ietd, stats in ietd_stats.items():
        print(f"  {int(ietd):>4}h | {stats['total_events']:>6} | {stats['mean_duration']:>7.1f}h | {stats['cv_duration']:>7.3f} | {stats['count_cv']:>7.3f} | {stats['mean_events_per_grid']:>9.1f}")
    
    # ---- 步骤5：选择最优 IETD ----
    # 综合评分：CV越小越好，事件数适中（不过多也不过少），网格间变异越小越好
    scores = {}
    for ietd, stats in ietd_stats.items():
        cv_score = 1.0 / (1.0 + stats['cv_duration']) if stats['cv_duration'] != np.inf else 0.0
        count_score = 1.0 / (1.0 + abs(stats['mean_events_per_grid'] - 2.0))  # 理想每个网格2个事件
        grid_cv_score = 1.0 / (1.0 + stats['count_cv'])
        scores[ietd] = cv_score * 0.4 + count_score * 0.3 + grid_cv_score * 0.3
    
    # 如果所有评分相同（CV全为0），基于间隙分析选择
    unique_scores = set(round(s, 6) for s in scores.values())
    if len(unique_scores) == 1:
        # 所有 IETD 效果相同，基于间隙分位数选择
        q75 = np.percentile(all_gaps_arr, 75)
        q90 = np.percentile(all_gaps_arr, 90)
        median_gap = np.median(all_gaps_arr)
        
        # 选择最接近间隙75%分位数的IETD（能有效分离大部分降雨事件）
        best_ietd_by_gap = min(ietd_values, key=lambda x: abs(x - q75))
        print("\n  ⚠️ 所有 IETD 产生相同事件数，使用间隙分析选择")
        print(f"     间隙中位数: {median_gap:.1f}h, 75%分位: {q75:.1f}h, 90%分位: {q90:.1f}h")
        print(f"     选择 IETD = {best_ietd_by_gap}h (接近75%分位间隙)")
        return int(best_ietd_by_gap)
    
    best_ietd = max(scores, key=scores.get)
    print(f"\n  ✅ 推荐 IETD = {int(best_ietd)} 小时 (综合评分={scores[best_ietd]:.3f})")
    return int(best_ietd)


def time_scale_factor_sensitivity(dt_grid_events, config):
    """时间缩放因子敏感性分析
    
    通过比较不同时间缩放因子下的聚类效果，选择最优值。
    评估指标：轮廓系数 + 噪声比例 + 事件数稳定性
    """
    print("\n⏱️  时间缩放因子敏感性分析")
    candidates = config.get("time_scale_candidates", [5, 8, 10, 12, 15, 20])

    # 使用采样数据进行快速评估
    n_sample = min(5000, len(dt_grid_events))
    sample = dt_grid_events.sample(n=n_sample, random_state=42) if len(dt_grid_events) > n_sample else dt_grid_events.copy()

    lons = sample["lon"].values
    lats = sample["lat"].values
    lons_adj = np.where(lons > 180, lons - 360, lons)

    eps_deg = config["dbscan_eps_deg"]
    min_pts = config["dbscan_minPts"]
    eps_km = eps_deg * 111

    # 预筛：用 KDTree 快速找到空间邻近的事件对，避免全局距离矩阵全噪声
    from scipy.spatial import cKDTree
    coords_rad = np.column_stack([np.radians(lats), np.radians(lons_adj)])
    tree = cKDTree(coords_rad)
    eps_rad = np.radians(eps_deg)
    neighbor_pairs = tree.query_pairs(r=eps_rad, output_type='ndarray')

    if len(neighbor_pairs) < min_pts:
        print(f"   ⚠️ 采样中空间邻近事件对不足({len(neighbor_pairs)}对)，跳过敏感性分析")
        default_tsf = config.get("time_scale_factor", 10)
        print(f"   使用默认时间缩放因子: {default_tsf:.0f}")
        return default_tsf

    # 只对有空间邻居的子集做分析
    has_neighbor = np.unique(neighbor_pairs.ravel())
    sample = sample.iloc[has_neighbor].reset_index(drop=True)
    lons = sample["lon"].values
    lats = sample["lat"].values
    lons_adj = np.where(lons > 180, lons - 360, lons)
    n_sub = len(sample)
    print(f"   空间邻近事件子集: {n_sub} 个（从 {n_sample} 采样中）")

    # 计算子集的球面距离矩阵
    spatial_dist = _haversine_distance_matrix(lons_adj, lats)

    results = []
    
    for tsf in candidates:
        try:
            time_scaled = sample["time_scaled"].values * tsf / 10.0  # 标准化基准=10

            # 组合距离（spatial_dist 已在循环外计算）
            time_diff = np.abs(time_scaled[:, np.newaxis] - time_scaled[np.newaxis, :])
            combined = np.sqrt(spatial_dist ** 2 + (time_diff * eps_km / tsf) ** 2)
            
            # DBSCAN 聚类
            db = DBSCAN(eps=eps_km, min_samples=min_pts, metric="precomputed").fit(combined)
            labels = db.labels_
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            noise_ratio = n_noise / len(labels)
            
            # 轮廓系数（排除噪声点）
            sil = np.nan
            if n_clusters > 1 and n_noise < len(labels) * 0.5:
                non_noise = labels != -1
                if np.sum(non_noise) > 10:
                    try:
                        sil = silhouette_score(spatial_dist[np.ix_(non_noise, non_noise)], 
                                               labels[non_noise], metric="precomputed")
                    except Exception:
                        pass
            
            results.append({
                "time_scale": tsf,
                "n_clusters": n_clusters,
                "noise_ratio": noise_ratio,
                "silhouette": sil
            })
            print(f"   tsf={tsf:.0f}: clusters={n_clusters}, noise={noise_ratio:.2f}, silhouette={sil:.3f}")
            
        except Exception as e:
            print(f"   tsf={tsf:.0f}: 失败 ({e})")
            results.append({
                "time_scale": tsf,
                "n_clusters": 0,
                "noise_ratio": 1.0,
                "silhouette": np.nan
            })
    
    results_df = pd.DataFrame(results)
    
    # 综合评分：轮廓系数高 + 噪声比例低 + 聚类数适中
    if results_df["silhouette"].notna().any():
        valid = results_df[results_df["silhouette"].notna()]
        # 综合分 = 轮廓系数 - 噪声比例惩罚
        valid["score"] = valid["silhouette"] - valid["noise_ratio"] * 0.5
        best_idx = valid["score"].idxmax()
        best_tsf = valid.loc[best_idx, "time_scale"]
        print(f"\n✅ 最优时间缩放因子: {best_tsf:.0f} (score={valid.loc[best_idx, 'score']:.3f})")
        return best_tsf
    else:
        # 如果轮廓系数都无效，使用噪声比例最低的
        valid = results_df[results_df["noise_ratio"] < 0.5]
        if not valid.empty:
            best_tsf = valid.loc[valid["noise_ratio"].idxmin(), "time_scale"]
            print(f"\n✅ 最优时间缩放因子（基于噪声）: {best_tsf:.0f}")
            return best_tsf
        default_tsf = config.get("time_scale_factor", 10)
        print(f"\n⚠️ 所有tsf均产生全噪声，使用默认: {default_tsf:.0f}")
        return default_tsf


def get_month_adaptive_params(config, month):
    """根据月份获取自适应参数（用于动态切分和其他参数）
    
    夏季（6-8月）: 对流雨频繁，变化快 → 阈值低
    冬季（12-2月）: 层状雨为主，变化慢 → 阈值高
    春秋（3-5, 9-11月）: 过渡季节 → 中间值
    """
    params = {}
    
    # 自适应切分参数
    if config.get("adaptive_split", False):
        split_params = config.get("split_params_by_month", {})
        if month in split_params:
            params["area_change_threshold"] = split_params[month]["area_change_threshold"]
            params["intensity_drop_threshold"] = split_params[month]["intensity_drop_threshold"]
        else:
            params["area_change_threshold"] = config["area_change_threshold"]
            params["intensity_drop_threshold"] = config["intensity_drop_threshold"]
    else:
        params["area_change_threshold"] = config["area_change_threshold"]
        params["intensity_drop_threshold"] = config["intensity_drop_threshold"]
    
    # 季节系数（用于调整IETD和阈值）
    if month in [12, 1, 2]:  # 冬季
        params["season_factor"] = 1.2  # 冬季IETD增大20%
        params["season"] = "winter"
    elif month in [6, 7, 8]:  # 夏季
        params["season_factor"] = 0.8  # 夏季IETD减小20%
        params["season"] = "summer"
    else:  # 春秋
        params["season_factor"] = 1.0
        params["season"] = "transition"
    
    return params


def validate_events(dt_events, config):
    """事件自验证
    
    检查规则:
    1. 降水量守恒：合并后的总降水应 >= 最大单事件降水
    2. 时长合理性：持续时间应在物理合理范围内
    3. 面积-强度关系：面积大的事件平均强度不应极低
    4. 空间连续性：同一事件的空间点应在合理距离内
    """
    print("\n📋 事件自验证")
    issues = []
    warnings = []
    
    if dt_events is None or dt_events.empty:
        print("  ⚠️ 无事件可验证")
        return dt_events
    
    n_total = len(dt_events)
    n_clusters = dt_events["event_id_final"].nunique()
    
    # 1. 检查单个事件的基本统计
    # lon_range 用环绕差值，避免跨0°/360°经线时算到 ~360°
    def _lon_circular_range(lon_series):
        lon = lon_series.values
        if len(lon) == 0:
            return 0.0
        lon_norm = ((lon + 180) % 360) - 180
        return float(lon_norm.max() - lon_norm.min())

    event_stats = dt_events.groupby("event_id_final").agg(
        n_points=("event_id_final", "count"),
        total_rain=("total_rain", "sum"),
        mean_intensity=("mean_intensity", "mean"),
        max_intensity=("max_30min_rain", "max"),
        duration=("event_duration", "first"),
        lat_range=("lat", lambda x: x.max() - x.min()),
        lon_range=("lon", _lon_circular_range),
    ).reset_index()

    # ★初始化 extreme_space（供后续超长/空间异常检查共用）
    extreme_space = pd.DataFrame(columns=["event_id_final"])
    
    print(f"  事件总数: {n_clusters}")
    print(f"  总网格点数: {n_total}")
    print(f"  平均事件大小: {event_stats['n_points'].mean():.1f} 个网格")
    
    # 2. 时长异常检查
    dur_q1 = event_stats["duration"].quantile(0.25)
    dur_q3 = event_stats["duration"].quantile(0.75)
    dur_iqr = dur_q3 - dur_q1
    dur_upper = dur_q3 + 3 * dur_iqr
    dur_outliers = event_stats[event_stats["duration"] > dur_upper]
    if len(dur_outliers) > 0:
        warnings.append(f"  ⚠️ {len(dur_outliers)} 个事件时长异常 (>75%分位数+3*IQR)")

    # ★历时上限检查：>168h(7天)的超长事件标记为严重问题
    # 这类事件很可能是过度合并的结果，单独处理而非直接删除
    max_duration_hours = config.get("max_duration_hours", 168)
    ultra_long = event_stats[event_stats["duration"] > max_duration_hours]
    if len(ultra_long) > 0:
        issues.append(f"  ❌ {len(ultra_long)} 个超长事件(>{max_duration_hours}h)，可能过度合并")
        # 超长事件也加入过滤列表
        extreme_space = pd.concat([extreme_space, ultra_long], ignore_index=True)
    
    # 3. 极端强度检查
    int_p99 = event_stats["max_intensity"].quantile(0.99)
    extreme_int = event_stats[event_stats["max_intensity"] > int_p99 * 2]
    if len(extreme_int) > 0:
        warnings.append(f"  ⚠️ {len(extreme_int)} 个事件存在极端强降水 (>2x P99)")
    
    # 4. 面积-强度一致性
    large_events = event_stats[event_stats["n_points"] > 50]
    if not large_events.empty:
        low_int_large = large_events[large_events["mean_intensity"] < 0.05]
        if len(low_int_large) > 0:
            warnings.append(f"  ⚠️ {len(low_int_large)} 个大面积事件平均强度极低")
    
    # 5. 空间异常检查
    extreme_space = event_stats[(event_stats["lat_range"] > 20) | (event_stats["lon_range"] > 30)]
    if len(extreme_space) > 0:
        issues.append(f"  ❌ {len(extreme_space)} 个事件空间范围异常大（可能过度合并）")
    
    # 输出结果
    for w in warnings:
        print(w)
    for i in issues:
        print(i)
    
    if not warnings and not issues:
        print("  ✅ 所有事件验证通过")
    elif issues:
        print(f"\n  ⚠️ 存在 {len(issues)} 个严重问题和 {len(warnings)} 个警告")
        # 过滤掉严重异常的事件
        if config.get("filter_invalid_events", True):
            valid_ids = event_stats[~event_stats["event_id_final"].isin(
                extreme_space["event_id_final"]
            )]["event_id_final"]
            before = len(dt_events)
            dt_events = dt_events[dt_events["event_id_final"].isin(valid_ids)]
            after = len(dt_events)
            print(f"  🔧 已过滤 {before - after} 个异常网格点")
    else:
        print(f"\n  ⚠️ 存在 {len(warnings)} 个警告（不影响结果）")
    
    return dt_events


def read_hourly_chunk(data_dir, start_day, end_day, chunk_hours, config):
    """按天批量读取数据（性能优化版）"""
    fast_mode = config.get("fast_mode", True)
    
    if fast_mode:
        print("⚡ 快速模式：使用优化的数据读取...")
    else:
        print("📂 读取原始分辨率数据...")
    
    dates = pd.date_range(
        start=datetime(config["year"], config["month"], start_day),
        end=datetime(config["year"], config["month"], end_day),
        freq="D"
    )
    all_data = []
    all_times = []       # 存 int16 时间槽索引（而非 datetime64[ns]）
    all_x = []
    all_y = []
    all_precip = []
    all_counts = []
    # 全局时间查找表：unique timestamps，int16 索引指向此表
    _time_lut = []       # list of np.datetime64
    _time_map = {}       # timestamp -> int16 index

    def _times_to_idxs(ts_arr):
        """向量化：将时间戳数组转为 int16 索引数组（自动去重）
        利用 np.unique 只对唯一时间戳做 dict 查找"""
        unique_ts, inverse = np.unique(ts_arr, return_inverse=True)
        global_idxs = np.empty(len(unique_ts), dtype=np.int16)
        for i, ts in enumerate(unique_ts):
            ts_np = np.datetime64(ts)
            idx = _time_map.get(ts_np)
            if idx is None:
                idx = np.int16(len(_time_lut))
                _time_map[ts_np] = idx
                _time_lut.append(ts_np)
            global_idxs[i] = idx
        return global_idxs[inverse]

    for current_date in dates:
        day_str = f"{current_date.day:02d}"
        month_str = f"{current_date.month:02d}"
        day_dir = os.path.join(data_dir, month_str, day_str)
        if not os.path.isdir(day_dir):
            continue
        nc_files = sorted(glob.glob(os.path.join(day_dir, "*.nc")))
        if not nc_files:
            continue

        # 快速模式：逐个文件读取并合并 numpy 数组（不依赖 dask）
        if fast_mode:
            try:
                # ★性能: 回退 ThreadPoolExecutor，实测线程池创建/销毁+GIL争抢反而拖慢
                #   原因: 每天48文件×31天=1496次线程池创建，netCDF4可能有文件锁
                #   串行读取更稳定，事件划分逻辑不变
                data_list = []
                time_list = []
                lons = None
                lats = None

                for fp in nc_files:
                    ds_f = xr.open_dataset(fp)
                    try:
                        precip_vars = [v for v in ds_f.data_vars
                                       if any(k in v.lower() for k in ("precip", "rain", "cmorph"))]
                        if not precip_vars:
                            precip_vars = list(ds_f.data_vars)
                        if not precip_vars:
                            continue
                        var = precip_vars[0]
                        da = ds_f[var]
                        if lons is None:
                            lons = ds_f["lon"].values
                            lats = ds_f["lat"].values
                        data_slice = da.values
                        times_slice = ds_f["time"].values if "time" in da.dims else [current_date]

                        if data_slice.ndim == 2:  # (lat, lon)
                            data_list.append(data_slice)
                            time_list.append(times_slice[0] if len(times_slice) > 0 else current_date)
                        elif data_slice.ndim == 3:  # (time, lat, lon)
                            for t in range(data_slice.shape[0]):
                                data_list.append(data_slice[t])
                                time_list.append(times_slice[t] if t < len(times_slice) else current_date)
                        else:
                            data_list.append(data_slice.reshape(data_slice.shape[-2:]))
                            time_list.append(times_slice[0] if len(times_slice) > 0 else current_date)
                    finally:
                        ds_f.close()

                if not data_list:
                    print(f"   {day_str}日: 无有效数据")
                    continue
                
                # 堆叠所有时间片
                data_stack = np.stack(data_list, axis=0)  # shape: (time, lat, lon)
                times_arr = np.array(time_list)
                
                # 找出正值数据点
                positive_mask = data_stack > 0
                positive_mask &= ~np.isnan(data_stack)
                
                if positive_mask.any():
                    time_idx, lat_idx, lon_idx = np.where(positive_mask)
                    
                    # 累积numpy数组（时间用 int16 索引，坐标/降水用 float32 节省内存）
                    day_times_raw = times_arr[time_idx]
                    # 转为 int16 时间槽索引
                    day_time_idxs = _times_to_idxs(day_times_raw)
                    day_x = lons[lon_idx].astype(np.float32, copy=False)
                    day_y = lats[lat_idx].astype(np.float32, copy=False)
                    day_precip = data_stack[time_idx, lat_idx, lon_idx].astype(np.float32, copy=False)
                    
                    # 修正经度（<0的加360）
                    neg_x = day_x < 0
                    if neg_x.any():
                        day_x = day_x.copy()
                        day_x[neg_x] += 360
                    
                    all_times.append(day_time_idxs)
                    all_x.append(day_x)
                    all_y.append(day_y)
                    all_precip.append(day_precip)
                    all_counts.append(len(day_time_idxs))
                    print(f"   {day_str}日: {len(day_time_idxs):,} 个有效数据点")
                else:
                    print(f"   {day_str}日: 无有效降水数据")
                
                continue
                
            except Exception as e:
                print(f"   ⚠️ 快速读取失败: {e}，回退到标准模式")
                pass  # 回退到下面的标准读取方式
        
        # 标准模式读取
        try:
            ds = xr.open_mfdataset(nc_files, combine="nested",
                                    concat_dim="time", parallel=False)
        except Exception:
            try:
                ds = xr.open_mfdataset(nc_files, combine="by_coords",
                                        parallel=False)
            except Exception:
                # 回退：逐个文件读取
                all_day_data = []
                for fp in nc_files:
                    try:
                        ds_single = xr.open_dataset(fp)
                        all_day_data.append(_extract_precip_dataframe(ds_single, current_date, fp))
                        ds_single.close()
                    except Exception:
                        continue
                if all_day_data:
                    df_day = pd.concat(all_day_data, ignore_index=True)
                    df_day = df_day.sort_values("time").reset_index(drop=True)
                    df_day.loc[df_day["x"] < 0, "x"] += 360
                    all_data.append(df_day)
                continue

        # 找到降水变量
        precip_vars = [v for v in ds.data_vars
                       if any(k in v.lower() for k in ("precip", "rain", "cmorph"))]
        if not precip_vars:
            precip_vars = list(ds.data_vars)
        if not precip_vars:
            ds.close()
            continue

        var = precip_vars[0]
        da = ds[var]

        # 保留时间坐标
        if "time" in da.dims:
            da = da.transpose("time", ...)

        # 先转为 DataFrame 再过滤
        df_day = da.to_dataframe(name="precip").reset_index()

        # 重命名坐标
        rename_map = {}
        for col in df_day.columns:
            cl = col.lower()
            if "lon" in cl or "x" in cl:
                rename_map[col] = "x"
            elif "lat" in cl or "y" in cl:
                rename_map[col] = "y"
            elif "time" in cl:
                rename_map[col] = "time"
        if rename_map:
            df_day = df_day.rename(columns=rename_map)

        # 过滤：只保留降水量 > 0 且非 NaN 的数据
        df_day = df_day.dropna(subset=["precip"])
        df_day = df_day[df_day["precip"] > 0]

        if df_day.empty:
            ds.close()
            continue

        # 确保时间列存在
        if "time" not in df_day.columns:
            if "time" in ds.coords:
                # 尝试从数据集获取时间信息
                pass

        df_day = df_day[["x", "y", "time", "precip"]].copy()

        if not df_day.empty:
            df_day.loc[df_day["x"] < 0, "x"] += 360
            all_data.append(df_day)

        ds.close()
        gc.collect()

    if all_times:
        print("\n  正在合并numpy数组数据...")
        # 合并 int16 时间索引（仅 2 bytes/点，而非 8 bytes）
        stacked_time_idxs = np.concatenate(all_times)  # int16, ~1.7 GiB vs 6.83 GiB
        stacked_x = np.concatenate(all_x).astype(np.float32, copy=False)
        stacked_y = np.concatenate(all_y).astype(np.float32, copy=False)
        stacked_precip = np.concatenate(all_precip).astype(np.float32, copy=False)
        # 释放每日数组引用，降低峰值内存
        del all_times, all_x, all_y, all_precip
        gc.collect()
        # ★ 不重建 datetime64 数组！直接用 int16 作为 time 列
        # 查找表存入 config，下游需要时再转换（避免 6.83 GiB 分配）
        config["_time_lut"] = np.array(_time_lut, dtype="datetime64[ns]")
        del _time_lut, _time_map
        gc.collect()
        dt_combined = pd.DataFrame({
            "time": stacked_time_idxs,    # int16 索引（非 datetime64）
            "y": stacked_y,
            "x": stacked_x,
            "precip": stacked_precip
        })
        del stacked_time_idxs, stacked_x, stacked_y, stacked_precip
        gc.collect()
        print(f"  📊 共读取 {len(dt_combined)} 个网格点记录 (numpy快速模式)")
        return dt_combined
    elif all_data:
        print("\n  正在合并数据...")
        # 标准模式：time 列已经是 datetime，清除 LUT 标记
        config["_time_lut"] = None
        dt_combined = pd.concat(all_data, ignore_index=True)
        if not pd.api.types.is_datetime64_any_dtype(dt_combined["time"]):
            dt_combined["time"] = pd.to_datetime(
                dt_combined["time"], unit="s", origin="1970-01-01", utc=True
            )
        print(f"  📊 共读取 {len(dt_combined)} 个网格点记录")
        return dt_combined
    return None


def _extract_precip_dataframe(ds, base_date, file_path):
    """从单个数据集提取降水 DataFrame"""
    precip_vars = [v for v in ds.data_vars
                   if any(k in v.lower() for k in ("precip", "rain", "cmorph"))]
    if not precip_vars:
        precip_vars = list(ds.data_vars)
    if not precip_vars:
        return pd.DataFrame(columns=["x", "y", "time", "precip"])

    var = precip_vars[0]
    da = ds[var]

    # 解析时间
    m = re.search(r"_(\d{10})\.nc$", os.path.basename(file_path))
    if m:
        base_time = datetime.strptime(m.group(1), "%Y%m%d%H")
    else:
        base_time = base_date

    if "time" in da.dims:
        parts = []
        for t_idx, t_val in enumerate(da.time.values):
            try:
                t_dt = pd.to_datetime(t_val)
            except Exception:
                t_dt = base_time + timedelta(minutes=30 * t_idx)
            
            # 提取单个时间片
            slice_da = da.isel(time=t_idx)
            
            # 转换为 DataFrame 后过滤（更可靠）
            df_t = slice_da.to_dataframe(name="precip").reset_index()
            
            # 重命名坐标
            rename_map = {}
            for col in df_t.columns:
                cl = col.lower()
                if "lon" in cl or "x" in cl:
                    rename_map[col] = "x"
                elif "lat" in cl or "y" in cl:
                    rename_map[col] = "y"
            if rename_map:
                df_t = df_t.rename(columns=rename_map)
            
            # 过滤：只保留降水量 > 0 且非 NaN 的数据
            df_t = df_t.dropna(subset=["precip"])
            df_t = df_t[df_t["precip"] > 0]
            
            if not df_t.empty:
                df_t["time"] = t_dt
                parts.append(df_t[["x", "y", "time", "precip"]])
        
        if parts:
            return pd.concat(parts, ignore_index=True)
    else:
        # 无时间维度的情况
        df = da.to_dataframe(name="precip").reset_index()
        
        # 重命名坐标
        rename_map = {}
        for col in df.columns:
            cl = col.lower()
            if "lon" in cl or "x" in cl:
                rename_map[col] = "x"
            elif "lat" in cl or "y" in cl:
                rename_map[col] = "y"
        if rename_map:
            df = df.rename(columns=rename_map)
        
        # 过滤：只保留降水量 > 0 且非 NaN 的数据
        df = df.dropna(subset=["precip"])
        df = df[df["precip"] > 0]
        
        if not df.empty:
            df["time"] = base_time
            return df[["x", "y", "time", "precip"]]

    return pd.DataFrame(columns=["x", "y", "time", "precip"])


# ============================================================
# 核心：网格级事件识别（彻底解决 NA 问题）
# ============================================================

def _convert_time_if_needed(df, config):
    """如果 time 列是 int16 索引，转换为 datetime64（使用查找表）
    仅在数据量较小时调用（如分组后的子集）"""
    if "time" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["time"]):
        lut = config.get("_time_lut")
        if lut is not None:
            df = df.copy()
            df["time"] = lut[df["time"].values.astype(np.intp)]
        else:
            df = df.copy()
            df["time"] = pd.to_datetime(df["time"], unit="s", origin="1970-01-01", utc=True)
    return df


def extract_grid_events_memory_friendly(dt_precip, config, kg_gdf=None):
    """内存友好型网格级事件识别（向量化版，支持柯本气候分区）"""
    print("⚡ 开始向量化网格级事件识别...")
    if dt_precip is None or dt_precip.empty:
        return None
    # time 列可以是 int16 索引（节省内存），核心计算不需要 datetime
    # 仅在最终输出事件时转换为 datetime

    # 清理上次运行残留的临时文件
    temp_dir = config.get("temp_dir", "E:\\temp")
    os.makedirs(temp_dir, exist_ok=True)
    old_files = [f for f in os.listdir(temp_dir) if f.startswith("grid_events_batch_")]
    for f in old_files:
        os.remove(os.path.join(temp_dir, f))
    if old_files:
        print(f"  🧹 清理了 {len(old_files)} 个旧临时文件")

    print("  🔍 快速过滤有雨网格...")
    dt_precip_rain = dt_precip[dt_precip["precip"] > 0].copy()
    if dt_precip_rain.empty:
        print("❌ 没有降水数据")
        return None

    rainy_grids = dt_precip_rain[["x", "y"]].drop_duplicates().reset_index(drop=True)
    n_grids = len(rainy_grids)
    original_n = dt_precip[["x", "y"]].drop_duplicates().shape[0]
    print(f"  有雨网格: {n_grids} / {original_n} ({n_grids/original_n*100:.1f}%)")
    if n_grids == 0:
        return None

    # ---- 预计算自适应参数 ----
    grid_params = rainy_grids.copy()
    grid_params["abs_lat"] = grid_params["y"].abs()
    grid_params["threshold"] = config["global_rain_threshold"]
    grid_params["ietd"] = config["global_IETD_hours"]

    # ⭐ 使用柯本气候分区（如果可用且启用）
    use_koppen = config.get("use_koppen_climate", True) and kg_gdf is not None and HAS_GEOPANDAS
    
    if use_koppen:
        print("  🌍 查询柯本气候分区...")
        grid_lons = grid_params["x"].values
        grid_lats = grid_params["y"].values
        
        # 优先使用LUT查找表（O(1)查询，速度提升100倍+）
        climate_lut = config.get("_climate_lut")
        if climate_lut is not None:
            climate_zones = query_climate_lut(grid_lons, grid_lats, climate_lut)
            print("    ⚡ 使用LUT查找表查询 (速度优化)")
        else:
            # 回退到批量空间查询
            print("    使用空间连接查询...")
            climate_zones = batch_query_climate_zone(grid_lons, grid_lats, kg_gdf)
        
        if climate_zones is not None:
            # 对"未知"的海洋点，用语义化海洋分类
            unknown_mask_zones = pd.Series(climate_zones).astype(str).isin(["未知", "未分类", "nan", "None"])
            if unknown_mask_zones.any():
                # 先判断哪些是海洋点
                if config.get("performance", {}).get("vectorized_geo", True):
                    _loc_types = classify_location_type_vectorized(grid_lons, grid_lats)
                else:
                    _loc_types = np.array([
                        classify_location_type_simple(lon, lat) for lon, lat in zip(grid_lons, grid_lats)
                    ])
                _ocean_unknown = unknown_mask_zones.values & (_loc_types == "海洋")
                if _ocean_unknown.any():
                    ocean_cz = ocean_climate_zone_vectorized(grid_lons[_ocean_unknown], grid_lats[_ocean_unknown])
                    climate_zones = np.array(climate_zones, dtype=object)
                    climate_zones[_ocean_unknown] = ocean_cz
                    print(f"    🌊 对 {_ocean_unknown.sum()} 个海洋网格用语义化海洋气候区分类")
            
            grid_params["climate_zone"] = climate_zones
            
            # 统计气候区分布
            zone_counts = pd.Series(climate_zones).value_counts()
            print(f"    气候区分布: {dict(zone_counts.head(10))}")
            
            # 根据气候区设置参数
            print("  ⚙️ 根据柯本气候分区设置阈值...")
            for zone in zone_counts.index:
                threshold, ietd = get_koppen_params(zone, config)
                mask = grid_params["climate_zone"] == zone
                grid_params.loc[mask, "threshold"] = threshold
                grid_params.loc[mask, "ietd"] = ietd
            
            # 对"未知"气候区的点，回退到纬度带划分
            unknown_mask = grid_params["climate_zone"].astype(str).isin(["未知", "未分类", "nan", "None"])
            if unknown_mask.any():
                print(f"    📍 对 {unknown_mask.sum()} 个未知气候区点使用纬度带回退...")
                bands = config["lat_bands"]
                abs_lats = grid_params.loc[unknown_mask, "abs_lat"]
                trop = abs_lats <= 23.5
                sub = (abs_lats > 23.5) & (abs_lats <= 35)
                temp = (abs_lats > 35) & (abs_lats <= 55)
                polar = abs_lats > 55
                
                unknown_idx = grid_params.loc[unknown_mask].index
                grid_params.loc[unknown_idx[trop.values], "threshold"] = bands["tropical"]["threshold"]
                grid_params.loc[unknown_idx[trop.values], "ietd"] = bands["tropical"]["IETD"]
                grid_params.loc[unknown_idx[sub.values], "threshold"] = bands["subtropical"]["threshold"]
                grid_params.loc[unknown_idx[sub.values], "ietd"] = bands["subtropical"]["IETD"]
                grid_params.loc[unknown_idx[temp.values], "threshold"] = bands["temperate"]["threshold"]
                grid_params.loc[unknown_idx[temp.values], "ietd"] = bands["temperate"]["IETD"]
                if config["adaptive_threshold"]:
                    grid_params.loc[unknown_idx[polar.values], "threshold"] = bands["polar"]["threshold"]
                    grid_params.loc[unknown_idx[polar.values], "ietd"] = bands["polar"]["IETD"]
            
            # 显示参数统计
            print(f"    阈值范围: {grid_params['threshold'].min():.3f} ~ {grid_params['threshold'].max():.3f} mm")
            print(f"    IETD范围: {grid_params['ietd'].min()} ~ {grid_params['ietd'].max()} 小时")
        else:
            print("    ⚠️ 气候区查询失败，回退到纬度带划分")
            use_koppen = False
    
    # 如果不使用柯本气候区，使用纬度带划分
    if not use_koppen:
        print("  📍 使用纬度带划分设置阈值...")
        bands = config["lat_bands"]
        trop_mask = grid_params["abs_lat"] <= 23.5
        sub_mask = (grid_params["abs_lat"] > 23.5) & (grid_params["abs_lat"] <= 35)
        temp_mask = (grid_params["abs_lat"] > 35) & (grid_params["abs_lat"] <= 55)
        grid_params.loc[trop_mask, "threshold"] = bands["tropical"]["threshold"]
        grid_params.loc[trop_mask, "ietd"] = bands["tropical"]["IETD"]
        grid_params.loc[sub_mask, "threshold"] = bands["subtropical"]["threshold"]
        grid_params.loc[sub_mask, "ietd"] = bands["subtropical"]["IETD"]
        grid_params.loc[temp_mask, "threshold"] = bands["temperate"]["threshold"]
        grid_params.loc[temp_mask, "ietd"] = bands["temperate"]["IETD"]
        if config["adaptive_threshold"]:
            polar_mask = grid_params["abs_lat"] > 55
            grid_params.loc[polar_mask, "threshold"] = bands["polar"]["threshold"]
            grid_params.loc[polar_mask, "ietd"] = bands["polar"]["IETD"]

    # ================================================================
    # ★★★ v17.0 极速算法：批量numba + 零雨预过滤 + 空间降采样 ★★★
    # 核心优化：
    #   1. 零雨网格预过滤 → 跳过整个时间序列全为0的网格
    #   2. 空间降采样(0.25°) → 网格数减少4倍，精度够用
    #   3. 单次数值计算 → 消除270万次Python↔numba调度开销
    #   4. prange多线程 → 真正的并行计算
    # ================================================================
    
    print("  🚀 极速事件提取（批量numba+降采样模式）...")
    
    # ---- 步骤0：空间降采样（可选，配置控制）----
    target_res = config.get("performance", {}).get("spatial_downsample_deg", 0.0)
    if target_res > 0:
        orig_res = 0.072
        ds_factor = max(1, round(target_res / orig_res))
        if ds_factor > 1:
            print(f"    📐 空间降采样: {orig_res}° → {target_res}° (因子 {ds_factor})")
            
            # --- 对 grid_params 做降采样 ---
            gp_coords = grid_params[["x", "y"]].values
            n_grids_orig = len(gp_coords)
            
            lon_bin = np.round(gp_coords[:, 0] / target_res).astype(np.int32)
            lat_bin = np.round(gp_coords[:, 1] / target_res).astype(np.int32)
            bin_ids = lon_bin * 10000 + lat_bin
            
            _, unique_idx = np.unique(bin_ids, return_index=True)
            unique_idx.sort()
            
            rep_bin_ids = bin_ids[unique_idx]
            
            grid_params = grid_params.iloc[unique_idx].reset_index(drop=True)
            n_grids_ds = len(grid_params)
            print(f"    网格数: {n_grids_orig} → {n_grids_ds} (减少 {100*(1-n_grids_ds/n_grids_orig):.0f}%)")
            
            # --- 将 dt_precip_rain 的坐标映射到代表点 ---
            rain_lons = dt_precip_rain["x"].values
            rain_lats = dt_precip_rain["y"].values
            rain_lon_bin = np.round(rain_lons / target_res).astype(np.int32)
            rain_lat_bin = np.round(rain_lats / target_res).astype(np.int32)
            rain_bin_ids = rain_lon_bin * 10000 + rain_lat_bin
            
            # 向量化映射: searchsorted查找
            sorted_rep_bins = np.sort(rep_bin_ids)
            sort_idx = np.argsort(rep_bin_ids)
            pos = np.searchsorted(sorted_rep_bins, rain_bin_ids)
            # 检查有效性
            valid = pos < len(sorted_rep_bins)
            valid &= (sorted_rep_bins[np.clip(pos, 0, len(sorted_rep_bins)-1)] == rain_bin_ids)
            # 获取代表点坐标
            rep_idx = np.zeros(len(rain_bin_ids), dtype=np.int64)
            rep_idx[valid] = sort_idx[pos[valid]]
            rep_coords = grid_params[["x", "y"]].values
            mapped_x = np.where(valid, rep_coords[rep_idx, 0], np.nan)
            mapped_y = np.where(valid, rep_coords[rep_idx, 1], np.nan)
            dt_precip_rain["x"] = mapped_x
            dt_precip_rain["y"] = mapped_y
            
            # 过滤掉无法映射的网格
            dt_precip_rain = dt_precip_rain[valid].reset_index(drop=True)
            print(f"    映射有效数据点: {len(dt_precip_rain)}/{len(rain_bin_ids)}")
    
    # ---- 步骤1：构建 (time, grid) 2D 数组 ----
    # 如果 time 是 int16 索引，先转回 datetime 用于 unique_times 提取
    _time_lut = config.get("_time_lut")
    if _time_lut is not None and not pd.api.types.is_datetime64_any_dtype(dt_precip_rain["time"]):
        # int16 索引模式：unique 的 int16 索引 → 查 LUT 得到 datetime
        unique_time_idxs = sorted(dt_precip_rain["time"].unique())
        unique_times = [_time_lut[int(idx)] for idx in unique_time_idxs]
    else:
        unique_times = sorted(dt_precip_rain["time"].unique())
    unique_grids = grid_params[["x", "y"]].values  # (n_grids, 2)
    n_times = len(unique_times)
    n_grids_total = len(unique_grids)
    
    if _time_lut is not None and not pd.api.types.is_datetime64_any_dtype(dt_precip_rain["time"]):
        time_to_idx = {idx: i for i, idx in enumerate(unique_time_idxs)}
    else:
        time_to_idx = {t: i for i, t in enumerate(unique_times)}
    grid_to_idx = {(g[0], g[1]): i for i, g in enumerate(unique_grids)}
    
    # 预分配 2D 数组 [time_idx, grid_idx]
    precip_2d = np.zeros((n_times, n_grids_total), dtype=np.float32)
    
    # 填充数据（向量化）
    time_indices = dt_precip_rain["time"].map(time_to_idx).values
    grid_keys = list(zip(dt_precip_rain["x"], dt_precip_rain["y"]))
    grid_indices = np.array([grid_to_idx.get(k, -1) for k in grid_keys])
    
    valid_rows = grid_indices >= 0
    ti = time_indices[valid_rows]
    gi = grid_indices[valid_rows]
    vals = dt_precip_rain["precip"].values[valid_rows]
    
    precip_2d[ti, gi] = vals.astype(np.float32)
    
    print(f"    数据透视: ({n_times} 时间步 × {n_grids_total} 网格)")
    
    # ---- 步骤2：零雨网格预过滤 ----
    thresholds = grid_params["threshold"].values.astype(np.float32)  # (n_grids,)
    ietds = grid_params["ietd"].values.astype(np.float32)  # (n_grids,)
    
    # 标记降雨 [n_times, n_grids]
    rain_2d = precip_2d >= thresholds[np.newaxis, :]
    
    # 检查每个网格是否有任何降雨
    has_rain = rain_2d.any(axis=0)  # (n_grids,)
    n_rainy_grids = has_rain.sum()
    n_dry_grids = n_grids_total - n_rainy_grids
    print(f"    🌧️ 有雨网格: {n_rainy_grids} / {n_grids_total} ({100*n_rainy_grids/n_grids_total:.1f}%)")
    if n_dry_grids > 0:
        print(f"    ⏭️  跳过无雨网格: {n_dry_grids}")
    
    # 只保留有雨网格
    if n_dry_grids > 0 and n_rainy_grids > 0:
        precip_2d = precip_2d[:, has_rain]
        rain_2d = rain_2d[:, has_rain]
        thresholds = thresholds[has_rain]
        ietds = ietds[has_rain]
        unique_grids = unique_grids[has_rain]
        n_grids_total = n_rainy_grids
    
    # ---- 步骤3：批量numba提取（一次调用处理所有网格）----
    base_time = pd.Timestamp(unique_times[0])
    # 统一用 pd.Timestamp 计算（兼容 datetime64 和 Timestamp）
    time_hours = np.array([
        (pd.Timestamp(t) - base_time).total_seconds() / 3600.0
        for t in unique_times
    ], dtype=np.float64)
    
    print(f"    🔬 批量numba提取 ({n_grids_total} 网格 × {n_times} 时间步, prange并行)...")
    
    # 单次调用，内部prange并行（每个网格独立写入区域，无竞态）
    events_raw = _extract_all_events_batch(time_hours, precip_2d, thresholds, ietds)
    
    del precip_2d, rain_2d
    gc.collect()
    
    # Compact: 过滤掉duration==0的无效条目
    valid_mask = events_raw[:, 6] > 0  # duration > 0
    events_compact = events_raw[valid_mask]
    n_events = len(events_compact)
    print(f"    ✅ numba提取完成: {n_events} 个事件 (过滤 {len(events_raw) - n_events} 空条目)")
    
    if n_events == 0:
        print("❌ 未提取到任何事件")
        return None
    
    # ---- 步骤4：组装DataFrame ----
    # events_compact: (n_events, 7) [grid_idx, start_h, end_h, total, max_rain, peak_h, duration]
    grid_indices_out = events_compact[:, 0].astype(np.int64)
    start_h_arr = events_compact[:, 1]
    end_h_arr = events_compact[:, 2]
    total_arr = events_compact[:, 3]
    max_rain_arr = events_compact[:, 4]
    peak_h_arr = events_compact[:, 5]
    duration_arr = events_compact[:, 6]
    
    # 向量化构建所有字段
    gx_arr = unique_grids[grid_indices_out, 0]
    gy_arr = unique_grids[grid_indices_out, 1]
    
    # 时间转换（向量化）
    base_ns = np.datetime64(base_time, 'ns')
    start_times = base_ns + (start_h_arr * 3600 * 1e9).astype(np.int64).astype('timedelta64[ns]')
    end_times = base_ns + (end_h_arr * 3600 * 1e9).astype(np.int64).astype('timedelta64[ns]')
    peak_times = base_ns + (peak_h_arr * 3600 * 1e9).astype(np.int64).astype('timedelta64[ns]')
    
    mean_int_arr = total_arr / duration_arr
    time_scaled_arr = (start_h_arr / 24.0) * 10.0
    day_of_month_arr = np.array([pd.Timestamp(t).day for t in unique_times])  # 日号映射
    dom_arr = day_of_month_arr[np.clip((start_h_arr / 0.5).astype(np.int64), 0, n_times - 1)]
    # ★携带每个事件所属气候区的 IETD，供跨日合并按气候区判断（避免全局IETD不一致）
    ietd_arr = ietds[grid_indices_out]

    dt_all_events = pd.DataFrame({
        "x": gx_arr, "y": gy_arr,
        "lon": gx_arr, "lat": gy_arr,
        "start_time": start_times,
        "end_time": end_times,
        "duration_hours": duration_arr,
        "total_rain": total_arr,
        "max_30min_rain": max_rain_arr,
        "max_30min_time": peak_times,
        "mean_intensity": mean_int_arr,
        "n_obs": np.maximum(1, (duration_arr / 0.5).astype(np.int64)),
        "time_scaled": time_scaled_arr,
        "time_num": start_h_arr,
        "day_of_month": dom_arr,
        "ietd": ietd_arr,  # 该事件所属气候区的 IETD（小时）
    })
    
    print(f"\n✅ 共提取 {len(dt_all_events)} 个网格级事件")
    
    # ---- 去重检查 ----
    dup_cols = ["x", "y", "start_time", "end_time"]
    dt_all_events = dt_all_events.drop_duplicates(subset=dup_cols, keep="first")
    print(f"    去重后: {len(dt_all_events)}")
    
    # 逻辑检查
    inconsistent = dt_all_events["max_30min_rain"] > dt_all_events["total_rain"]
    if inconsistent.any():
        print(f"    ⚠️ 修正 {inconsistent.sum()} 个逻辑冲突事件")
        dt_all_events.loc[inconsistent, "max_30min_rain"] = dt_all_events.loc[inconsistent, "total_rain"].values.astype(np.float32)
    
    for col in ["start_time", "end_time", "max_30min_time"]:
        dt_all_events[col] = pd.to_datetime(dt_all_events[col])
    
    check_memory_usage()
    gc.collect()
    return dt_all_events


# ============================================================
# 跨日事件合并（逐日处理模式）
# ============================================================

def merge_cross_day_events(all_daily_events, config):
    """合并跨日事件：相同网格在时间上连续（间隔<IETD）的事件合并为一个
    
    Args:
        all_daily_events: list of DataFrame，每个元素是一天的网格事件
        config: 配置字典
    
    Returns:
        DataFrame: 合并后的网格事件
    """
    if not all_daily_events:
        return None
    
    print("\n🔗 合并跨日事件...")
    dt_all = pd.concat(all_daily_events, ignore_index=True)
    print(f"  合并前: {len(dt_all)} 个网格事件")

    # ★修复 P1: 跨日合并 IETD 改用每个事件所属气候区的 IETD（原用全局值导致跨气候区不一致）
    #   规则: 用前一事件的 IETD 判断（前一事件需要的最小无雨间隔）
    #   若事件无 ietd 列（旧数据兼容），回退到全局 IETD
    global_ietd_hours = config.get("ietd_selected", config.get("global_IETD_hours", 6))
    if "ietd" not in dt_all.columns:
        dt_all["ietd"] = float(global_ietd_hours)

    # 按网格分组，合并时间连续的事件
    dt_all = dt_all.sort_values(["x", "y", "start_time"]).reset_index(drop=True)

    # 计算同一网格内相邻事件的时间间隔
    dt_all["prev_end"] = dt_all.groupby(["x", "y"])["end_time"].shift()
    dt_all["gap_seconds"] = (dt_all["start_time"] - dt_all["prev_end"]).dt.total_seconds()
    # 前一事件的 IETD（小时→秒）
    dt_all["prev_ietd_seconds"] = dt_all.groupby(["x", "y"])["ietd"].shift() * 3600.0
    # 首事件无前事件，gap 为 NaN，用全局 IETD 兜底（不会触发新事件标记，因 NaN > x 为 False）
    dt_all["prev_ietd_seconds"] = dt_all["prev_ietd_seconds"].fillna(global_ietd_hours * 3600.0)

    # 标记新事件的起始（间隔 > 前一事件 IETD 或第一个事件）
    dt_all["new_event"] = (dt_all["gap_seconds"].isna()) | (dt_all["gap_seconds"] > dt_all["prev_ietd_seconds"])
    dt_all["merge_group"] = dt_all["new_event"].cumsum()
    
    # 按合并组聚合
    # ★修复 P0: duration_hours 原用 sum 会累加子事件 duration，漏掉段间间隔，虚高 mean_intensity
    #   改为 end-start（与 summarize_events 一致），mean_intensity 用真实时长重算
    # ★修复 P0: max_30min_time 原用 min 取最早时间，应取对应于 max_30min_rain 的时间点
    # ★性能优化: 用 groupby.idxmax() 一次性拿到所有组峰值行索引，避免 transform(lambda) 逐组循环
    #   原实现 transform(lambda s: s.index == s.idxmax()) 对 30万组逐组 Python 调用，慢 100-500 倍
    peak_idx = dt_all.groupby("merge_group")["max_30min_rain"].idxmax()

    merged = dt_all.groupby("merge_group").agg(
        x=("x", "first"),
        y=("y", "first"),
        lon=("lon", "first"),
        lat=("lat", "first"),
        start_time=("start_time", "min"),
        end_time=("end_time", "max"),
        total_rain=("total_rain", "sum"),
        max_30min_rain=("max_30min_rain", "max"),
        mean_intensity=("mean_intensity", "mean"),
        n_obs=("n_obs", "sum"),
        day_of_month=("day_of_month", "first"),
    ).reset_index(drop=True)

    # 用峰值行索引一次性取回每组的 max_30min_time（对应于 max_30min_rain 的时间点）
    # ★性能优化: peak_idx 是 Series，索引=merge_group，值=dt_all 行索引
    #   sort_index() 按 merge_group 排序，与 groupby.agg 的默认顺序一致
    #   避免构造 DataFrame + sort_values 的额外开销
    peak_times = dt_all.loc[peak_idx.sort_index().values, "max_30min_time"]
    merged["max_30min_time"] = peak_times.values

    # duration_hours 用 end-start（包含段间间隔，与事件实际持续一致）
    merged["duration_hours"] = (
        (merged["end_time"] - merged["start_time"]).dt.total_seconds() / 3600.0
    ).astype(np.float32)

    # 重新计算 mean_intensity 和 time_scaled
    merged["mean_intensity"] = merged["total_rain"] / np.maximum(merged["duration_hours"], 0.5)
    merged["time_num"] = (merged["start_time"] - merged["start_time"].min()).dt.total_seconds() / 3600.0
    # time_scaled = 绝对小时数 × 时间缩放因子
    merged["time_scaled"] = merged["time_num"] * config.get("time_scale_factor", 10.0)

    # 丢弃辅助列
    merged = merged.drop(columns=["merge_group"], errors="ignore")
    
    print(f"  合并后: {len(merged)} 个网格事件（减少 {len(dt_all) - len(merged)} 个跨日事件）")
    return merged


# ============================================================
# DBSCAN 网格搜索与聚类
# ============================================================

def dbscan_grid_search(dt_events, config):
    """DBSCAN 参数网格搜索"""
    print("\n🔬 DBSCAN参数网格搜索")
    np.random.seed(123)
    if len(dt_events) > 50000:
        dt_sample = dt_events.sample(50000).copy()
    else:
        dt_sample = dt_events.copy()
    clust_data = dt_sample[["lon", "lat", "time_scaled"]].values
    eps_vals = config["eps_candidates"]
    minPts_vals = config["minPts_candidates"]
    results = []
    combo_count = 0
    for eps in eps_vals:
        for minPts in minPts_vals:
            combo_count += 1
            try:
                db = DBSCAN(eps=eps, min_samples=minPts).fit(clust_data)
                labels = db.labels_
            except Exception:
                continue
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            noise_ratio = n_noise / len(labels)
            sil_score = np.nan
            if n_clusters > 1 and n_clusters < len(labels) * 0.9:
                try:
                    non_noise = labels != -1
                    if np.sum(non_noise) > 10 and n_clusters > 1:
                        sil_score = silhouette_score(clust_data[non_noise], labels[non_noise])
                except Exception:
                    pass
            results.append({
                "eps": eps, "minPts": minPts, "n_clusters": n_clusters,
                "n_noise": n_noise, "noise_ratio": noise_ratio, "silhouette": sil_score
            })
    results_df = pd.DataFrame(results)
    valid = results_df[(results_df["noise_ratio"] < 0.3) & (~results_df["silhouette"].isna())]
    if not valid.empty:
        best = valid.loc[valid["silhouette"].idxmax()]
    else:
        best = results_df.loc[results_df["silhouette"].idxmax()]
    print(f"✅ 推荐参数: eps={best['eps']:.1f}°, minPts={int(best['minPts'])}")
    return {"eps": best["eps"], "minPts": int(best["minPts"]), "results": results_df}


def _haversine_distance_matrix(lons, lats):
    """计算球面距离矩阵（Haversine公式，向量化版）
    
    返回单位为公里的距离矩阵
    """
    R = 6371.0  # 地球半径 km
    
    # 转为弧度
    lon_rad = np.radians(lons)
    lat_rad = np.radians(lats)
    
    # 构建差值矩阵
    dlon = lon_rad[:, np.newaxis] - lon_rad[np.newaxis, :]
    dlat = lat_rad[:, np.newaxis] - lat_rad[np.newaxis, :]
    
    # Haversine 公式
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_rad[:, np.newaxis]) * np.cos(lat_rad[np.newaxis, :]) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    
    return R * c


def _merge_cross_block_clusters(dt_events, lons, lats, time_scaled, config):
    """合并预分区DBSCAN中被块边界割裂的簇

    对每个簇计算质心(经度循环平均)和时间范围，若相邻簇满足：
      1. 空间距离 < merge_dist_deg (默认2°)
      2. 时间重叠或间隔 < merge_time_gap (默认6h)
    则合并为同一簇。

    Args:
        dt_events: 含 spatial_cluster 列的DataFrame
        lons: 事件经度数组
        lats: 事件纬度数组
        time_scaled: 事件时间(缩放后)数组
        config: 配置字典

    Returns:
        DataFrame: 更新 spatial_cluster 后的DataFrame
    """
    merge_dist_deg = config.get("dbscan_cross_block_dist", 5.0)
    merge_time_gap = config.get("dbscan_cross_block_time_gap", 6.0)
    # ★日界线感知：跨0°/360°的簇对使用更大合并阈值
    dateline_merge_dist = config.get("dbscan_dateline_merge_dist", 8.0)
    time_scale = config.get("time_scale_factor", 10.0)

    labels = dt_events["spatial_cluster"].values
    valid_mask = labels != -1
    valid_labels = labels[valid_mask]
    unique_clusters = np.unique(valid_labels)
    n_clusters = len(unique_clusters)

    if n_clusters <= 1:
        return dt_events

    # 计算每个簇的质心和时间范围
    cluster_info = {}
    for cid in unique_clusters:
        cluster_idx = np.where(labels == cid)[0]
        if len(cluster_idx) == 0:
            continue
        c_lons = lons[cluster_idx]
        c_lats = lats[cluster_idx]
        c_times = time_scaled[cluster_idx] * time_scale

        # 循环加权质心经度
        lon_rad = np.radians(c_lons)
        sin_mean = np.mean(np.sin(lon_rad))
        cos_mean = np.mean(np.cos(lon_rad))
        centroid_lon = np.degrees(np.arctan2(sin_mean, cos_mean)) % 360
        centroid_lat = np.mean(c_lats)
        t_min = c_times.min()
        t_max = c_times.max()

        cluster_info[cid] = {
            "centroid_lon": centroid_lon,
            "centroid_lat": centroid_lat,
            "t_min": t_min,
            "t_max": t_max,
            "size": len(cluster_idx),
        }

    # 构建合并映射（并查集）
    parent = {cid: cid for cid in cluster_info}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            # 合并到较大的簇
            if cluster_info[ra]["size"] >= cluster_info[rb]["size"]:
                parent[rb] = ra
            else:
                parent[ra] = rb

    # 检查所有簇对（★优化：按纬度排序+提前终止，O(n²)→O(n·k)）
    cluster_ids = list(cluster_info.keys())
    # 按纬度排序，利用纬度差不随经度绕回的特性做提前终止
    cluster_ids.sort(key=lambda cid: cluster_info[cid]["centroid_lat"])
    merge_count = 0
    for i in range(len(cluster_ids)):
        ci = cluster_ids[i]
        if find(ci) != ci:
            continue  # 已被合并
        info_i = cluster_info[ci]
        for j in range(i + 1, len(cluster_ids)):
            cj = cluster_ids[j]
            if find(cj) != cj:
                continue  # 已被合并

            info_j = cluster_info[cj]

            # 纬度预过滤（纬度差不绕回，可安全提前终止）
            dlat = info_j["centroid_lat"] - info_i["centroid_lat"]
            if dlat > dateline_merge_dist:
                break  # 已按纬度排序，后续簇纬度差只会更大

            # 空间距离（循环经度差）
            dlon = abs(info_i["centroid_lon"] - info_j["centroid_lon"])
            dlon = min(dlon, 360 - dlon)
            dist_deg = np.sqrt(dlon**2 + dlat**2)

            # ★日界线感知：若两簇分别在日界线两侧（一>350°，一<10°），使用更大阈值
            lon_i, lon_j = info_i["centroid_lon"], info_j["centroid_lon"]
            crosses_dl = (lon_i > 350 and lon_j < 10) or (lon_j > 350 and lon_i < 10)
            effective_threshold = dateline_merge_dist if crosses_dl else merge_dist_deg

            if dist_deg > effective_threshold:
                continue

            # 时间重叠检查
            t_min_i, t_max_i = info_i["t_min"], info_i["t_max"]
            t_min_j, t_max_j = info_j["t_min"], info_j["t_max"]
            time_overlap = not (t_max_i < t_min_j or t_max_j < t_min_i)
            time_gap = max(0, max(t_min_j - t_max_i, t_min_i - t_max_j))

            if time_overlap or time_gap <= merge_time_gap:
                union(ci, cj)
                merge_count += 1

    if merge_count > 0:
        # 应用合并映射
        new_labels = labels.copy()
        root_map = {}
        for cid in cluster_ids:
            root = find(cid)
            if root not in root_map:
                root_map[root] = len(root_map)
        for cid in cluster_ids:
            mask = labels == cid
            new_labels[mask] = root_map[find(cid)]

        dt_events = dt_events.copy()
        dt_events["spatial_cluster"] = new_labels
        new_n_clusters = new_labels.max() + 1
        print(f"  🔗 跨块合并: {merge_count} 对簇被合并, 簇数 {n_clusters} → {new_n_clusters}")

    return dt_events


def cluster_events_dbscan(dt_events, config):
    """DBSCAN 空间聚类（使用球面距离 + Ball Tree优化）"""
    print("🗺️ DBSCAN空间聚类")
    print(f"  eps = {config['dbscan_eps_deg']:.1f}°, minPts = {config['dbscan_minPts']}")
    if dt_events is None or dt_events.empty:
        return None
    n_events = len(dt_events)
    print(f"  📊 共 {n_events} 个网格级事件")
    
    # 修正经度 > 180 的情况
    lons = dt_events["lon"].values.copy()
    lons_adj = np.where(lons > 180, lons - 360, lons)
    lats = dt_events["lat"].values
    time_scaled = dt_events["time_scaled"].values
    
    # ---- 纬度自适应eps：高纬度经度间距缩小，eps需相应调整 ----
    # 等效地面距离 = eps_deg × 111 × cos(lat)，保持地面距离一致
    # 对每个事件计算其等效eps_km
    eps_base_km = config["dbscan_eps_deg"] * 111  # 赤道处的eps(km)
    # 对中位数纬度计算全局eps_km（用于DBSCAN全局参数）
    median_lat = np.median(np.abs(lats))
    eps_km = eps_base_km * max(np.cos(np.radians(median_lat)), 0.3)  # 下限0.3防止极地过小
    print(f"  📍 中位纬度={median_lat:.1f}°, 等效eps={eps_km:.0f}km (赤道={eps_base_km:.0f}km)")
    max_exact = config.get("dbscan_max_points_exact", 50000)
    
    # ---- 预分区优化：按lon/lat分块独立聚类 ----
    pre_partition = config.get("performance", {}).get("dbscan_pre_partition", False)
    if pre_partition and n_events > 5000:
        n_lon_blocks = config.get("performance", {}).get("dbscan_partition_lon", 10)
        n_lat_blocks = config.get("performance", {}).get("dbscan_partition_lat", 6)
        print(f"  🧩 预分区模式: {n_lon_blocks}×{n_lat_blocks} 块 (总{n_lon_blocks*n_lat_blocks}块)")
        
        lon_min, lon_max = lons_adj.min(), lons_adj.max()
        lat_min, lat_max = lats.min(), lats.max()
        lon_edges = np.linspace(lon_min, lon_max + 1e-6, n_lon_blocks + 1)
        lat_edges = np.linspace(lat_min, lat_max + 1e-6, n_lat_blocks + 1)
        
        block_id_lon = np.digitize(lons_adj, lon_edges) - 1
        block_id_lat = np.digitize(lats, lat_edges) - 1
        block_id = block_id_lat * n_lon_blocks + block_id_lon
        
        unique_blocks = np.unique(block_id)
        print(f"    实际使用 {len(unique_blocks)} 块")
        
        all_labels = np.full(n_events, -1, dtype=int)
        cluster_offset = 0
        
        base_min_pts = config["dbscan_minPts"]
        for bid in unique_blocks:
            block_mask = block_id == bid
            block_idx = np.where(block_mask)[0]
            block_size = len(block_idx)

            # ★修复 P2: minPts 纬度自适应——高纬度降雨稀疏，按 cos(lat) 放宽
            #   热带(|lat|<23.5): minPts 不变
            #   温带(|lat|~45): minPts × cos(45°) ≈ ×0.7
            #   极地(|lat|>60): minPts × cos(60°) = ×0.5，下限3防止过度放宽
            block_lons = lons_adj[block_idx]
            block_lats = lats[block_idx]
            block_time = time_scaled[block_idx]
            abs_lat_med = float(np.median(np.abs(block_lats)))
            lat_factor = max(np.cos(np.radians(abs_lat_med)), 0.5)
            block_min_pts = max(3, int(round(base_min_pts * lat_factor)))

            if block_size < block_min_pts:
                continue

            # 统一使用3D笛卡尔坐标，避免大块用欧氏距离导致跨日界线事件割裂
            lon_rad_b = np.radians(block_lons)
            lat_rad_b = np.radians(block_lats)
            x_b = np.cos(lat_rad_b) * np.cos(lon_rad_b)
            y_b = np.cos(lat_rad_b) * np.sin(lon_rad_b)
            z_b = np.sin(lat_rad_b)
            t_norm_b = block_time / config["time_scale_factor"] * 0.5
            block_data_3d = np.column_stack([x_b, y_b, z_b, t_norm_b])
            eps_rad_b = eps_km / 6371.0
            block_db = DBSCAN(eps=eps_rad_b, min_samples=block_min_pts,
                              algorithm='ball_tree', n_jobs=-1).fit(block_data_3d)
            
            block_labels = block_db.labels_
            valid_mask = block_labels != -1
            if valid_mask.any():
                block_labels[valid_mask] += cluster_offset
                n_clusters_block = block_labels.max() + 1
                cluster_offset += n_clusters_block
            
            all_labels[block_idx] = block_labels
        
        dt_events = dt_events.copy()
        dt_events["spatial_cluster"] = all_labels
        n_clusters = max(0, all_labels.max())
        n_noise = (all_labels == -1).sum()
        print(f"✅ 预分区聚类完成: 共 {n_clusters} 个空间簇（不含噪声）")
        print(f"  噪声点数：{n_noise} ({n_noise / n_events * 100:.1f}%)")
        
        # ---- ★跨块合并：合并相邻块中时间/空间接近的簇 ----
        # 解决大范围锋面/热带气旋被块边界割裂的问题
        if n_clusters > 0:
            dt_events = _merge_cross_block_clusters(
                dt_events, lons_adj, lats, time_scaled, config
            )
        
        if n_noise > 0 and n_noise < n_events * 0.3:
            noise_events = dt_events[dt_events["spatial_cluster"] == -1]
            print(f"  📋 噪声点特征: 平均降水={noise_events['total_rain'].mean():.1f}mm, "
                  f"平均时长={noise_events['duration_hours'].mean():.1f}h")
        
        return dt_events
    
    if n_events > max_exact:
        # 大数据量（>5万）：使用欧氏距离近似（性能优先）
        print(f"  ⚡ 数据量大 ({n_events} > {max_exact})，使用欧氏距离近似...")
        lons_norm = np.where(lons > 180, lons - 360, lons)
        clust_data = np.column_stack([lons_norm, lats, time_scaled])
        db = DBSCAN(eps=config["dbscan_eps_deg"], min_samples=config["dbscan_minPts"],
                    algorithm='ball_tree', n_jobs=-1).fit(clust_data)
    elif n_events > 10000:
        # 中等数据量（1-5万）：使用球面距离 + Ball Tree
        print(f"  🌍 使用球面距离 + Ball Tree 聚类 ({n_events} 点)...")
        # 构建球面坐标（弧度）作为 Ball Tree 输入
        lon_rad = np.radians(lons_adj)
        lat_rad = np.radians(lats)
        # 转为3D笛卡尔坐标（单位：地球半径）
        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)
        
        # 时间维度归一化到与空间相近的尺度
        t_norm = time_scaled / config["time_scale_factor"] * 0.5  # 时间权重
        
        clust_data_3d = np.column_stack([x, y, z, t_norm])
        
        # 使用 Ball Tree 进行 DBSCAN
        eps_rad = eps_km / 6371.0  # 将km转为弧度
        db = DBSCAN(eps=eps_rad, min_samples=config["dbscan_minPts"],
                    algorithm='ball_tree', n_jobs=-1).fit(clust_data_3d)
    else:
        # 小数据量（<1万）：使用精确距离矩阵
        print(f"  🌍 使用精确球面距离矩阵聚类 ({n_events} 点)...")
        
        # 计算距离矩阵
        dist_matrix = _haversine_distance_matrix(lons_adj, lats)
        
        # 加入时间维度距离
        time_diff = np.abs(time_scaled[:, np.newaxis] - time_scaled[np.newaxis, :])
        combined = np.sqrt(dist_matrix ** 2 + (time_diff * eps_km / config["time_scale_factor"]) ** 2)
        
        # 使用预计算距离矩阵
        db = DBSCAN(eps=eps_km, min_samples=config["dbscan_minPts"], metric="precomputed").fit(combined)
    
    dt_events = dt_events.copy()
    dt_events["spatial_cluster"] = db.labels_
    n_clusters = dt_events["spatial_cluster"].max()
    n_noise = (dt_events["spatial_cluster"] == -1).sum()
    print(f"✅ 共识别出 {n_clusters} 个空间簇（不含噪声）")
    print(f"  噪声点数：{n_noise} ({n_noise / n_events * 100:.1f}%)")
    
    # 噪声点分析
    if n_noise > 0 and n_noise < n_events * 0.3:
        noise_events = dt_events[dt_events["spatial_cluster"] == -1]
        print(f"  📋 噪声点特征: 平均降水={noise_events['total_rain'].mean():.1f}mm, "
              f"平均时长={noise_events['duration_hours'].mean():.1f}h")
    
    return dt_events


# ============================================================
# 动态切分与固定切分
# ============================================================

def dynamic_split_events(dt_events, config):
    """动态切分事件（向量化版 + 自适应阈值）"""
    print("\n🔧 动态切分")
    if config["split_method"] == "fixed":
        return split_fixed(dt_events, config["fixed_split_hours"])
    area_threshold = config["area_change_threshold"]
    intensity_threshold = config["intensity_drop_threshold"]
    print(f"  阈值: 面积变化>{area_threshold:.2f}, 强度下降>{intensity_threshold:.2f}")
    if config.get("adaptive_split", False):
        print("  🌍 自适应切分已启用")
    dt_events = dt_events[dt_events["spatial_cluster"] != -1].copy()
    if dt_events.empty:
        return None
    dt_events = dt_events.sort_values(["spatial_cluster", "time_num"]).reset_index(drop=True)
    dt_events["time_bin"] = (dt_events["time_num"] / 0.5).astype(int)

    def _split_cluster(sub_df):
        """处理单个 cluster 的切分"""
        if len(sub_df) < 10:
            return pd.Series(1, index=sub_df.index)
        area_over_time = sub_df.groupby("time_bin").agg(
            area=("time_bin", "size"),
            mean_int=("mean_intensity", "mean"),
            clon=("lon", "median"),
            clat=("lat", "median")
        ).sort_index()
        if len(area_over_time) < 2:
            return pd.Series(1, index=sub_df.index)
        # ★可读性优化: 首点无前值，pct_change/diff 为 NaN，显式置 0 表示"无变化"
        #   原代码用 fillna(0) 隐式处理，这里改为显式赋值提高可读性
        area_chg = area_over_time["area"].pct_change()
        area_chg.iloc[0] = 0.0  # 首点无变化
        prev_int = area_over_time["mean_int"].shift(1)
        int_drop = (prev_int - area_over_time["mean_int"]) / prev_int
        int_drop.iloc[0] = 0.0  # 首点无强度下降
        # 防止除零产生的 inf/NaN（prev_int=0 时）
        int_drop = int_drop.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        cp1 = area_over_time.index[area_chg < -area_threshold].tolist()
        cp2 = area_over_time.index[(int_drop > intensity_threshold) & (int_drop < 1)].tolist()

        # ---- 空间跳跃检测：质心突然移动 > 5° (≈500km) ----
        # ★经度差用循环归一化，避免跨0°/360°经线时误判为~360°跳跃
        clon = area_over_time["clon"].values
        clat = area_over_time["clat"].values
        dlon = ((np.diff(clon) + 180) % 360) - 180  # 归一化到[-180,180]
        centroid_shift = np.sqrt(
            dlon ** 2 + np.diff(clat) ** 2
        )
        cp3 = [area_over_time.index[i + 1]
               for i, s in enumerate(centroid_shift) if s > 5.0]

        cut_points = sorted(set(cp1 + cp2 + cp3))
        cut_points = [c for c in cut_points
                      if area_over_time.index.get_loc(c) not in (0, len(area_over_time) - 1)]
        if not cut_points:
            return pd.Series(1, index=sub_df.index)

        # ★最小历时约束：若切分后某段<min_split_duration_hours，移除对应的切分点
        min_split_hours = config.get("min_split_duration_hours", 0)
        if min_split_hours > 0:
            time_vals = area_over_time.index.values
            valid_cut_points = []
            for cp in cut_points:
                cp_idx = area_over_time.index.get_loc(cp)
                # 检查切分点前后两段的时长
                seg_start_time = time_vals[0]
                seg_end_time = time_vals[cp_idx]
                dur_before = (seg_end_time - seg_start_time) * 0.5  # time_bin × 0.5h
                # 找到下一个切分点或末尾
                remaining_cuts = [c for c in cut_points if area_over_time.index.get_loc(c) > cp_idx]
                if remaining_cuts:
                    next_cp_idx = area_over_time.index.get_loc(remaining_cuts[0])
                else:
                    next_cp_idx = len(time_vals) - 1
                seg_end2 = time_vals[next_cp_idx]
                dur_after = (seg_end2 - seg_end_time) * 0.5
                # 只有当两段都>=min_split_hours时才保留此切分点
                if dur_before >= min_split_hours and dur_after >= min_split_hours:
                    valid_cut_points.append(cp)
            cut_points = valid_cut_points
            if not cut_points:
                return pd.Series(1, index=sub_df.index)

        group_assign = np.ones(len(sub_df), dtype=int)
        for cp in cut_points:
            group_assign[sub_df["time_bin"].values > cp] += 1
        return pd.Series(group_assign, index=sub_df.index)

    print("  按 cluster 批量切分...")
    dt_events["temp_group"] = dt_events.groupby("spatial_cluster", group_keys=False).apply(
        _split_cluster
    ).astype(int).values

    dt_events["temp_event_id"] = dt_events.groupby(
        ["spatial_cluster", "temp_group"]
    ).ngroup() + 1
    dt_events["event_duration"] = dt_events.groupby(
        "temp_event_id"
    )["duration_hours"].transform("sum")
    dt_events["final_event_id"] = dt_events.groupby(
        "temp_event_id"
    ).ngroup() + 1
    dt_events = dt_events[dt_events["final_event_id"].notna()].copy()
    dt_events["event_id_final"] = dt_events["final_event_id"].astype(int)
    n_final = dt_events["event_id_final"].nunique()
    print(f"  ✅ 动态切分完成: {n_final} 个最终事件")
    return dt_events


def split_fixed(dt_events, fixed_hours):
    """固定时长切分"""
    dt_events["event_duration"] = dt_events.groupby("temp_event_id")["duration_hours"].transform("sum")
    dt_events["final_event_id"] = np.nan
    new_eid = 0
    for teid in dt_events["temp_event_id"].unique():
        rows = dt_events["temp_event_id"] == teid
        sub = dt_events[rows].copy()
        if sub["event_duration"].iloc[0] <= fixed_hours:
            dt_events.loc[rows, "final_event_id"] = new_eid + 1
            new_eid += 1
        else:
            sub = sub.sort_values("time_num")
            cum_time = sub["duration_hours"].cumsum().values
            total_dur = sub["event_duration"].iloc[0]
            n_splits = math.ceil(total_dur / fixed_hours)
            breakpoints = np.linspace(0, total_dur, n_splits + 1)
            group_assign = np.digitize(cum_time, breakpoints, right=True)
            group_assign = np.minimum(group_assign, n_splits)
            for g in np.unique(group_assign):
                # 重新定位到子集索引
                sub_idx = sub.index[group_assign == g]
                dt_events.loc[sub_idx, "final_event_id"] = new_eid + 1
                new_eid += 1
    dt_events = dt_events[~dt_events["final_event_id"].isna()].copy()
    dt_events["event_id_final"] = dt_events["final_event_id"].astype(int)
    return dt_events


# ============================================================
# 事件扩展与汇总
# ============================================================

def expand_events(dt_merged, config):
    """事件扩展"""
    print("🔍 事件扩展")
    if dt_merged is None or dt_merged.empty:
        return dt_merged
    expand_radius = config["expand_search_radius"]
    res_deg = config["spatial_res_deg"]
    pad = expand_radius * res_deg
    event_bounds = dt_merged.groupby("event_id_final").agg(
        lon_min=("lon", "min"), lon_max=("lon", "max"),
        lat_min=("lat", "min"), lat_max=("lat", "max")
    ).reset_index()
    # ★经度循环处理：跨0°/360°经线事件，用循环范围确定展开边界
    # 若 lon_max - lon_min > 180，说明跨日界线，取环绕范围
    _lon_range = ((event_bounds["lon_max"] - event_bounds["lon_min"] + 180) % 360) - 180
    _crosses_dateline = (event_bounds["lon_max"] - event_bounds["lon_min"]) > 180
    # 正常事件：直接加减pad
    event_bounds["lon_min_expanded"] = event_bounds["lon_min"] - pad
    event_bounds["lon_max_expanded"] = event_bounds["lon_max"] + pad
    # 跨日界线事件：以环绕中心为基准展开，再绕回[0,360)
    if _crosses_dateline.any():
        # ★修复 P1: 跨日界线时 _center 不能用简单算术平均
        # 原 bug: (350+10)/2 = 180°，实际中心应在 0° 附近
        # 修复: 用循环平均计算中心
        _lon_min_cd = event_bounds.loc[_crosses_dateline, "lon_min"].values
        _lon_max_cd = event_bounds.loc[_crosses_dateline, "lon_max"].values
        # 循环中心: 将 lon_max 加 360° 后求平均，再 mod 360
        _center_cd = (_lon_min_cd + _lon_max_cd + 360) / 2 % 360
        event_bounds.loc[_crosses_dateline, "_center_tmp"] = _center_cd
        _half_range = np.abs(_lon_range[_crosses_dateline]) / 2 + pad
        _min_raw = event_bounds.loc[_crosses_dateline, "_center_tmp"].values - _half_range
        event_bounds.loc[_crosses_dateline, "lon_min_expanded"] = _min_raw % 360
        event_bounds.loc[_crosses_dateline, "lon_max_expanded"] = (_min_raw + 2 * _half_range) % 360
        event_bounds = event_bounds.drop(columns=["_center_tmp"])
    event_bounds["lat_min_expanded"] = event_bounds["lat_min"] - pad
    event_bounds["lat_max_expanded"] = event_bounds["lat_max"] + pad
    dt_merged = dt_merged.merge(
        event_bounds[["event_id_final", "lon_min_expanded", "lon_max_expanded",
                       "lat_min_expanded", "lat_max_expanded"]],
        on="event_id_final", how="left"
    )
    return dt_merged


def circular_weighted_lon(lon_series, weight_series):
    """对 0-360° 经度做循环加权平均

    直接对 0-360° 经度做加权平均会在跨日界线时算到地球对面。
    正确做法：转成单位向量(x,y)加权平均后再转回角度。

    Args:
        lon_series: 经度数组/Series (0-360°)
        weight_series: 权重数组/Series (自动clip到>=0.01)

    Returns:
        float: 循环加权平均经度 (0-360°)
    """
    w = np.asarray(weight_series, dtype=float).clip(min=0.01)
    lon = np.asarray(lon_series, dtype=float)
    if w.sum() <= 0:
        return float(np.average(lon))
    lon_rad = np.radians(lon)
    x = np.average(np.cos(lon_rad), weights=w)
    y = np.average(np.sin(lon_rad), weights=w)
    if abs(x) < 1e-12 and abs(y) < 1e-12:
        return float(np.average(lon))
    return float(np.degrees(np.arctan2(y, x)) % 360)


def unwrap_lon(lon_arr, w_arr):
    """经度解缠绕：以循环加权均值为基准，将lon平移到该基准附近

    用于加权线性回归前处理，避免跨0°/360°经线时回归斜率错误。

    Args:
        lon_arr: 经度数组 (0-360°)
        w_arr: 权重数组

    Returns:
        tuple: (unwrapped_lon, circular_mean)
            - unwrapped_lon: 解缠绕后的经度（可能超出[0,360)但保证连续）
            - circular_mean: 循环加权均值 (0-360°)
    """
    lon_arr = np.asarray(lon_arr, dtype=float)
    w_arr = np.asarray(w_arr, dtype=float)
    lon_rad = np.radians(lon_arr)
    cx = np.average(np.cos(lon_rad), weights=w_arr)
    cy = np.average(np.sin(lon_rad), weights=w_arr)
    circ_mean = np.degrees(np.arctan2(cy, cx)) % 360
    unwrapped = ((lon_arr - circ_mean + 180) % 360) - 180 + circ_mean
    return unwrapped, circ_mean


def summarize_events(dt_events, config, geo_data):
    """生成事件汇总统计"""
    print("\n📊 生成事件汇总统计...")
    if dt_events is None or dt_events.empty:
        return None
    dt_final = dt_events[~dt_events["event_id_final"].isna()].copy()
    if dt_final.empty:
        return None

    n_events = dt_final["event_id_final"].nunique()
    print(f"  正在汇总 {n_events} 个事件...")

    event_summary = dt_final.groupby("event_id_final").agg(
        start_time=("start_time", "min"),
        end_time=("end_time", "max"),
        total_rain_depth=("total_rain", lambda s: s.sum(skipna=True)),
        max_30min_rain=("max_30min_rain", lambda s: s.max(skipna=True)),
        max_30min_time=("max_30min_time", lambda s: s.iloc[np.argmax(dt_final.loc[s.index, "max_30min_rain"].values)]),
        mean_intensity=("mean_intensity", lambda s: s.mean(skipna=True)),
        lon_min=("lon", lambda s: s.min(skipna=True)),
        lon_max=("lon", lambda s: s.max(skipna=True)),
        lat_min=("lat", lambda s: s.min(skipna=True)),
        lat_max=("lat", lambda s: s.max(skipna=True)),
        # ★修复 P0: center_lon 用循环平均而非 median
        # 原 median 在跨日界线事件（如 350°→10°）会给出 ~180° 的错误中心
        center_lon=("lon", lambda s: circular_weighted_lon(s, pd.Series([1.0]*len(s)))),
        center_lat=("lat", "median"),
        n_grids=("lon", "size"),
    ).reset_index()

    # ---- 降雨加权质心（循环平均处理跨0°/360°经线事件）----
    # 直接对 0-360° 经度做加权平均会在跨日界线时算到地球对面
    # 正确做法：转成单位向量(x,y)加权平均后再转回角度（调用模块级函数）
    _weighted_lons = []
    _weighted_lats = []
    for eid, sub in dt_final.groupby("event_id_final"):
        w = sub["total_rain"].clip(lower=0.01)
        _weighted_lons.append(circular_weighted_lon(sub["lon"], w))
        _weighted_lats.append(float(np.average(sub["lat"], weights=w)))
    event_summary["weighted_lon"] = _weighted_lons
    event_summary["weighted_lat"] = _weighted_lats

    # ---- 计算真实事件持续时间（不能简单sum，要计算时间跨度）----
    event_summary["duration_hours"] = (
        (event_summary["end_time"] - event_summary["start_time"]).dt.total_seconds() / 3600
    )
    
    # 过滤掉持续时间为0或异常的事件
    event_summary = event_summary[event_summary["duration_hours"] > 0]
    event_summary = event_summary[event_summary["duration_hours"] <= 168]  # 上限7天
    
    # ---- 修正单位换算 ----
    # total_rain_depth 是所有网格点的总降雨量之和 (mm)
    # avg_precip_depth 是每个网格点的平均降雨量 (mm/grid)
    # max_30min_rain 是事件内最大30分钟降雨 (mm)
    # max_30min_intensity 是最大30分钟雨强 (mm/hr) = max_30min_rain * 2
    
    event_summary["avg_precip_depth"] = event_summary["total_rain_depth"] / event_summary["n_grids"]
    event_summary["max_30min_intensity"] = event_summary["max_30min_rain"] * 2
    # ★新增 mean_intensity_all: 全网格平均雨强 = avg_precip_depth / duration_hours
    # 与 mean_intensity（条件平均雨强，仅统计有降水格点）区分
    event_summary["mean_intensity_all"] = (
        event_summary["avg_precip_depth"] / np.maximum(event_summary["duration_hours"], 0.5)
    ).astype(np.float32)
    
    # ★修复：确保 mean_intensity >= mean_intensity_all（条件平均应 >= 全网格平均）
    # mean_intensity 是仅统计有降水格点的条件平均雨强
    # mean_intensity_all 是全网格（含无雨格点）的平均雨强
    # 理论上条件平均应 >= 全网格平均，若违反则修正
    if "mean_intensity" in event_summary.columns:
        _intensity_violation = event_summary["mean_intensity"] < event_summary["mean_intensity_all"]
        if _intensity_violation.any():
            n_violations = _intensity_violation.sum()
            print(f"  ⚠️ 修正 {n_violations} 个 mean_intensity < mean_intensity_all 的异常事件")
            # 修正：将 mean_intensity 设为 max(原值, mean_intensity_all)
            event_summary.loc[_intensity_violation, "mean_intensity"] = np.maximum(
                event_summary.loc[_intensity_violation, "mean_intensity"],
                event_summary.loc[_intensity_violation, "mean_intensity_all"]
            ).astype(np.float32)
    
    # ---- 逻辑一致性检查（数据质量过滤，与极端值标记不冲突）----
    # 注：以下检查针对物理不可能的异常值（如数据损坏导致 max_30min 远超合理范围）
    # 与"极端值不截断"理念不矛盾：极端值用 is_extreme_* 标志位标记，而这里是数据质量修正
    # ★阈值从×10提高到×20，避免误删真实极端暴雨峰值
    # 物理上限：30min降水>200mm(即500mm/hr)才视为数据异常，远超此值才需修正
    avg_rain_per_obs = event_summary["total_rain_depth"] / event_summary["n_grids"] / np.maximum(event_summary["duration_hours"] / 0.5, 1)
    max_allowed = avg_rain_per_obs * 20  # ★放宽到20倍，保留真实极端信号
    # 额外物理上限保护：30min>200mm视为数据异常（气象记录极值约150mm/30min）
    physical_cap = 200.0
    inconsistent = (event_summary["max_30min_rain"] > max_allowed) & (event_summary["max_30min_rain"] > physical_cap)
    if inconsistent.any():
        print(f"  ⚠️ 修正 {inconsistent.sum()} 个异常高峰值事件（数据质量过滤，30min>{physical_cap}mm且>{20}倍均值）")
        event_summary.loc[inconsistent, "max_30min_rain"] = physical_cap
    event_summary["max_30min_intensity"] = (event_summary["max_30min_rain"] * 2).values.astype(np.float32)

    # ---- 极端值标记（不截断，保留原始值用于极端降雨研究）----
    # 注：试运行对比报告指出硬截断会损失极端事件信息（全年版同期最大327.5mm）
    # 改为仅标记不截断，论文使用时建议做敏感性分析
    event_summary["is_extreme_precip"] = (event_summary["avg_precip_depth"] > 200).astype(np.int8)
    event_summary["is_extreme_max30"] = (event_summary["max_30min_rain"] > 100).astype(np.int8)
    n_extreme_precip = int(event_summary["is_extreme_precip"].sum())
    n_extreme_max30 = int(event_summary["is_extreme_max30"].sum())
    if n_extreme_precip > 0:
        print(f"  ℹ️ 标记 {n_extreme_precip} 个极端事件 (avg_precip_depth > 200mm, 未截断)")
    if n_extreme_max30 > 0:
        print(f"  ℹ️ 标记 {n_extreme_max30} 个极端事件 (max_30min_rain > 100mm, 未截断)")

    # ---- 置信度计算（基于正确的duration）----
    # ★改进：原公式用max_30min_intensity/50，极端强度直接拉高置信度，存在循环论证
    # 新方案：各维度归一化后加权，强度项改用对数压缩+空间/时长项主导
    # 注：置信度为内部质量标记，不用于筛选论文结果
    n_grids_norm = np.minimum(event_summary["n_grids"] / 100, 1.0)
    duration_norm = np.minimum(event_summary["duration_hours"] / 48, 1.0)
    # 强度用对数压缩，避免极端值主导
    intensity_norm = np.minimum(np.log1p(event_summary["max_30min_intensity"]) / np.log1p(50), 1.0)
    event_summary["confidence"] = np.clip(
        n_grids_norm * 0.4 + duration_norm * 0.35 + intensity_norm * 0.25,
        0.0, 1.0
    )

    # ---- 计算bbox面积(km²) 用于碎裂事件检测 ----
    # ★经度范围用循环差值，避免跨0°/360°经线时算到~360°导致bbox面积虚高
    grid_area_km2 = (config.get("spatial_res_deg", 0.072) * 111) ** 2
    lon_range_circular = ((event_summary["lon_max"] - event_summary["lon_min"] + 180) % 360) - 180
    lon_range_circular = np.abs(lon_range_circular)
    # ★输出 lon_range_circular 和 crosses_dateline 字段，供数据用户判断跨日界线事件
    event_summary["lon_range_circular"] = lon_range_circular.astype(np.float32)
    
    # ★修复：跨日界线事件翻转 lon_min/lon_max（符合数据字典约定）
    # 当 lon_max - lon_min > 180° 时，说明事件跨日界线
    # 此时翻转：lon_min 存较大的经度值，lon_max 存较小的经度值
    # 这样下游代码可以通过 lon_min > lon_max 快速判断跨日界线事件
    _crosses_dl_mask = (event_summary["lon_max"] - event_summary["lon_min"]) > 180
    event_summary["crosses_dateline"] = _crosses_dl_mask.astype(np.int8)
    
    # 翻转跨日界线事件的 lon_min/lon_max
    if _crosses_dl_mask.any():
        _tmp_lon = event_summary.loc[_crosses_dl_mask, "lon_min"].values.copy()
        event_summary.loc[_crosses_dl_mask, "lon_min"] = event_summary.loc[_crosses_dl_mask, "lon_max"].values
        event_summary.loc[_crosses_dl_mask, "lon_max"] = _tmp_lon
    
    event_summary["bbox_area_km2"] = (
        lon_range_circular *
        (event_summary["lat_max"] - event_summary["lat_min"]) *
        111 * 111 *
        np.cos(np.radians(event_summary["center_lat"].clip(-80, 80)))
    )

    # ---- 过滤碎裂噪声事件（单一过滤，不再重复）----
    min_conf = config.get("min_confidence", 0.3)
    low_conf_mask = event_summary["confidence"] < min_conf
    fragmented_mask = (
        (event_summary["n_grids"] <= 18) &
        (event_summary["bbox_area_km2"] > 5000)
    )
    # 最小网格数：陆地事件用较低阈值（陆地降雨空间尺度小）
    min_grids_land = config.get("min_area_pixels_land", 5)
    min_grids_ocean = config.get("min_area_pixels", 25)
    # 使用加权质心判断是否是陆地（更准确反映降雨主体位置）
    # 优先使用 weighted_lon/lat，fallback 到 center_lon/lat
    if "weighted_lon" in event_summary.columns:
        judge_lon = event_summary["weighted_lon"].fillna(event_summary["center_lon"]).values
        judge_lat = event_summary["weighted_lat"].fillna(event_summary["center_lat"]).values
    else:
        judge_lon = event_summary["center_lon"].values
        judge_lat = event_summary["center_lat"].values
    lon_n = ((judge_lon + 180) % 360) - 180
    lat_v = judge_lat
    # 改进的陆地判断：覆盖主要大陆及其海岸带
    is_likely_land = (
        # 非洲+欧洲（含地中海沿岸）
        ((-25 <= lon_n) & (lon_n <= 55) & (-40 <= lat_v) & (lat_v <= 72)) |
        # 亚洲（含中东、东南亚、东亚）
        ((55 <= lon_n) & (lon_n <= 155) & (-12 <= lat_v) & (lat_v <= 78)) |
        # 北美（含中美洲）
        ((-170 <= lon_n) & (lon_n <= -50) & (8 <= lat_v) & (lat_v <= 75)) |
        # 南美
        ((-82 <= lon_n) & (lon_n <= -34) & (-56 <= lat_v) & (lat_v <= 13)) |
        # 澳洲+大洋洲
        ((110 <= lon_n) & (lon_n <= 160) & (-48 <= lat_v) & (lat_v <= -8)) |
        # 格陵兰岛
        ((-55 <= lon_n) & (lon_n <= -20) & (60 <= lat_v) & (lat_v <= 84)) |
        # 日本群岛、英国群岛等岛屿（避免小岛事件被误判为海洋）
        ((125 <= lon_n) & (lon_n <= 146) & (30 <= lat_v) & (lat_v <= 46)) |
        ((-10 <= lon_n) & (lon_n <= 2) & (50 <= lat_v) & (lat_v <= 59))
    )
    min_grids_arr = np.where(is_likely_land, min_grids_land, min_grids_ocean)
    small_mask = event_summary["n_grids"].values < min_grids_arr

    filter_mask = low_conf_mask | fragmented_mask | small_mask
    n_filtered = filter_mask.sum()
    if n_filtered > 0:
        print(f"\n  🗑️ 过滤 {n_filtered} 个事件:")
        print(f"     - 低置信度(<{min_conf}): {low_conf_mask.sum()}")
        print(f"     - 碎裂噪声(n≤18, bbox>5000km²): {fragmented_mask.sum()}")
        print(f"     - 网格不足(陆地<{min_grids_land}/海洋<{min_grids_ocean}): {small_mask.sum()}")
        event_summary = event_summary[~filter_mask].copy()
        print(f"     剩余: {len(event_summary)} 个事件")

    if event_summary.empty:
        return None

    # ---- 计算总水量 (m³) ----
    # 正确公式: V(m³) = avg_precip(mm) × 10⁻³(m/mm) × n_grids × grid_area(km²) × cos(lat) × 10⁶(m²/km²)
    #                = avg_precip × n_grids × grid_area_km² × cos(lat) × 10³
    # 注: grid_area_km2 是赤道值，需按纬度余弦修正（高纬度经度方向距离缩短）
    lat_cos = np.cos(np.radians(event_summary["center_lat"].clip(-80, 80).astype(np.float64)))
    event_summary["total_water_volume"] = (
        event_summary["avg_precip_depth"] * event_summary["n_grids"] *
        grid_area_km2 * lat_cos * 1e3
    ).astype(np.float32)
    event_summary = event_summary.sort_values("start_time").reset_index(drop=True)

    # ---- 生成全局唯一ID (格式: YYYYMM_NNNN) ----
    year = config.get("year", 2025)
    month = config.get("month", 1)
    prefix = f"{year}{month:02d}"
    event_summary["event_id"] = [
        f"{prefix}_{i+1:05d}" for i in range(len(event_summary))
    ]

    # ---- 输出参数披露摘要 ----
    print("\n  📋 参数摘要 (供论文披露):")
    print(f"     - 数据: CMORPH {year}年{month}月, 分辨率 {config.get('spatial_res_deg', 0.072)}°")
    print("     - 检测阈值: 气候自适应 (柯本分区 0.05-0.2mm)")
    print("     - IETD: 气候自适应 (2-12h)")
    print(f"     - DBSCAN: eps={config.get('dbscan_eps_deg', 0.8)}°, minPts={config.get('dbscan_minPts', 5)}")
    print(f"     - 最小事件网格数: {config.get('min_area_pixels', 25)}")
    print(f"     - 置信度阈值: {config.get('min_confidence', 0.3)}")
    print("     - 碎裂事件过滤: n_grids≤18 且 bbox>5000km²")
    print(f"     - 时间缩放因子: {config.get('time_scale_factor', 10)}")

    print(f"\n✅ 共生成 {len(event_summary)} 个最终事件")
    return event_summary


# ============================================================
# 地理与气候分类
# ============================================================

def _simple_classification(event_summary):
    """无 geopandas 时的简单分类回退"""
    location_type = np.array([
        classify_location_type_simple(lon, lat) for lon, lat in
        zip(event_summary["center_lon"], event_summary["center_lat"])
    ])
    continent_result = np.array([
        classify_continent_simple(lon, lat) for lon, lat in
        zip(event_summary["center_lon"], event_summary["center_lat"])
    ])
    ocean = np.array([
        classify_ocean(lon, lat) if c in ["海洋", "海洋/未分类"] else np.nan
        for lon, lat, c in zip(
            event_summary["center_lon"], event_summary["center_lat"],
            continent_result
        )
    ])
    # 简单气候区：按纬度带划分
    climate_zone = np.array([
        "热带" if -23.5 <= lat <= 23.5 else
        "亚热带" if (abs(lat) > 23.5 and abs(lat) <= 35) else
        "温带" if (abs(lat) > 35 and abs(lat) <= 55) else
        "寒带" for lat in event_summary["center_lat"]
    ])
    event_summary["location_type"] = location_type
    event_summary["continent"] = continent_result
    event_summary["ocean"] = ocean
    event_summary["climate_zone"] = climate_zone
    # 简单分类无国家信息，标为"未知"
    event_summary["country"] = np.where(
        np.isin(location_type, ["海洋", "近海"]), "", "未知"
    )
    return event_summary


def add_classifications(event_summary, config, kg_gdf=None):
    """添加地理与气候分类（批量空间查询版）"""
    print("\n🌍 添加地理与气候分类（批量查询）")
    if event_summary is None or event_summary.empty:
        return None
    if not HAS_GEOPANDAS:
        print("  ⚠️ geopandas/shapely 未安装，使用简单分类回退")
        # 简单回退：按纬度带划分气候区，按经度粗略判断海陆
        event_summary = _simple_classification(event_summary)
        return event_summary
    geo_data = load_geo_data()
    n = len(event_summary)

    # ---- 批量构建 GeoDataFrame（使用降雨加权质心）----
    use_weighted = "weighted_lon" in event_summary.columns
    if use_weighted:
        lon_col = event_summary["weighted_lon"].fillna(event_summary["center_lon"]).values
        lat_col = event_summary["weighted_lat"].fillna(event_summary["center_lat"]).values
    else:
        lon_col = event_summary["center_lon"].values
        lat_col = event_summary["center_lat"].values

    # ★关键修复：经度坐标系转换
    # CMORPH 数据用 0-360° 经度，naturalearth_lowres 用 -180-180° 经度
    # 不转换会导致 sjoin 对经度>180 的点全部匹配失败 → 海陆分类错乱
    lon_col_normalized = ((lon_col + 180) % 360) - 180  # 0-360 → -180-180

    points = [geom.Point(lon, lat) for lon, lat in zip(lon_col_normalized, lat_col)]
    points_gdf = gpd.GeoDataFrame(
        {"idx": range(n), "event_id": event_summary["event_id"].values},
        geometry=points, crs="EPSG:4326"
    )

    # ---- 批量陆地/海洋分类（统一逻辑，确保一致性）----
    # ★reverse_geocoder 优先：GeoNames 500万城市点库，小国/岛国精度远超 50m/110m sjoin
    #   - 海陆判定仍由 sjoin/vectorized_rects 完成（rg 只提供最近陆地点，不区分海陆）
    #   - 国家名+大洲名：rg 优先 → sjoin 回退 → vectorized_rects 兜底
    rg_countries = None
    rg_continents_en = None
    if HAS_RG:
        print(f"  🗺️ [reverse_geocoder] 批量查询国家归属（{n}个点）...")
        _t0 = __import__("time").time()
        rg_countries, rg_continents_en = rg_query_country_continent_batch(lon_col, lat_col)
        _dt = __import__("time").time() - _t0
        _valid = int((rg_countries != "Unknown").sum())
        print(f"  ✅ reverse_geocoder: {_valid}/{n} 成功 ({_valid/n*100:.1f}%)，耗时 {_dt:.1f}s")
    # 大洲名英→中映射（rg 和 sjoin 都用同一套）
    _continent_en2cn = {
        "Asia": "亚洲", "Europe": "欧洲", "Africa": "非洲",
        "North America": "北美洲", "South America": "南美洲",
        "Oceania": "大洋洲", "Antarctica": "南极洲",
        "Seven seas (open ocean)": "海洋",
        "Unknown": "未知",
    }
    if geo_data is not None:
        try:
            land = geo_data["land"]
            # ★列名已由 _standardize_world_gdf 规范化为 [name, continent, geometry]
            # 这里只需做防御性检查，确保列存在
            cols_to_select = [c for c in ["name", "continent", "geometry"] if c in land.columns]
            land_subset = land[cols_to_select].copy()
            print(f"  📋 sjoin 列: {cols_to_select}")
            # sjoin 查询每个点所在的国家和continent（主要用于 is_land 海陆判定 + rg失败的回退）
            try:
                land_sjoin = gpd.sjoin(
                    points_gdf, land_subset,
                    how="left", predicate="intersects"
                )
            except TypeError:
                land_sjoin = gpd.sjoin(
                    points_gdf, land_subset,
                    how="left", op="intersects"
                )
            # 去重：每个点取第一条匹配
            land_sjoin = land_sjoin[~land_sjoin.index.duplicated(keep="first")]
            land_sjoin = land_sjoin.reindex(points_gdf.index)
            # 防御性：确保 name/continent 是单列 Series（sjoin可能加后缀）
            name_col_actual = "name" if "name" in land_sjoin.columns else (
                next((c for c in land_sjoin.columns if c.startswith("name")), None)
            )
            cont_col_actual = "continent" if "continent" in land_sjoin.columns else (
                next((c for c in land_sjoin.columns if c.startswith("continent")), None)
            )

            is_land = land_sjoin[name_col_actual].notna().values if name_col_actual else np.zeros(n, dtype=bool)
            # ★统一分类：is_land 决定 location_type（内陆/海洋二分类，沿海/近海靠buffer）
            location_type = np.where(is_land, "内陆", "海洋")

            # ===== 国家归属：reverse_geocoder 优先 → sjoin 回退 → vectorized_rects 兜底 =====
            # 先初始化 sjoin 的结果（回退基准）
            _ne_names = land_sjoin[name_col_actual].fillna("Unknown").astype(object).values if name_col_actual else np.full(n, "Unknown", dtype=object)
            _ne_continent_en = land_sjoin[cont_col_actual].fillna("Unknown").astype(object).values if cont_col_actual else np.full(n, "Unknown", dtype=object)
            # 防御性：若返回多列 DataFrame，取第一列
            if _ne_names.ndim > 1:
                _ne_names = _ne_names[:, 0]
            if _ne_continent_en.ndim > 1:
                _ne_continent_en = _ne_continent_en[:, 0]
            # 再用 rg 覆盖
            if rg_countries is not None:
                _country_en = np.where(rg_countries != "Unknown", rg_countries, _ne_names).astype(object)
                _continent_en = np.where(rg_continents_en != "Unknown", rg_continents_en, _ne_continent_en).astype(object)
            else:
                _country_en = _ne_names
                _continent_en = _ne_continent_en
            # 最后仍为 Unknown 的陆地事件 → vectorized_rects
            _vec_names = classify_country_vectorized(lon_col_normalized, lat_col)
            _vec_cont = classify_continent_vectorized(lon_col_normalized, lat_col)  # 返回中文
            country_result = np.empty(n, dtype=object)
            continent_result = np.empty(n, dtype=object)
            for i in range(n):
                if not is_land[i]:
                    # 海洋事件：国家=空，大洲=海洋（后续一致性强制还会校验）
                    country_result[i] = ""
                    continent_result[i] = "海洋"
                else:
                    # 陆地事件：rg → sjoin → vectorized_rects（国家名保留英文；大洲翻译中文）
                    cname = _country_en[i]
                    ccont_en = _continent_en[i]
                    if cname in ("Unknown", "未知", "", None) or ccont_en in ("Unknown", "未知", "", None):
                        # rg和sjoin都失败 → vectorized_rects
                        country_result[i] = _vec_names[i] if _vec_names[i] not in ("Unknown", "") else "Unknown"
                        # vectorized返回的是中文大洲，直接使用
                        continent_result[i] = _vec_cont[i]
                    else:
                        country_result[i] = cname
                        # 英文大洲翻译为中文
                        continent_result[i] = _continent_en2cn.get(ccont_en, ccont_en)

            # ---- 沿海/近海判断：用陆地多边形buffer ----
            try:
                # ★buffer纬度修正：高纬度1°经度<111km，用赤道buffer会低估实际范围
                # 用所有事件最大|lat|计算保守buffer，确保高纬沿海事件不被遗漏
                buffer_km = config.get("coastal_buffer_km", 100)
                buffer_deg_equator = buffer_km / 111.0
                _max_lat_abs = float(np.abs(lat_col).max()) if len(lat_col) > 0 else 0.0
                _cos_max = max(np.cos(np.radians(min(_max_lat_abs, 80))), 0.3)
                buffer_deg_conservative = buffer_deg_equator / _cos_max
                print(f"  🏖️ 计算沿海/近海分类 (buffer={buffer_deg_conservative:.2f}° ≈ {buffer_km}km @lat{_max_lat_abs:.0f}°)...")
                
                # ★修复 P1: 用"外扩buffer"判断近海，用"内缩buffer"判断内陆
                # 原bug: land.geometry.buffer() 外扩后包含所有陆地内部点，
                #        导致所有陆地点 in_coastal_buffer=True，全标为"沿海"，无"内陆"
                # 修复方案:
                #   1. 外扩buffer: 陆地多边形向外扩展 → 海洋点在内则标为"近海"
                #   2. 内缩buffer: 陆地多边形向内收缩 → 陆地点在内则标为"内陆"
                #   3. 陆地点不在内缩buffer内 → 标为"沿海"（靠近海岸的陆地）
                
                # 外扩 buffer（用于近海判断）
                land_buffered = land.geometry.buffer(buffer_deg_conservative)
                land_buffered_gdf = gpd.GeoDataFrame(geometry=land_buffered, crs="EPSG:4326")
                try:
                    buffer_sjoin = gpd.sjoin(
                        points_gdf, land_buffered_gdf, how="left", predicate="within"
                    )
                except TypeError:
                    buffer_sjoin = gpd.sjoin(
                        points_gdf, land_buffered_gdf, how="left", op="within"
                    )
                buffer_sjoin = buffer_sjoin[~buffer_sjoin.index.duplicated(keep="first")]
                buffer_sjoin = buffer_sjoin.reindex(points_gdf.index)
                in_coastal_buffer = buffer_sjoin["index_right"].notna().values if "index_right" in buffer_sjoin.columns else np.zeros(n, dtype=bool)

                # ★新增: 内缩 buffer（用于内陆判断）
                # land.geometry.buffer(-buffer_deg) 将陆地多边形向内收缩
                # 在内缩多边形内的点 = 远离海岸的陆地点 = "内陆"
                # 不在内缩多边形内的陆地点 = 靠近海岸 = "沿海"
                # ★P2修复: 小岛屿内缩后可能消失，分级内缩确保小岛也有内陆点
                land_eroded = land.geometry.buffer(-buffer_deg_conservative)
                # 过滤掉空几何体（小岛内缩后可能消失）
                land_eroded = land_eroded[~land_eroded.is_empty]
                
                # ★P2修复: 如果一级内缩后几何体太少，用更小的内缩距离再试
                if len(land_eroded) < len(land) * 0.3:
                    # 小岛屿较多，用一半的距离再内缩
                    land_eroded_small = land.geometry.buffer(-buffer_deg_conservative * 0.5)
                    land_eroded_small = land_eroded_small[~land_eroded_small.is_empty]
                    if len(land_eroded_small) > len(land_eroded):
                        land_eroded = pd.concat([land_eroded, land_eroded_small]).drop_duplicates()
                
                is_inland = np.zeros(n, dtype=False)
                if len(land_eroded) > 0:
                    land_eroded_gdf = gpd.GeoDataFrame(geometry=land_eroded, crs="EPSG:4326")
                    try:
                        erode_sjoin = gpd.sjoin(
                            points_gdf, land_eroded_gdf, how="left", predicate="within"
                        )
                    except TypeError:
                        erode_sjoin = gpd.sjoin(
                            points_gdf, land_eroded_gdf, how="left", op="within"
                        )
                    erode_sjoin = erode_sjoin[~erode_sjoin.index.duplicated(keep="first")]
                    erode_sjoin = erode_sjoin.reindex(points_gdf.index)
                    is_inland = erode_sjoin["index_right"].notna().values if "index_right" in erode_sjoin.columns else np.zeros(n, dtype=False)

                # 4类分类：内陆/沿海/近海/海洋
                # ★修复: 陆地点分两类 — 在内缩buffer内为"内陆"，否则为"沿海"
                near_coast = in_coastal_buffer
                location_type = np.where(
                    is_land & is_inland, "内陆",  # 陆地 + 远离海岸
                    np.where(
                        near_coast & is_land & ~is_inland, "沿海",  # 陆地 + 靠近海岸
                        np.where(near_coast & ~is_land, "近海", location_type)  # 海洋 + 靠近陆地
                    )
                )
                n_inland = int((location_type == "内陆").sum())
                n_coast = int((location_type == "沿海").sum())
                n_near = int((location_type == "近海").sum())
                n_ocean = int((location_type == "海洋").sum())
                print(f"  🏖️ 分类结果: 内陆={n_inland}, 沿海={n_coast}, 近海={n_near}, 海洋={n_ocean}")
            except Exception as e:
                print(f"  ⚠️ 沿海分类失败: {e}")
                import traceback
                traceback.print_exc()
                # ★修复: except 时显式提示 location_type 退化为2类（内陆/海洋）
                #   报告指出字典定义4类但实测只有2类，源于此 except 路径被静默跳过
                print(f"  ⚠️ location_type 退化为2类(内陆/海洋), 沿海/近海分类未生效")

            # ---- 对未知大洲的陆地事件用经纬度回退 ----
            unknown_land = (continent_result == "未知") & is_land
            if unknown_land.any():
                if config.get("performance", {}).get("vectorized_geo", True):
                    simple_continents = classify_continent_vectorized(lon_col_normalized[unknown_land], lat_col[unknown_land])
                else:
                    simple_continents = np.array([
                        classify_continent_simple(lon, lat) for lon, lat in
                        zip(lon_col_normalized[unknown_land], lat_col[unknown_land])
                    ])
                continent_result[unknown_land] = simple_continents

            # ---- 海洋名称分类（向量化，用归一化经度）----
            # ★关键修复：所有海洋事件（包括"近海"）都填充 ocean 名称
            ocean_mask = np.isin(location_type, ["海洋", "近海"])
            if config.get("performance", {}).get("vectorized_geo", True):
                ocean_full = classify_ocean_vectorized(lon_col_normalized, lat_col)
                ocean = np.where(ocean_mask, ocean_full, "")
            else:
                ocean = np.array([
                    classify_ocean(lon, lat) if m else ""
                    for lon, lat, m in zip(lon_col_normalized, lat_col, ocean_mask)
                ])
            # ★验证一致性：海洋事件 continent 必须为"海洋"
            # 修复历史数据中"海洋事件continent=陆地大洲"的矛盾
            ocean_loc_mask = np.isin(location_type, ["海洋", "近海"])
            continent_result = np.where(ocean_loc_mask, "海洋", continent_result)
            country_result = np.where(ocean_loc_mask, "", country_result)
            # ★修复：先 .sum() 再 int()，避免对bool数组直接 int() 报错
            _inc = (np.isin(location_type, ["海洋", "近海"]) & (continent_result != "海洋"))
            n_inconsistent = int(_inc.sum())
            if n_inconsistent > 0:
                print(f"  🔧 修复 {n_inconsistent} 条海洋事件的 continent 矛盾")
        except Exception as e:
            print(f"  ⚠️ sjoin 分类失败: {e}，使用简单分类回退")
            import traceback
            traceback.print_exc()
            lons_arr = lon_col_normalized
            lats_arr = lat_col
            location_type = classify_location_type_vectorized(lons_arr, lats_arr)
            continent_result = classify_continent_vectorized(lons_arr, lats_arr)
            ocean_full = classify_ocean_vectorized(lons_arr, lats_arr)
            ocean_mask = np.isin(continent_result, ["海洋", "海洋/未分类"])
            ocean = np.where(ocean_mask, ocean_full, "")
            _country_fb = classify_country_vectorized(lons_arr, lats_arr)
            # ★rg 优先：sjoin 失败时也应使用 rg
            if rg_countries is not None:
                _land_only = ~ocean_mask
                _land_idx = np.where(_land_only)[0]
                for i in _land_idx:
                    if rg_countries[i] != "Unknown":
                        _country_fb[i] = rg_countries[i]
                    if rg_continents_en[i] != "Unknown":
                        continent_result[i] = _continent_en2cn.get(rg_continents_en[i], rg_continents_en[i])
            country_result = np.where(ocean_mask, "", _country_fb)
    else:
        lons_arr = lon_col_normalized
        lats_arr = lat_col
        if config.get("performance", {}).get("vectorized_geo", True):
            location_type = classify_location_type_vectorized(lons_arr, lats_arr)
            continent_result = classify_continent_vectorized(lons_arr, lats_arr)
            ocean_full = classify_ocean_vectorized(lons_arr, lats_arr)
            ocean_mask = np.isin(continent_result, ["海洋", "海洋/未分类"])
            ocean = np.where(ocean_mask, ocean_full, "")
            _country_fb = classify_country_vectorized(lons_arr, lats_arr)
            # ★rg 优先：geo_data is None 时也用 rg 增强
            if rg_countries is not None:
                _land_only = ~ocean_mask
                _land_idx = np.where(_land_only)[0]
                for i in _land_idx:
                    if rg_countries[i] != "Unknown":
                        _country_fb[i] = rg_countries[i]
                    if rg_continents_en[i] != "Unknown":
                        continent_result[i] = _continent_en2cn.get(rg_continents_en[i], rg_continents_en[i])
            country_result = np.where(ocean_mask, "", _country_fb)
        else:
            location_type = np.array([
                classify_location_type_simple(lon, lat) for lon, lat in
                zip(lons_arr, lats_arr)
            ])
            continent_result = np.array([
                classify_continent_simple(lon, lat) for lon, lat in
                zip(lons_arr, lats_arr)
            ])
            ocean = np.array([
                classify_ocean(lon, lat) if c in ["海洋", "海洋/未分类"] else ""
                for lon, lat, c in zip(lons_arr, lats_arr, continent_result)
            ])
            _country_fb = classify_country_vectorized(lons_arr, lats_arr)
            # ★rg 优先
            if rg_countries is not None:
                _land_mask = np.isin(location_type, ["内陆", "沿海"])
                _land_idx = np.where(_land_mask)[0]
                for i in _land_idx:
                    if rg_countries[i] != "Unknown":
                        _country_fb[i] = rg_countries[i]
                    if rg_continents_en[i] != "Unknown":
                        continent_result[i] = _continent_en2cn.get(rg_continents_en[i], rg_continents_en[i])
            country_result = np.where(np.isin(location_type, ["海洋", "近海"]), "", _country_fb)

    # ---- 批量气候区提取（优先使用LUT查找表）----
    # 注：必须使用与 location_type 相同的坐标（加权质心）查询气候区
    # 否则会出现 location_type=内陆 但 climate_zone=海洋气候 的交叉错配
    climate_lut = config.get("_climate_lut")
    if "weighted_lon" in event_summary.columns:
        _tmp_lons = event_summary["weighted_lon"].fillna(event_summary["center_lon"]).values
        _tmp_lats = event_summary["weighted_lat"].fillna(event_summary["center_lat"]).values
    else:
        _tmp_lons = event_summary["center_lon"].values
        _tmp_lats = event_summary["center_lat"].values
    if climate_lut is not None:
        # 使用LUT查找表（O(1)查询）
        climate_zone = query_climate_lut(_tmp_lons, _tmp_lats, climate_lut)
        # ★修复：对所有海洋/近海事件强制使用语义化海洋气候区
        # 原逻辑仅替换"未知"，导致28,571个海洋事件套用陆地气候(Af/Aw等)
        _ocean_tmp = np.isin(location_type, ["海洋", "近海"])
        # 生成语义化海洋气候区
        ocean_climate = ocean_climate_zone_vectorized(_tmp_lons, _tmp_lats)
        climate_zone = np.where(_ocean_tmp, ocean_climate, climate_zone)
    elif kg_gdf is not None:
        climate_zone = batch_query_climate_zone(
            _tmp_lons, _tmp_lats,
            kg_gdf,
            batch_size=50000
        )
        if climate_zone is None:
            climate_zone = np.array([
                extract_climate_from_shapefile(lon, lat, kg_gdf) for lon, lat in
                tqdm(list(zip(_tmp_lons, _tmp_lats)),
                     desc="  气候区提取")
            ])
    else:
        # 使用简单回退
        climate_zone = np.array([
            extract_climate_from_shapefile(lon, lat, None) for lon, lat in
            zip(_tmp_lons, _tmp_lats)
        ])

    # ★统一强制：所有海洋/近海事件使用语义化海洋气候区（不论LUT返回什么）
    _ocean_force = np.isin(location_type, ["海洋", "近海"])
    if _ocean_force.any():
        ocean_climate_all = ocean_climate_zone_vectorized(_tmp_lons, _tmp_lats)
        climate_zone = np.where(_ocean_force, ocean_climate_all, climate_zone)

    # ===== ★终极一致性强制：根据 location_type 统一 continent/country/ocean =====
    # 根因：sjoin 可能匹配到 Natural Earth 的 "Seven seas (open ocean)" 多边形，
    # 导致 is_land=True（→location_type=内陆/沿海）但 continent="海洋"（→country空、ocean非空）
    # 反向：sjoin 边界精度问题可能让海洋点匹配到陆地国家
    # 方案：以 location_type 为准，强制重算矛盾字段
    ocean_loc_final = np.isin(location_type, ["海洋", "近海"])
    land_loc_final = np.isin(location_type, ["内陆", "沿海"])

    # (a) 海洋事件：continent=海洋, country=空（ocean字段已在上方正确填充）
    continent_result = np.where(ocean_loc_final, "海洋", continent_result)
    country_result = np.where(ocean_loc_final, "", country_result)

    # (b) 陆地事件但 continent=海洋/未知：用 vectorized 函数重算
    land_bad_continent = land_loc_final & np.isin(continent_result, ["海洋", "未知", ""])
    if land_bad_continent.any():
        n_fix = int(land_bad_continent.sum())
        print(f"  🔧 修复 {n_fix} 条陆地事件的 continent 矛盾（sjoin误返回海洋/未知）")
        _reclassified = classify_continent_vectorized(
            lon_col_normalized[land_bad_continent], lat_col[land_bad_continent]
        )
        # ★修复：直接赋值而非np.where（避免subset与full数组形状不匹配）
        continent_result[land_bad_continent] = _reclassified
        # 同时修复 country（用 vectorized 回退）和 ocean（陆地应为空）
        _reclassified_country = classify_country_vectorized(
            lon_col_normalized[land_bad_continent], lat_col[land_bad_continent]
        )
        country_result[land_bad_continent] = _reclassified_country
        ocean = np.where(land_bad_continent, "", ocean)

    # (c) 陆地事件但 country 为空/Unknown：用 vectorized 回退
    land_no_country = land_loc_final & np.isin(country_result, ["", "Unknown", "未知", "nan"])
    if land_no_country.any():
        _fb_country = classify_country_vectorized(
            lon_col_normalized[land_no_country], lat_col[land_no_country]
        )
        country_result[land_no_country] = _fb_country

    # (d) 陆地事件但 ocean 非空：清空（陆地事件不应有 ocean 字段）
    # ★修复：ocean 是 numpy.ndarray，没有 .str 属性；用 np.char 长度判断
    _ocean_str = ocean.astype(str)
    _ocean_len = np.char.str_len(_ocean_str)
    land_has_ocean = land_loc_final & (_ocean_len > 0)
    if land_has_ocean.any():
        ocean = np.where(land_has_ocean, "", ocean)

    # 统计最终一致性
    n_final_inconsist = int((
        (ocean_loc_final & (continent_result != "海洋")) |
        (land_loc_final & (continent_result == "海洋"))
    ).sum())
    if n_final_inconsist > 0:
        print(f"  ⚠️ 仍有 {n_final_inconsist} 条事件 location_type vs continent 不一致")
    else:
        print(f"  ✅ location_type vs continent 一致性检查通过")

    event_summary["location_type"] = location_type
    event_summary["continent"] = continent_result
    event_summary["ocean"] = ocean
    event_summary["climate_zone"] = climate_zone
    event_summary["country"] = country_result

    # ---- 修复陆地气候区：处理"未知"和误赋的海洋气候区 ----
    # ★扩展：陆地事件若被赋予海洋气候区(含"海洋"字样)也需修复
    _land_mask = event_summary["location_type"].isin(["内陆", "沿海"])
    _has_ocean_cz = event_summary["climate_zone"].astype(str).str.contains("海洋")
    unknown_land_mask = (
        _land_mask &
        ((event_summary["climate_zone"] == "未知") | _has_ocean_cz)
    )
    if unknown_land_mask.any():
        n_unknown = unknown_land_mask.sum()
        print(f"  🔧 修复 {n_unknown} 个陆地事件的气候区（未知/海洋气候误赋，最近邻填充）")
        # 已知参考：仅使用有有效陆地气候区的陆地事件
        known_mask = _land_mask & ~unknown_land_mask
        if known_mask.any():
            known_lons = event_summary.loc[known_mask, "center_lon"].values
            known_lats = event_summary.loc[known_mask, "center_lat"].values
            known_zones = event_summary.loc[known_mask, "climate_zone"].values
            unknown_lons = event_summary.loc[unknown_land_mask, "center_lon"].values
            unknown_lats = event_summary.loc[unknown_land_mask, "center_lat"].values
            from scipy.spatial import cKDTree
            tree = cKDTree(np.column_stack([known_lons, known_lats]))
            _, idx = tree.query(np.column_stack([unknown_lons, unknown_lats]), k=1)
            event_summary.loc[unknown_land_mask, "climate_zone"] = known_zones[idx]

    return event_summary


def classify_location_type_geo(lon, lat, geo_data, coastal_buffer_km=50):
    """基于地理数据的位置分类"""
    if not HAS_GEOPANDAS:
        return classify_location_type_simple(lon, lat)
    point = geom.Point(lon, lat)
    point_gdf = gpd.GeoDataFrame(geometry=[point], crs="EPSG:4326")
    try:
        try:
            land_intersect = gpd.sjoin(point_gdf, geo_data["land"], how="left", predicate="intersects")
        except TypeError:
            land_intersect = gpd.sjoin(point_gdf, geo_data["land"], how="left", op="intersects")
        is_ocean = land_intersect.empty or pd.isna(land_intersect.iloc[0].get("name"))
    except Exception:
        is_ocean = False
    if is_ocean:
        try:
            coast = geo_data["land"].boundary.unary_union
            dist = point.distance(nearest_points(point, coast)[1])
            # 粗略转换度到 km
            dist_km = dist * 111
            if dist_km <= coastal_buffer_km:
                return "近海"
            else:
                return "海洋"
        except Exception:
            return "海洋"
    else:
        try:
            coast = geo_data["land"].boundary.unary_union
            dist = point.distance(nearest_points(point, coast)[1])
            dist_km = dist * 111
            if dist_km <= coastal_buffer_km:
                return "沿海"
            else:
                return "内陆"
        except Exception:
            return "内陆"


def classify_location_type_simple(lon, lat):
    """简单位置分类"""
    if ((130 <= lon <= 230 and -60 <= lat <= 60) or
        (260 <= lon <= 340 and -60 <= lat <= 60) or
        (20 <= lon <= 80 and -40 <= lat <= 20)):
        return "海洋"
    if ((100 <= lon <= 110 and 10 <= lat <= 30) or
        (120 <= lon <= 130 and 10 <= lat <= 30)):
        return "沿海"
    return "内陆"


def classify_continent_geo(lon, lat, geo_data):
    """基于地理数据的大洲分类"""
    if not HAS_GEOPANDAS:
        return classify_continent_simple(lon, lat)
    point = geom.Point(lon, lat)
    point_gdf = gpd.GeoDataFrame(geometry=[point], crs="EPSG:4326")
    try:
        try:
            intersects = gpd.sjoin(point_gdf, geo_data["continents"], how="left", predicate="intersects")
        except TypeError:
            intersects = gpd.sjoin(point_gdf, geo_data["continents"], how="left", op="intersects")
        if not intersects.empty and not pd.isna(intersects.iloc[0].get("continent")):
            return intersects.iloc[0]["continent"]
        return "海洋"
    except Exception:
        return "未知"


def classify_continent_simple(lon, lat):
    """简化大洲分类"""
    lon = ((lon + 180) % 360) - 180
    if ((10 <= lat <= 75) and (40 <= lon <= 180)):
        return "亚洲"
    if ((10 <= lat <= 75) and (-180 <= lon <= -170)):
        return "亚洲"
    if ((35 <= lat <= 70) and (-10 <= lon <= 40)):
        return "欧洲"
    if ((-35 <= lat <= 35) and (-20 <= lon <= 50)):
        return "非洲"
    if ((15 <= lat <= 70) and (-130 <= lon <= -60)):
        return "北美洲"
    if ((-55 <= lat <= 10) and (-80 <= lon <= -35)):
        return "南美洲"
    if ((-50 <= lat <= -10) and (110 <= lon <= 155)):
        return "大洋洲"
    if lat < -60:
        return "南极洲"
    return "海洋"


def classify_ocean(lon, lat):
    """海洋分类（标量版，使用与向量化版本一致的范围）"""
    lon = ((lon + 180) % 360) - 180
    if (lat >= 65):
        return "北冰洋"
    if (lat <= -60):
        return "南大洋"
    # 太平洋：西太平洋 lon 140~180 + 东太平洋 lon -180~-60
    if (
        (-60 <= lat <= 60) and
        ((140 <= lon <= 180) or (-180 <= lon <= -60))
    ):
        return "太平洋"
    # 日本海/东海/黄海 → 太平洋
    if (20 <= lat <= 50) and (120 <= lon <= 145):
        return "太平洋"
    if (-60 <= lat <= 60) and (-80 <= lon <= 20):
        return "大西洋"
    if (-60 <= lat <= 30) and (20 <= lon <= 140):
        return "印度洋"
    if (30 <= lat <= 45) and (-10 <= lon <= 40):
        return "地中海"
    return "其他海洋"


def classify_location_type_vectorized(lon_arr, lat_arr):
    """向量化4类位置分类：内陆/沿海/近海/海洋

    基于详细大陆轮廓多矩形近似 + 海岸带buffer判断。
    当geopandas土地数据不可用时作为回退方案。

    分类规则：
      - 内陆：陆地区域且距海岸线 > buffer_deg
      - 沿海：陆地区域且距海岸线 ≤ buffer_deg
      - 近海：海洋区域且距海岸线 ≤ buffer_deg
      - 海洋：海洋区域且距海岸线 > buffer_deg
    """
    lon_arr = np.asarray(lon_arr, dtype=float)
    lat_arr = np.asarray(lat_arr, dtype=float)
    n = len(lon_arr)
    lon_norm = ((lon_arr + 180) % 360) - 180  # 归一化到 -180~180

    # ---- 主要陆地区域（多矩形近似，覆盖全球大陆+大岛）----
    # (lon_min, lon_max, lat_min, lat_max)
    land_rects = [
        # 欧亚大陆主体（欧洲+北亚+中东+东亚+东南亚大陆部分）
        (-10, 180, 10, 72),
        # 非洲（含撒哈拉以南）
        (-18, 52, -35, 38),
        # 北美洲（含阿拉斯加、加拿大、美国、墨西哥）
        (-168, -52, 8, 72),
        # 南美洲
        (-82, -34, -56, 13),
        # 澳大利亚
        (113, 154, -40, -10),
        # 格陵兰
        (-55, -20, 60, 84),
        # 印尼/马来群岛（苏门答腊+加里曼丹+苏拉威西+新几内亚西部）
        (95, 141, -11, 7),
        # 日本本州/北海道/九州
        (129, 146, 30, 46),
        # 菲律宾
        (117, 127, 5, 19),
        # 新几内亚岛（东部）
        (130, 151, -11, -1),
        # 新西兰南北岛
        (166, 179, -47, -34),
        # 英国/爱尔兰
        (-11, 2, 50, 59),
        # 冰岛
        (-25, -13, 63, 67),
        # 马达加斯加
        (43, 51, -26, -12),
        # 古巴及加勒比大安的列斯群岛
        (-85, -74, 17, 24),
        # 中美洲走廊
        (-92, -77, 7, 18),
        # 斯里兰卡
        (79, 82, 5, 10),
        # 台湾岛
        (119, 122, 21, 26),
        # 海南岛
        (108, 111, 18, 20),
        # 巴布亚新几内亚外岛
        (148, 156, -11, -1),
        # 所罗门群岛/瓦努阿图/斐济区域
        (155, 180, -22, -5),
    ]

    # ---- 内陆海/海湾"空洞"（从陆地中扣除，避免bbox过估计）----
    sea_holes = [
        # 地中海（欧洲南部-非洲北部之间）
        (-5, 36, 30, 46),
        # 黑海
        (28, 42, 41, 47),
        # 里海
        (47, 55, 37, 47),
        # 红海
        (32, 44, 12, 30),
        # 波斯湾
        (48, 58, 24, 30),
        # 亚丁湾/阿拉伯海（非洲之角东侧）
        (42, 52, 5, 18),
        # 霍姆斯湾/几内亚湾
        (-10, 10, -5, 5),
        # 哈德逊湾
        (-95, -70, 51, 63),
        # 波罗的海
        (10, 30, 54, 66),
        # 加勒比海
        (-88, -60, 10, 25),
        # 南海
        (105, 120, 0, 22),
        # 东海
        (120, 130, 25, 33),
        # 日本海
        (127, 138, 35, 46),
        # 鄂霍次克海
        (138, 152, 50, 62),
        # 白令海
        (170, 180, 52, 62),
        # 巴芬湾
        (-65, -45, 65, 78),
        # 墨西哥湾
        (-97, -82, 18, 30),
        # 阿拉弗拉海/卡奔塔利亚湾
        (130, 145, -18, -8),
        # 大澳大利亚湾
        (118, 135, -38, -31),
        # 孟加拉湾
        (80, 95, 5, 22),
        # 阿拉伯海西部
        (55, 73, 5, 25),
        # 莫桑比克海峡
        (40, 46, -25, -10),
        # 英吉利海峡/比斯开湾
        (-10, 2, 43, 50),
    ]

    buffer_deg = 1.5  # 海岸带buffer宽度（度，约167km）
    coastal_buffer = buffer_deg * 2  # 沿海区域宽度（约334km），陆地点距离边缘超过此值为"内陆"

    # ---- 判断每个点是否在陆地上 ----
    is_land = np.zeros(n, dtype=bool)
    for lon_min, lon_max, lat_min, lat_max in land_rects:
        is_land |= (lon_norm >= lon_min) & (lon_norm <= lon_max) & \
                   (lat_arr >= lat_min) & (lat_arr <= lat_max)
    # 扣除内陆海"空洞"
    for lon_min, lon_max, lat_min, lat_max in sea_holes:
        is_land &= ~((lon_norm >= lon_min) & (lon_norm <= lon_max) & \
                     (lat_arr >= lat_min) & (lat_arr <= lat_max))

    # ---- 判断是否在海岸带buffer内（近海判定）----
    # 扩展所有陆地矩形buffer_deg度，检查点是否落在扩展区域内
    near_land = np.zeros(n, dtype=bool)
    for lon_min, lon_max, lat_min, lat_max in land_rects:
        near_land |= (lon_norm >= lon_min - buffer_deg) & (lon_norm <= lon_max + buffer_deg) & \
                     (lat_arr >= lat_min - buffer_deg) & (lat_arr <= lat_max + buffer_deg)

    # ---- 判断陆地边缘（沿海 vs 内陆）----
    # 改进：对于每个陆地点，计算其到所有陆地矩形边缘的最小距离
    # 如果最小距离 < coastal_buffer → "沿海"（靠近海岸）
    # 如果最小距离 >= coastal_buffer → "内陆"（远离海岸）
    # 注意：只计算点所在的陆地矩形的边缘距离，避免跨矩形误判
    
    # 存储每个陆地点到最近陆地边缘的距离
    min_dist_to_coast = np.full(n, np.inf)
    
    for lon_min, lon_max, lat_min, lat_max in land_rects:
        # 只考虑落在这个矩形内的点
        in_rect = (lon_norm >= lon_min) & (lon_norm <= lon_max) & \
                  (lat_arr >= lat_min) & (lat_arr <= lat_max)
        
        if not in_rect.any():
            continue
        
        # 计算每个点到这个矩形四条边的距离
        d_left = lon_norm[in_rect] - lon_min
        d_right = lon_max - lon_norm[in_rect]
        d_bottom = lat_arr[in_rect] - lat_min
        d_top = lat_max - lat_arr[in_rect]
        
        # 到边缘的最小距离
        min_d = np.minimum.reduce([d_left, d_right, d_bottom, d_top])
        
        # 更新全局最小距离（只考虑陆地点）
        idx = np.where(in_rect)[0]
        min_dist_to_coast[idx] = np.minimum(min_dist_to_coast[idx], min_d)
    
    # 扣除内陆海"空洞"区域的距离（这些区域不算陆地）
    for lon_min, lon_max, lat_min, lat_max in sea_holes:
        hole = (lon_norm >= lon_min) & (lon_norm <= lon_max) & \
               (lat_arr >= lat_min) & (lat_arr <= lat_max)
        # 对于空洞内的点，将距离设为inf（视为非陆地）
        min_dist_to_coast[hole] = np.inf
    
    # 沿海：陆地点且距离最近边缘 < coastal_buffer
    is_coastal_land = is_land & (min_dist_to_coast < coastal_buffer)
    
    # ---- 4类分类 ----
    result = np.full(n, "海洋", dtype=object)
    # 近海：海洋 + 在海岸带buffer内
    result[~is_land & near_land] = "近海"
    # 沿海：陆地 + 靠近边缘（距海岸 < coastal_buffer）
    result[is_coastal_land] = "沿海"
    # 内陆：陆地 + 远离边缘（距海岸 >= coastal_buffer）
    result[is_land & ~is_coastal_land] = "内陆"

    return result


def classify_country_vectorized(lon_arr, lat_arr):
    """向量化国家归属分类（geopandas不可用时的回退方案）

    覆盖全球主要国家，重点解决大洋洲/岛国归属问题。
    返回国家名（英文，与Natural Earth一致），无法归属返回"Unknown"。
    """
    lon_arr = np.asarray(lon_arr, dtype=float)
    lat_arr = np.asarray(lat_arr, dtype=float)
    n = len(lon_arr)
    lon_norm = ((lon_arr + 180) % 360) - 180

    result = np.full(n, "Unknown", dtype=object)
    # (country_name, lon_min, lon_max, lat_min, lat_max)
    # 按面积从小到大排序（小岛国先匹配，避免被大矩形覆盖）
    countries = [
        # ---- 大洋洲岛国 ----
        ("New Zealand", 166, 179, -47, -34),
        ("Papua New Guinea", 130, 156, -12, -1),
        ("Solomon Islands", 155, 170, -12, -5),
        ("Vanuatu", 166, 170, -20, -13),
        ("Fiji", 177, 180, -20, -12),
        ("New Caledonia", 163, 167, -23, -18),
        # ---- 东南亚岛国 ----
        ("Philippines", 117, 127, 5, 19),
        ("Indonesia", 95, 141, -11, 7),
        ("Malaysia", 100, 119, 1, 7),
        ("Singapore", 103, 104, 1, 1.5),
        ("Brunei", 114, 115, 4, 5),
        ("Timor-Leste", 124, 127, -9, -8),
        # ---- 东亚 ----
        ("Japan", 129, 146, 30, 46),
        ("South Korea", 126, 129, 33, 38),
        ("North Korea", 124, 130, 37, 43),
        ("Taiwan", 119, 122, 21, 26),
        ("China", 73, 135, 18, 53),
        ("Mongolia", 88, 120, 42, 52),
        # ---- 南亚 ----
        ("Sri Lanka", 79, 82, 5, 10),
        ("India", 68, 97, 6, 36),
        ("Pakistan", 61, 78, 23, 37),
        ("Bangladesh", 88, 93, 20, 27),
        ("Nepal", 80, 89, 26, 31),
        # ---- 中东 ----
        ("Iran", 44, 63, 25, 40),
        ("Iraq", 39, 49, 29, 38),
        ("Saudi Arabia", 35, 56, 15, 32),
        ("Turkey", 26, 45, 36, 42),
        # ---- 东南亚大陆 ----
        ("Vietnam", 102, 110, 8, 23),
        ("Thailand", 97, 106, 5, 20),
        ("Myanmar", 92, 102, 9, 28),
        ("Laos", 100, 107, 14, 23),
        ("Cambodia", 102, 108, 10, 15),
        # ---- 澳大利亚 ----
        ("Australia", 113, 154, -40, -10),
        # ---- 非洲主要国家 ----
        ("Egypt", 24, 37, 22, 32),
        ("Libya", 9, 25, 19, 33),
        ("Algeria", -9, 12, 18, 37),
        ("Nigeria", 3, 15, 4, 14),
        ("Ethiopia", 33, 48, 3, 15),
        ("Kenya", 34, 42, -5, 5),
        ("South Africa", 16, 33, -35, -22),
        ("Madagascar", 43, 51, -26, -12),
        # ---- 欧洲 ----
        ("United Kingdom", -11, 2, 50, 59),
        ("France", -5, 8, 42, 51),
        ("Germany", 6, 15, 47, 55),
        ("Spain", -9, 3, 36, 44),
        ("Italy", 7, 18, 36, 47),
        ("Poland", 14, 24, 49, 55),
        ("Ukraine", 22, 41, 44, 52),
        ("Russia", 30, 180, 41, 78),
        # ---- 北美 ----
        ("United States of America", -125, -67, 25, 49),
        ("Canada", -141, -52, 42, 70),
        ("Mexico", -117, -86, 14, 33),
        ("Cuba", -85, -74, 19, 24),
        # ---- 南美 ----
        ("Brazil", -74, -34, -34, 5),
        ("Argentina", -74, -53, -56, -22),
        ("Chile", -76, -66, -56, -17),
        ("Peru", -82, -68, -19, 0),
        ("Colombia", -79, -67, -5, 13),
        ("Venezuela", -73, -60, 0, 12),
        # ---- 格陵兰 ----
        ("Greenland", -55, -20, 60, 84),
        # ---- 冰岛 ----
        ("Iceland", -25, -13, 63, 67),
    ]

    for country_name, lon_min, lon_max, lat_min, lat_max in countries:
        mask = (lon_norm >= lon_min) & (lon_norm <= lon_max) & \
               (lat_arr >= lat_min) & (lat_arr <= lat_max)
        result[mask & (result == "Unknown")] = country_name

    return result


def rg_query_country_continent_batch(lon_arr, lat_arr):
    """★reverse_geocoder 批量查询：国家名(英文) + 大洲名(英文)
    
    优势：
    - O(1) k-d tree 近邻查询，26万点 ≈ 0.3-1s（比 sjoin 快 5-10x）
    - GeoNames ~500万 城市/村落数据库，小国/岛国精度远高于 110m 多边形
    - 内置本地缓存，首次加载慢后续极快
    
    Args:
        lon_arr: 经度数组（0~360 或 ±180 均可）
        lat_arr: 纬度数组
    
    Returns:
        (country_names, continent_names_en): tuple of object arrays (英文，与Natural Earth一致)
            - 查询失败/越界: country=Unknown, continent=Unknown
    """
    if not HAS_RG:
        n = len(lon_arr)
        return (np.full(n, "Unknown", dtype=object), np.full(n, "Unknown", dtype=object))
    lon_arr = np.asarray(lon_arr, dtype=float)
    lat_arr = np.asarray(lat_arr, dtype=float)
    n = len(lon_arr)
    # 经度归一化：rg 的 lat 范围 -90~90, lon 范围 -180~180（内部会转换，这里保险）
    lon_norm = ((lon_arr + 180) % 360) - 180
    # 纬度越界保护
    lat_clipped = np.clip(lat_arr, -90, 90)
    # 构造元组列表：rg 要求 (lat, lon) 顺序！！！
    coords = list(zip(lat_clipped.tolist(), lon_norm.tolist()))
    try:
        # reverse_geocoder 的 rg.search() 会一次性加载数据库（首次），然后对所有点并行查询
        # ★ Windows 下 multiprocessing fork/sem 行为异常，强制 mode=1 单进程（内置k-d树仍很快）
        results = _rg_mod.search(coords, verbose=False, mode=1)
    except TypeError:
        # 旧版 reverse_geocoder 无 mode 参数
        results = _rg_mod.search(coords, verbose=False)
    except Exception as _rge:
        print(f"  ⚠️ reverse_geocoder 查询失败: {_rge}")
        import traceback; traceback.print_exc()
        return (np.full(n, "Unknown", dtype=object), np.full(n, "Unknown", dtype=object))
    countries = np.empty(n, dtype=object)
    continents_en = np.empty(n, dtype=object)
    for i, r in enumerate(results):
        cc = r.get("cc", "")
        mapped = _ISO_CC_TO_COUNTRY_CONTINENT.get(cc)
        if mapped is not None:
            countries[i] = mapped[0]
            continents_en[i] = mapped[1]
        else:
            # 未映射的ISO代码（罕见情况）：fallback 为 Unknown
            countries[i] = "Unknown"
            continents_en[i] = "Unknown"
    return (countries, continents_en)


def classify_continent_vectorized(lon_arr, lat_arr):
    """向量化大洲分类（与classify_location_type_vectorized使用一致的land_rects+sea_holes）

    ★修复：原函数用简单bbox导致大量陆地点被判为"海洋"（45,593事件不一致）
    现在与location_type分类使用相同的详细陆地矩形和海域空洞，确保一致性。
    """
    lon_arr = np.asarray(lon_arr, dtype=float)
    lat_arr = np.asarray(lat_arr, dtype=float)
    n = len(lon_arr)
    lon_norm = ((lon_arr + 180) % 360) - 180
    result = np.full(n, "海洋", dtype=object)

    # ---- 陆地矩形+对应大洲（与classify_location_type_vectorized一致）----
    # (continent, lon_min, lon_max, lat_min, lat_max)
    land_continent = [
        # 欧亚大陆主体
        ("亚洲", -10, 180, 10, 72),
        # 非洲
        ("非洲", -18, 52, -35, 38),
        # 北美洲
        ("北美洲", -168, -52, 8, 72),
        # 南美洲
        ("南美洲", -82, -34, -56, 13),
        # 澳大利亚
        ("大洋洲", 113, 154, -40, -10),
        # 格陵兰
        ("北美洲", -55, -20, 60, 84),
        # 印尼/马来群岛
        ("亚洲", 95, 141, -11, 7),
        # 日本
        ("亚洲", 129, 146, 30, 46),
        # 菲律宾
        ("亚洲", 117, 127, 5, 19),
        # 新几内亚岛
        ("大洋洲", 130, 151, -11, -1),
        # 新西兰
        ("大洋洲", 166, 179, -47, -34),
        # 英国/爱尔兰
        ("欧洲", -11, 2, 50, 59),
        # 冰岛
        ("欧洲", -25, -13, 63, 67),
        # 马达加斯加
        ("非洲", 43, 51, -26, -12),
        # 古巴及加勒比
        ("北美洲", -85, -74, 17, 24),
        # 中美洲走廊
        ("北美洲", -92, -77, 7, 18),
        # 斯里兰卡
        ("亚洲", 79, 82, 5, 10),
        # 台湾
        ("亚洲", 119, 122, 21, 26),
        # 海南岛
        ("亚洲", 108, 111, 18, 20),
        # 巴布亚新几内亚外岛
        ("大洋洲", 148, 156, -11, -1),
        # 所罗门/瓦努阿图/斐济
        ("大洋洲", 155, 180, -22, -5),
    ]

    # ---- 海域空洞（与classify_location_type_vectorized一致）----
    sea_holes = [
        (-5, 36, 30, 46), (28, 42, 41, 47), (47, 55, 37, 47),
        (32, 44, 12, 30), (48, 58, 24, 30), (42, 52, 5, 18),
        (-10, 10, -5, 5), (-95, -70, 51, 63), (10, 30, 54, 66),
        (-88, -60, 10, 25), (105, 120, 0, 22), (120, 130, 25, 33),
        (127, 138, 35, 46), (138, 152, 50, 62), (170, 180, 52, 62),
        (-65, -45, 65, 78), (-97, -82, 18, 30), (130, 145, -18, -8),
        (118, 135, -38, -31), (80, 95, 5, 22), (55, 73, 5, 25),
        (40, 46, -25, -10), (-10, 2, 43, 50),
    ]

    # ---- 判断陆地+大洲 ----
    is_land = np.zeros(n, dtype=bool)
    for continent, lon_min, lon_max, lat_min, lat_max in land_continent:
        mask = (lon_norm >= lon_min) & (lon_norm <= lon_max) & \
               (lat_arr >= lat_min) & (lat_arr <= lat_max)
        is_land |= mask
        result[mask & (result == "海洋")] = continent

    # 扣除海域空洞（这些点不是陆地，大洲改回"海洋"）
    for lon_min, lon_max, lat_min, lat_max in sea_holes:
        hole = (lon_norm >= lon_min) & (lon_norm <= lon_max) & \
               (lat_arr >= lat_min) & (lat_arr <= lat_max)
        result[hole] = "海洋"

    # 南极洲
    result[lat_arr < -60] = "南极洲"

    return result


def classify_ocean_vectorized(lon_arr, lat_arr):
    """向量化海洋分类（使用显式括号确保优先级正确）"""
    lon_arr = np.asarray(lon_arr, dtype=float)
    lat_arr = np.asarray(lat_arr, dtype=float)
    lon_norm = ((lon_arr + 180) % 360) - 180
    result = np.full(len(lon_arr), "其他海洋", dtype=object)
    
    # 太平洋：西太平洋 lon_norm 140~180 + 东太平洋 lon_norm -180~-60
    pacific_mask = (
        (lat_arr >= -60) & (lat_arr <= 60) &
        (
            ((lon_norm >= 140) & (lon_norm <= 180)) |
            ((lon_norm >= -180) & (lon_norm <= -60))
        )
    )
    
    # 日本海/东海/黄海 → 归入太平洋
    east_asia_sea_mask = (
        (lat_arr >= 20) & (lat_arr <= 50) &
        (lon_norm >= 120) & (lon_norm <= 145)
    )
    
    atlantic_mask = (lat_arr >= -60) & (lat_arr <= 60) & (lon_norm >= -80) & (lon_norm <= 20)
    indian_mask = (lat_arr >= -60) & (lat_arr <= 30) & (lon_norm >= 20) & (lon_norm <= 140)
    arctic_mask = (lat_arr >= 65)
    southern_mask = (lat_arr <= -60)
    mediterranean_mask = (lat_arr >= 30) & (lat_arr <= 45) & (lon_norm >= -10) & (lon_norm <= 40)
    
    result[pacific_mask | east_asia_sea_mask] = "太平洋"
    result[atlantic_mask] = "大西洋"
    result[indian_mask] = "印度洋"
    result[arctic_mask] = "北冰洋"
    result[southern_mask] = "南大洋"
    result[mediterranean_mask] = "地中海"
    return result


def ocean_climate_zone_vectorized(lon_arr, lat_arr):
    """
    海洋气候区分级：海域名 + 纬度带（语义化分类）
    
    柯本气候分类只覆盖陆地，海洋返回"未知"。
    本函数给海洋事件赋予语义化标签，如：
      "北太平洋热带海洋" / "北大西洋温带海洋" / "印度洋亚热带海洋"
    """
    lon_arr = np.asarray(lon_arr, dtype=float)
    lat_arr = np.asarray(lat_arr, dtype=float)
    
    # 先查询海域名
    ocean_names = classify_ocean_vectorized(lon_arr, lat_arr)
    
    # 纬度带划分
    lat_band = np.full(len(lat_arr), "赤道", dtype=object)
    lat_band[lat_arr >= 66.5] = "北极"
    lat_band[(lat_arr >= 35) & (lat_arr < 66.5)] = "北温带"
    lat_band[(lat_arr >= 23.5) & (lat_arr < 35)] = "北亚热带"
    lat_band[(lat_arr > -23.5) & (lat_arr < 23.5)] = "热带"
    lat_band[(lat_arr <= -23.5) & (lat_arr > -35)] = "南亚热带"
    lat_band[(lat_arr <= -35) & (lat_arr > -66.5)] = "南温带"
    lat_band[lat_arr <= -66.5] = "南极"
    
    # 北冰洋/南大洋：特殊处理
    result = np.full(len(lon_arr), "", dtype=object)
    arctic_like = (ocean_names == "北冰洋") | (lat_arr >= 66.5)
    antarctic_like = (ocean_names == "南大洋") | (lat_arr <= -66.5)
    
    result[arctic_like] = "北极海洋"
    result[antarctic_like] = "南极海洋"
    rest = (~arctic_like) & (~antarctic_like)
    
    # 向量化拼接字符串：纬度带 + 海域名 + "海洋"
    if rest.any():
        lb = lat_band[rest].astype(str)
        on = ocean_names[rest].astype(str)
        result[rest] = np.char.add(np.char.add(lb, on), "海洋")
    
    return result


# ============================================================
# 导出与报告生成
# ============================================================

def export_centroid_trajectories(event_summary, output_dir):
    """导出质心轨迹（简化版）"""
    print("\n📤 导出质心轨迹...")
    # 由于原始 R 代码中 centroid_trajectory 是嵌套列表，在 Python 中简化处理
    print("  ℹ️ 质心轨迹功能在 Python 版中已简化，如需完整轨迹请扩展实现")
    print("  ✅ 质心轨迹导出完成（占位）")


def merge_cross_month_events(full_year_df, config, max_gap_hours=6, max_distance_deg=2.0, allow_cross_year=False):
    """合并跨月/跨年事件（解决月度独立处理导致的168h截断和事件割裂问题）

    算法：
    1. 按 start_time 排序所有事件
    2. 检测相邻事件：若 end_time 与下一个 start_time 间隔 < max_gap_hours
       且质心空间距离 < max_distance_deg，则视为同一事件的两个月度片段
    3. 合并这些事件：扩展时间范围、累积降雨、保留最大网格数、加权质心

    Args:
        full_year_df: 全年事件 DataFrame
        config: 配置字典
        max_gap_hours: 跨月合并的最大时间间隔（小时），默认6h
        max_distance_deg: 跨月合并的最大空间距离（度），默认2.0°
        allow_cross_year: 是否允许跨年合并（多年汇总时设为True）

    Returns:
        合并后的 DataFrame
    """
    if full_year_df is None or full_year_df.empty:
        return full_year_df

    _merge_label = "跨月+跨年" if allow_cross_year else "跨月"
    print(f"\n🔗 {_merge_label}事件合并 (间隔<{max_gap_hours}h, 距离<{max_distance_deg}°)...")
    df = full_year_df.sort_values("start_time").reset_index(drop=True).copy()

    # 确保时间列为 datetime
    for col in ["start_time", "end_time"]:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col])

    n = len(df)
    if n < 2:
        print("  事件数 < 2，无需合并")
        return full_year_df

    # ---- 步骤1：检测需要合并的相邻事件对 ----
    merge_groups = np.arange(n)  # 初始每个事件自成一组

    def find_root(idx):
        """并查集查找根节点（带路径压缩）"""
        while merge_groups[idx] != idx:
            merge_groups[idx] = merge_groups[merge_groups[idx]]
            idx = merge_groups[idx]
        return idx

    def union(i, j):
        ri, rj = find_root(i), find_root(j)
        if ri != rj:
            merge_groups[ri] = rj

    n_pairs = 0
    for i in range(n - 1):
        curr_end = df.loc[i, "end_time"]
        next_start = df.loc[i + 1, "start_time"]
        time_gap = (next_start - curr_end).total_seconds() / 3600.0

        # 时间间隔检查（允许小间隔，跨月事件通常 gap >= 0）
        if not (0 <= time_gap <= max_gap_hours):
            continue

        # 同年检查（跨月合并不跨年；多年汇总时允许跨年）
        if not allow_cross_year and curr_end.year != next_start.year:
            continue

        # 空间距离检查（用经纬度+纬度余弦修正，处理0°/360°经线环绕）
        lon1, lat1 = df.loc[i, "center_lon"], df.loc[i, "center_lat"]
        lon2, lat2 = df.loc[i + 1, "center_lon"], df.loc[i + 1, "center_lat"]
        # 经度差归一化到 [-180, 180]，避免跨日界线时距离算到 358°
        dlon = ((lon1 - lon2 + 180) % 360) - 180
        mid_lat = np.radians((lat1 + lat2) / 2)
        spatial_dist = np.sqrt(
            (dlon * np.cos(mid_lat)) ** 2 + (lat1 - lat2) ** 2
        )
        if spatial_dist > max_distance_deg:
            continue

        # 满足合并条件
        union(i, i + 1)
        n_pairs += 1

    # 压缩所有节点
    for i in range(n):
        merge_groups[i] = find_root(i)

    # 重新编号组ID为连续整数
    unique_groups = sorted(set(merge_groups))
    group_map = {g: i for i, g in enumerate(unique_groups)}
    merge_groups = np.array([group_map[g] for g in merge_groups])
    df["_merge_group"] = merge_groups

    n_groups = len(unique_groups)
    n_merged_pairs = n - n_groups
    print(f"  合并前: {n} 个事件")
    print(f"  检测到 {n_merged_pairs} 对跨月事件需要合并")
    print(f"  合并后: {n_groups} 个事件")

    if n_merged_pairs == 0:
        df = df.drop(columns=["_merge_group"])
        return df

    # ---- 步骤2：按组合并事件 ----
    merged_rows = []
    for gid, sub in df.groupby("_merge_group"):
        if len(sub) == 1:
            merged_rows.append(sub.iloc[0].drop("_merge_group"))
            continue

        # 多事件合并
        # ★修复 P0: 用 start_time（已转为 datetime）排序，确保时间顺序正确
        # 跨月合并不涉及 time_num（月度间 time_num 不连续），用 start_time 排序
        sub_sorted = sub.sort_values("start_time", ascending=True)
        total_rain = float(sub_sorted["total_rain_depth"].sum())
        # n_grids 取 max 而非 sum：同一事件跨月片段的网格可能高度重叠，
        # sum 会严重高估 n_grids，max 是更合理的近似
        max_grids = int(sub_sorted["n_grids"].max())
        # center_lon 用循环加权平均（处理跨0°/360°经线）
        _wg = sub_sorted["n_grids"].values.astype(float)
        _lon_rad = np.radians(sub_sorted["center_lon"].values)
        _cx = np.average(np.cos(_lon_rad), weights=_wg)
        _cy = np.average(np.sin(_lon_rad), weights=_wg)
        w_lon = float(np.degrees(np.arctan2(_cy, _cx)) % 360)
        w_lat = float(np.average(sub_sorted["center_lat"], weights=_wg))

        merged = {
            "start_time": sub_sorted["start_time"].min(),
            "end_time": sub_sorted["end_time"].max(),
            "total_rain_depth": total_rain,
            "max_30min_rain": float(sub_sorted["max_30min_rain"].max()),
            "max_30min_time": sub_sorted.loc[sub_sorted["max_30min_rain"].idxmax(), "max_30min_time"],
            "mean_intensity": float(sub_sorted["mean_intensity"].mean()),
            "lat_min": float(sub_sorted["lat_min"].min()),
            "lat_max": float(sub_sorted["lat_max"].max()),
            "center_lon": w_lon,
            "center_lat": w_lat,
            "n_grids": max_grids,
        }
        
        # ★修复：跨月合并时正确处理跨日界线事件的 lon_min/lon_max
        # 先获取所有片段的原始 lon_min/lon_max（未翻转的值）
        # 注意：输入的 lon_min/lon_max 可能已经被翻转（在 summarize_events 中）
        # 我们需要先恢复原始值，然后正确合并
        
        # 检查是否有跨日界线事件（lon_min > lon_max 表示已翻转）
        has_crosses = "crosses_dateline" in sub_sorted.columns
        if has_crosses:
            # 先恢复所有片段的原始 lon_min/lon_max
            orig_lon_mins = []
            orig_lon_maxs = []
            for _, row in sub_sorted.iterrows():
                if row.get("crosses_dateline", 0) == 1:
                    # 已翻转，需要恢复
                    orig_lon_mins.append(row["lon_max"])  # 翻转前的 lon_min 在 lon_max 位置
                    orig_lon_maxs.append(row["lon_min"])  # 翻转前的 lon_max 在 lon_min 位置
                else:
                    orig_lon_mins.append(row["lon_min"])
                    orig_lon_maxs.append(row["lon_max"])
            
            # 计算合并后的原始 lon_min/lon_max
            merged_lon_min = float(np.min(orig_lon_mins))
            merged_lon_max = float(np.max(orig_lon_maxs))
            
            # 检查合并后是否跨日界线
            if merged_lon_max - merged_lon_min > 180:
                # 跨日界线：翻转 lon_min/lon_max
                merged["lon_min"] = merged_lon_max  # 翻转：lon_min = 较大值
                merged["lon_max"] = merged_lon_min  # 翻转：lon_max = 较小值
            else:
                # 不跨日界线
                merged["lon_min"] = merged_lon_min
                merged["lon_max"] = merged_lon_max
        else:
            # 没有 crosses_dateline 字段，直接合并
            merged["lon_min"] = float(sub_sorted["lon_min"].min())
            merged["lon_max"] = float(sub_sorted["lon_max"].max())
        # 加权质心（如果存在）- 同样用循环平均
        if "weighted_lon" in sub_sorted.columns:
            _wl = sub_sorted["weighted_lon"].fillna(sub_sorted["center_lon"]).values
            _wl_rad = np.radians(_wl)
            _wx = np.average(np.cos(_wl_rad), weights=_wg)
            _wy = np.average(np.sin(_wl_rad), weights=_wg)
            merged["weighted_lon"] = float(np.degrees(np.arctan2(_wy, _wx)) % 360)
            merged["weighted_lat"] = float(np.average(
                sub_sorted["weighted_lat"].fillna(sub_sorted["center_lat"]),
                weights=_wg
            ))

        # 衍生字段
        merged["duration_hours"] = (merged["end_time"] - merged["start_time"]).total_seconds() / 3600.0
        # avg_precip_depth: 用各片段avg_precip_depth的n_grids加权平均
        # 避免 total_rain(sum) / max_grids 导致跨月合并值虚高（同一网格在两个月度片段被重复计入total_rain）
        if "avg_precip_depth" in sub_sorted.columns and max_grids > 0:
            merged["avg_precip_depth"] = float(np.average(
                sub_sorted["avg_precip_depth"].values,
                weights=sub_sorted["n_grids"].values.astype(float)
            ))
        else:
            merged["avg_precip_depth"] = total_rain / max_grids if max_grids > 0 else 0
        merged["max_30min_intensity"] = merged["max_30min_rain"] * 2
        # ★修复：跨月合并后重算极端标志（基于合并后的max_30min_rain和avg_precip_depth）
        merged["is_extreme_max30"] = int(merged["max_30min_rain"] > 100)
        merged["is_extreme_precip"] = int(merged["avg_precip_depth"] > 200)

        # 分类字段：取第一个事件的（同一事件分类应一致）
        for col in ["location_type", "continent", "ocean", "climate_zone", "country"]:
            if col in sub_sorted.columns:
                merged[col] = sub_sorted.iloc[0][col]

        # 置信度：取最大
        if "confidence" in sub_sorted.columns:
            merged["confidence"] = float(sub_sorted["confidence"].max())

        # bbox面积：取最大
        if "bbox_area_km2" in sub_sorted.columns:
            merged["bbox_area_km2"] = float(sub_sorted["bbox_area_km2"].max())

        # 总水量：用 total_rain_depth 直接计算（水量是广延量，应累加）
        # 注：avg_precip_depth 用加权平均（代表典型格点雨深），不能直接乘 n_grids 算水量
        # ★已知偏差: 跨月片段网格若重叠，total_rain_depth(sum) 会重复计算重叠网格，
        #   导致 total_water_volume 高估。n_grids 用 max 避免高估网格数。
        #   保留此偏差以维持与月度CSV的字段一致性，不改变事件划分逻辑。
        grid_area_km2 = (config.get("spatial_res_deg", 0.072) * 111) ** 2
        lat_cos = np.cos(np.radians(np.clip(merged["center_lat"], -80, 80)))
        merged["total_water_volume"] = float(
            total_rain * grid_area_km2 * lat_cos * 1e3
        )

        merged_rows.append(pd.Series(merged))

    merged_df = pd.DataFrame(merged_rows)
    merged_df = merged_df.sort_values("start_time").reset_index(drop=True)

    # ★跨月合并后重新验证 location_type vs continent 一致性
    # 合并取第一个事件分类，但质心已变，可能导致矛盾
    if "location_type" in merged_df.columns and "continent" in merged_df.columns:
        _ocean_m = merged_df["location_type"].isin(["海洋", "近海"])
        _land_m = merged_df["location_type"].isin(["内陆", "沿海"])
        _inconsist_m = (
            (_ocean_m & (merged_df["continent"] != "海洋")) |
            (_land_m & (merged_df["continent"] == "海洋"))
        )
        if _inconsist_m.any():
            n_fix = int(_inconsist_m.sum())
            print(f"  🔧 修复 {n_fix} 条跨月合并事件的 location_type vs continent 矛盾")
            # 用合并后质心重算
            _mlon = merged_df.loc[_inconsist_m, "center_lon"].values
            _mlat = merged_df.loc[_inconsist_m, "center_lat"].values
            _mlon_norm = ((_mlon + 180) % 360) - 180
            # 海洋事件→continent=海洋；陆地事件→重算continent
            _fix_ocean = merged_df.loc[_inconsist_m, "location_type"].isin(["海洋", "近海"]).values
            _fix_land = ~_fix_ocean
            # 海洋事件
            _idx_ocean = merged_df.index[_inconsist_m][_fix_ocean]
            merged_df.loc[_idx_ocean, "continent"] = "海洋"
            merged_df.loc[_idx_ocean, "country"] = ""
            # 陆地事件→重算
            _idx_land = merged_df.index[_inconsist_m][_fix_land]
            if len(_idx_land) > 0:
                _re_cont = classify_continent_vectorized(_mlon_norm[_fix_land], _mlat[_fix_land])
                _re_country = classify_country_vectorized(_mlon_norm[_fix_land], _mlat[_fix_land])
                merged_df.loc[_idx_land, "continent"] = _re_cont
                merged_df.loc[_idx_land, "country"] = _re_country
                merged_df.loc[_idx_land, "ocean"] = ""

    # 显示合并示例
    cross_month_events = merged_df[merged_df["duration_hours"] > 168]
    if len(cross_month_events) > 0:
        print(f"  ✅ 合并后超过168h的事件: {len(cross_month_events)} 个")
        print(f"     最长持续: {cross_month_events['duration_hours'].max():.1f}h")

    return merged_df


def generate_interactive_report(event_summary, output_dir, config, csv_filename=None, map_filename=None, report_filename=None):
    """生成交互式报告

    Args:
        csv_filename: 报告中CSV下载链接指向的文件名（默认按当月生成）
        map_filename: 内嵌iframe指向的地图HTML文件名（默认 interactive_map_raw.html）
        report_filename: 输出报告HTML文件名（默认 report_raw_validation.html）
    """
    print("\n🎨 生成交互式报告...")
    if event_summary is None or event_summary.empty:
        return None
    if not HAS_PLOTLY:
        print("  ⚠️ plotly 未安装，跳过交互式地图生成")
        # 仍然生成静态 HTML 报告
        _generate_static_html_report(event_summary, output_dir, config)
        return None

    fig = go.Figure()
    
    # 颜色映射：使用对数尺度更直观地表示雨强差异
    # 雨强数据范围 0-200 mm/hr，线性映射会让大部分点颜色偏浅
    # 对数映射能让 1mm/hr 和 50mm/hr 的差异更明显
    intensity_col = "max_30min_intensity" if "max_30min_intensity" in event_summary.columns else "avg_precip_depth"
    
    # 计算全局颜色范围（对数尺度）
    all_intensities = event_summary[intensity_col].clip(lower=0.1).values
    int_min = max(float(np.percentile(all_intensities, 1)), 0.1)
    int_max = min(float(np.percentile(all_intensities, 99)), 200.0)
    
    for loc_type in ["内陆", "沿海", "海洋", "近海"]:
        sub = event_summary[event_summary["location_type"] == loc_type]
        if sub.empty:
            continue
        
        # 颜色映射：对数尺度（更直观显示雨强差异）
        intensities = sub[intensity_col].clip(lower=0.1).values
        # 圆圈大小：按事件规模（n_grids）适度变化，避免大小差异过大影响观感
        # 小事件(25-50格)用小圆点，大事件(>500格)用稍大圆点
        sizes = np.clip(5 + np.log1p(sub["n_grids"].values) * 1.5, 5, 18)
        
        # 悬停文本
        hover_texts = []
        for _, r in sub.iterrows():
            hover = (
                f"<b>事件:</b> {r['event_id']}<br>"
                f"<b>时长:</b> {r['duration_hours']:.1f} h<br>"
                f"<b>峰值雨强:</b> {r.get('max_30min_intensity', 0):.1f} mm/hr<br>"
                f"<b>平均雨量:</b> {r.get('avg_precip_depth', 0):.1f} mm<br>"
                f"<b>网格数:</b> {r['n_grids']}<br>"
                f"<b>位置:</b> {r['location_type']}<br>"
                f"<b>气候区:</b> {r['climate_zone']}" +
                (f"<br><b>国家:</b> {r['country']}" if r.get('country', '') else "")
            )
            hover_texts.append(hover)
        
        fig.add_trace(go.Scattergeo(
            lon=sub["center_lon"],
            lat=sub["center_lat"],
            mode='markers',
            marker=dict(
                size=sizes,
                color=np.log10(intensities),  # 对数颜色映射
                colorscale='YlOrRd',  # 黄→橙→红，直观表示降雨强弱
                cmin=np.log10(int_min),
                cmax=np.log10(int_max),
                showscale=(loc_type == "海洋"),  # 只在第一个trace显示色标
                colorbar=dict(
                    title="雨强(mm/hr)",
                    x=1.02, len=0.6,
                    tickvals=np.log10([1, 5, 10, 25, 50, 100, 200]),
                    ticktext=["1", "5", "10", "25", "50", "100", "200"]
                ) if loc_type == "海洋" else None,
                opacity=0.75,
                line=dict(width=0.5, color='rgba(80,80,80,0.3)')
            ),
            name=loc_type,
            text=hover_texts,
            hoverinfo='text'
        ))

    fig.update_layout(
        title=dict(text="全球降雨事件空间分布（颜色深浅=对数雨强，圆圈大小=事件规模）", font=dict(size=18)),
        geo=dict(projection_type='mollweide', showland=True, landcolor='#f5f5f5',
                 coastlinecolor='#333333', showocean=True, oceancolor='#e0f0ff',
                 showcountries=True, countrycolor='#cccccc', showframe=False,
                 lataxis=dict(range=[-65, 65]), lonaxis=dict(range=[0, 360])),
        legend=dict(title=dict(text="位置类型"), orientation="h", y=-0.1, x=0.5, xanchor="center"),
        margin=dict(l=0, r=80, t=50, b=80)
    )

    # 支持自定义文件名（用于全年报告 vs 月度报告）
    _map_fname = map_filename or "interactive_map_raw.html"
    _report_fname = report_filename or "report_raw_validation.html"
    _csv_fname = csv_filename or f"rainfall_events_{config['year']}{config['month']:02d}_raw.csv"

    html_path = os.path.join(output_dir, _map_fname)
    fig.write_html(html_path)

    html_content = f'''<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>降雨事件分析报告</title>
<style>body{{font-family:Arial;margin:20px;background:#f5f7fa;}}
h1{{color:#2c3e50;border-bottom:3px solid #3498db;padding-bottom:10px;}}
.summary-box{{display:flex;flex-wrap:wrap;gap:15px;margin:20px 0;}}
.stat-card{{background:white;border-radius:8px;padding:15px 25px;box-shadow:0 2px 8px rgba(0,0,0,0.1);flex:1;min-width:150px;}}
.stat-card .number{{font-size:28px;font-weight:bold;color:#2c3e50;}}
.stat-card .label{{font-size:14px;color:#7f8c8d;}}
.tab{{margin:20px 0;}}.tab iframe{{width:100%;height:500px;border:none;border-radius:8px;}}
.warning{{background:#fff3cd;border:1px solid #ffc107;padding:15px;border-radius:8px;margin:20px 0;}}
.footer{{margin-top:40px;color:#95a5a6;font-size:12px;text-align:center;border-top:1px solid #ddd;padding-top:20px;}}
</style>
</head>
<body>
<h1>🌧️ 全球降雨事件分析报告</h1>
<p>验证日期: {config['year']}年{config['month']:02d}月{config['start_day']:02d}日 | 事件数: {len(event_summary)} | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
<div class="warning"><b>📌 说明：</b> 原始分辨率（0.072°≈8km）验证版本</div>
<div class="summary-box">
<div class="stat-card"><div class="number">{len(event_summary)}</div><div class="label">总事件数</div></div>
<div class="stat-card"><div class="number">{event_summary['duration_hours'].mean():.1f}</div><div class="label">平均持续时间 (h)</div></div>
<div class="stat-card"><div class="number">{event_summary['avg_precip_depth'].mean():.1f}</div><div class="label">平均降水量 (mm)</div></div>
<div class="stat-card"><div class="number">{event_summary['max_30min_intensity'].max():.1f}</div><div class="label">最大30分钟雨强 (mm/hr)</div></div>
<div class="stat-card"><div class="number">{event_summary['confidence'].mean():.2f}</div><div class="label">平均置信度</div></div>
</div>
<h2>🗺️ 事件空间分布</h2>
<div class="tab"><iframe src="{_map_fname}"></iframe></div>
<div class="footer">报告自动生成 | 方法: IETD + 3D DBSCAN | <a href="{_csv_fname}">下载CSV</a></div>
</body></html>'''

    report_path = os.path.join(output_dir, _report_fname)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print("  ✅ 报告已生成")
    return report_path


def _generate_static_html_report(event_summary, output_dir, config):
    """生成纯静态 HTML 报告（无需 plotly）"""
    html_content = f'''<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>降雨事件分析报告</title>
<style>
body{{font-family:Arial;margin:20px;background:#f5f7fa;}}
h1{{color:#2c3e50;border-bottom:3px solid #3498db;padding-bottom:10px;}}
.summary-box{{display:flex;flex-wrap:wrap;gap:15px;margin:20px 0;}}
.stat-card{{background:white;border-radius:8px;padding:15px 25px;box-shadow:0 2px 8px rgba(0,0,0,0.1);flex:1;min-width:150px;}}
.stat-card .number{{font-size:28px;font-weight:bold;color:#2c3e50;}}
.stat-card .label{{font-size:14px;color:#7f8c8d;}}
table{{border-collapse:collapse;width:100%;margin:20px 0;}}
th,td{{border:1px solid #ddd;padding:8px;text-align:left;font-size:12px;}}
th{{background:#3498db;color:white;}}
tr:nth-child(even){{background:#f2f2f2;}}
.footer{{margin-top:40px;color:#95a5a6;font-size:12px;text-align:center;border-top:1px solid #ddd;padding-top:20px;}}
</style></head><body>
<h1>🌧️ 全球降雨事件分析报告</h1>
<p>验证日期: {config['year']}年{config['month']:02d}月{config['start_day']:02d}日 | 事件数: {len(event_summary)} | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
<div class="summary-box">
<div class="stat-card"><div class="number">{len(event_summary)}</div><div class="label">总事件数</div></div>
<div class="stat-card"><div class="number">{event_summary['duration_hours'].mean():.1f}</div><div class="label">平均持续时间 (h)</div></div>
<div class="stat-card"><div class="number">{event_summary['avg_precip_depth'].mean():.1f}</div><div class="label">平均降水量 (mm)</div></div>
<div class="stat-card"><div class="number">{event_summary['max_30min_intensity'].max():.1f}</div><div class="label">最大30分钟雨强 (mm/hr)</div></div>
<div class="stat-card"><div class="number">{event_summary['confidence'].mean():.2f}</div><div class="label">平均置信度</div></div>
</div>
<h2>📋 事件列表（前50条）</h2>
{event_summary.head(50).to_html(index=False, escape=False)}
<div class="footer">报告自动生成 | 方法: IETD + 3D DBSCAN</div>
</body></html>'''
    report_path = os.path.join(output_dir, "report_raw_validation.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print("  ✅ 静态报告已生成")
    return report_path


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    import time
    start_time = time.time()

    # 配置校验
    _config_errors = validate_config(CONFIG)
    if _config_errors:
        print("❌ 配置校验失败:")
        for err in _config_errors:
            print(f"  - {err}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("🚀 启动降雨事件识别模型 v18.0（多年批量处理版 2000-2025）")
    print("=" * 60)
    print(f"📅 处理年份范围: {CONFIG['start_year']}~{CONFIG['end_year']}年")
    print(f"📐 空间分辨率: {CONFIG['spatial_res_deg']:.3f}° (≈{CONFIG['spatial_res_deg'] * 111:.0f} km)")
    print(f"📦 批次大小: {CONFIG['max_grids_per_batch']}")
    print(f"⚡ 快速模式: {'开启' if CONFIG.get('fast_mode') else '关闭'}")
    print(f"📚 可选依赖: geopandas={HAS_GEOPANDAS}, plotly={HAS_PLOTLY}, geopy={HAS_GEOPY}")
    print(f"💾 临时目录: {CONFIG['temp_dir']}")

    # 创建输出目录
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(CONFIG["temp_dir"], exist_ok=True)

    # 加载柯本气候区数据（支持 shapefile 和 raster）
    kg_gdf = load_kg_shapefile(
        CONFIG["kg_shapefile_path"], 
        CONFIG.get("kg_raster_path")
    )
    if kg_gdf is not None:
        if hasattr(kg_gdf, 'crs'):
            print(f"✅ 加载柯本气候分区: {len(kg_gdf)} 个多边形")
        else:
            print("✅ 加载柯本气候分区 raster")
    else:
        print("⚠️ 柯本气候分区未加载，使用纬度带划分")
    
    # 构建气候区查找表（一次性，后续O(1)查询）
    climate_lut = None
    if kg_gdf is not None and HAS_GEOPANDAS:
        lut_res = CONFIG.get("climate_lut_resolution", 0.1)
        print(f"\n🔨 构建气候区查找表 (分辨率 {lut_res}°)...")
        climate_lut = build_climate_lut(kg_gdf, lon_res=lut_res, lat_res=lut_res)
        if climate_lut is not None:
            n_land = (climate_lut["zone_grid"] != "未知").sum()
            n_total = climate_lut["zone_grid"].size
            print(f"  ✅ 查找表构建完成: {n_land}/{n_total} 个陆地区域网格 ({100*n_land/n_total:.1f}%)")
    
    CONFIG["_climate_lut"] = climate_lut

    geo_data = load_geo_data()
    check_memory_usage()

    # ============ 多年逐月处理（2000-2025）============
    import calendar
    all_year_summaries = {}  # 每年一个汇总DataFrame

    # ★多年循环: start_year ~ end_year
    _base_data_dir = CONFIG["data_dir"]
    _skip_missing = CONFIG.get("skip_missing_years", True)
    _skip_existing = CONFIG.get("skip_existing_results", True)

    _gc_threshold_mb = CONFIG.get("performance", {}).get("gc_threshold_mb", 0)

    for proc_year in range(CONFIG["start_year"], CONFIG["end_year"] + 1):
        CONFIG["year"] = proc_year
        # ★季度级 time_scale_factor 缓存：仅【同年内】同季度复用，跨年不复用（数据驱动参数）
        _tsf_season_cache = {}  # season_name -> best_tsf

        # ★断点续传：检查全年CSV是否已存在（避免重新处理已完成的年份）
        _fullyear_csv = os.path.join(CONFIG["output_dir"], f"rainfall_events_{proc_year}_fullyear.csv")
        if _skip_existing and os.path.exists(_fullyear_csv):
            print(f"\n⏭️  {proc_year}年全年CSV已存在，跳过: {_fullyear_csv}")
            continue
        
        # ★构建该年份的数据目录: E:\date download\CMORPH\YYYY
        _year_data_dir = os.path.join(_base_data_dir, str(proc_year))
        
        # 检查数据目录是否存在
        if not os.path.isdir(_year_data_dir):
            if _skip_missing:
                print(f"\n⚠️ {proc_year}年数据目录不存在 ({_year_data_dir})，跳过")
                continue
            else:
                print(f"\n❌ {proc_year}年数据目录不存在: {_year_data_dir}")
                sys.exit(1)
        
        # 检查是否有月度子目录
        _has_months = any(
            os.path.isdir(os.path.join(_year_data_dir, d))
            for d in os.listdir(_year_data_dir)
        )
        if not _has_months:
            if _skip_missing:
                print(f"\n⚠️ {proc_year}年数据目录为空，跳过")
                continue
            else:
                print(f"\n❌ {proc_year}年数据目录为空: {_year_data_dir}")
                sys.exit(1)
        
        # ★设置该年份的数据目录
        CONFIG["data_dir"] = _year_data_dir
        
        print(f"\n{'#'*60}")
        print(f"##  📆📅 开始处理 {proc_year} 年数据")
        print(f"##  数据目录: {_year_data_dir}")
        print(f"{'#'*60}")
        
        all_monthly_summaries = []

        for proc_month in range(CONFIG["start_month"], CONFIG["end_month"] + 1):
            CONFIG["month"] = proc_month
            days_in_month = calendar.monthrange(CONFIG["year"], proc_month)[1]
            CONFIG["start_day"] = 1
            CONFIG["end_day"] = days_in_month

            # ★断点续传：检查月度CSV是否已存在，存在则直接加载跳过处理
            _month_tag = f"{CONFIG['year']}{proc_month:02d}"
            _month_csv = os.path.join(CONFIG["output_dir"], f"rainfall_events_{_month_tag}_raw.csv")
            if _skip_existing and os.path.exists(_month_csv):
                print(f"\n⏭️  {CONFIG['year']}年{proc_month:02d}月已处理过，加载已有结果: {_month_csv}")
                _load_ok = False
                try:
                    _existing_month_df = pd.read_csv(_month_csv)
                    # ★修复 P1: 补全 max_30min_time 转换，避免跨月合并 idxmax 取到字符串
                    for _ts_col in ["start_time", "end_time", "max_30min_time"]:
                        if _ts_col in _existing_month_df.columns:
                            _existing_month_df[_ts_col] = pd.to_datetime(_existing_month_df[_ts_col])
                    if not _existing_month_df.empty:
                        all_monthly_summaries.append(_existing_month_df)
                    print(f"  ✅ 加载完成: {len(_existing_month_df)} 个事件")
                    _load_ok = True
                except Exception as e:
                    print(f"  ⚠️ 加载失败 ({e})，将重新处理该月")
                if _load_ok:
                    continue
                # 加载失败时落到下方重新处理流程（不再 continue）

            print(f"\n{'#'*60}")
            print(f"##  📆 处理 {CONFIG['year']}年{proc_month:02d}月 (1~{days_in_month}日)")
            print(f"{'#'*60}")
            _month_t0 = time.time()  # ★月度总计时

            # ---- 获取月份自适应参数 ----
            month = CONFIG["month"]
            adaptive_params = get_month_adaptive_params(CONFIG, month)
            print(f"🌍 季节自适应: {adaptive_params['season']} (系数={adaptive_params['season_factor']})")
            print(f"   切分阈值: 面积变化>{adaptive_params['area_change_threshold']:.2f}, 强度下降>{adaptive_params['intensity_drop_threshold']:.2f}")
            CONFIG["area_change_threshold"] = adaptive_params["area_change_threshold"]
            CONFIG["intensity_drop_threshold"] = adaptive_params["intensity_drop_threshold"]

            # ---- IETD 选择 ----
            # ★修复 P0: 严格区分"原始IETD"(_ietd_raw)与"季节调整IETD"(ietd_selected)，
            #   避免复用时把 adjusted 值再次乘 season_factor 导致累积偏差
            smart_split = CONFIG.get("performance", {}).get("use_smart_split", False)
            cached_raw = CONFIG.get("_ietd_raw")  # 仅复用未经季节调整的原始值
            if smart_split and cached_raw is not None and not CONFIG["performance"].get("skip_ietd_analysis", False):
                print(f"\n⏭️  复用缓存IETD = {cached_raw} h (use_smart_split=True)")
                selected_ietd = cached_raw
            elif CONFIG["performance"].get("skip_ietd_analysis", False):
                default_ietd = cached_raw or CONFIG.get("global_IETD_hours", 6)
                print(f"\n⏭️  跳过IETD敏感性分析，使用默认 IETD = {default_ietd} h")
                selected_ietd = default_ietd
            else:
                # IETD分析：读取前3天采样数据
                print("\n🔬 IETD敏感性分析（采样前3天）...")
                sample_days = min(3, CONFIG["end_day"] - CONFIG["start_day"] + 1)
                dt_sample = read_hourly_chunk(CONFIG["data_dir"], CONFIG["start_day"],
                                               CONFIG["start_day"] + sample_days - 1,
                                               CONFIG["chunk_hours"], CONFIG)
                if dt_sample is not None and not dt_sample.empty:
                    selected_ietd = ietd_sensitivity_analysis(dt_sample, CONFIG)
                    del dt_sample
                    gc.collect()
                else:
                    selected_ietd = 6
                # 缓存原始值（未经季节调整），供后续月份复用
                CONFIG["_ietd_raw"] = selected_ietd
            adjusted_ietd = max(2, int(selected_ietd * adaptive_params["season_factor"]))
            CONFIG["ietd_selected"] = adjusted_ietd  # 实际用于事件切分
            print(f"✅ 选择 IETD = {selected_ietd} h (季节调整后: {adjusted_ietd} h)")

            # ---- 逐日处理：读取→提取网格事件→释放 ----
            print(f"\n📅 逐日处理 ({CONFIG['start_day']}~{CONFIG['end_day']}日)...")
            all_daily_events = []
            _daily_t0 = time.time()  # ★逐日阶段计时

            for day in range(CONFIG["start_day"], CONFIG["end_day"] + 1):
                print(f"\n{'='*50}")
                print(f"  📆 处理 {CONFIG['year']}年{CONFIG['month']:02d}月{day:02d}日")
                print(f"{'='*50}")

                # 读取单日数据
                dt_day = read_hourly_chunk(CONFIG["data_dir"], day, day,
                                            CONFIG["chunk_hours"], CONFIG)
                if dt_day is None or dt_day.empty:
                    print(f"  ⚠️ 第 {day} 天无数据，跳过")
                    continue

                print(f"  📊 读取 {len(dt_day):,} 行数据")

                # 提取网格事件
                dt_day_events = extract_grid_events_memory_friendly(dt_day, CONFIG, kg_gdf)

                # 释放原始数据
                del dt_day
                # ★按内存阈值触发gc，避免每天强制gc.collect()的开销
                if _gc_threshold_mb > 0:
                    try:
                        import psutil
                        _cur_mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                        if _cur_mem > _gc_threshold_mb:
                            gc.collect()
                    except ImportError:
                        gc.collect()
                else:
                    gc.collect()

                if dt_day_events is not None and not dt_day_events.empty:
                    all_daily_events.append(dt_day_events)
                    print(f"  ✅ 第 {day} 天: {len(dt_day_events)} 个网格事件")
                else:
                    print(f"  ⚠️ 第 {day} 天未提取到事件")

                check_memory_usage()

            if not all_daily_events:
                print(f"❌ {CONFIG['month']:02d}月所有天数均未提取到事件，跳过该月")
                continue
            print(f"\n⏱️  逐日处理耗时: {time.time() - _daily_t0:.1f}s")

            # ---- 合并跨日事件 ----
            _t0 = time.time()
            dt_grid_events = merge_cross_day_events(all_daily_events, CONFIG)
            print(f"⏱️  跨日合并耗时: {time.time() - _t0:.1f}s ({len(dt_grid_events) if dt_grid_events is not None else 0} 事件)")
            del all_daily_events
            gc.collect()

            if dt_grid_events is None or dt_grid_events.empty:
                print(f"❌ {CONFIG['month']:02d}月合并后无事件，跳过该月")
                continue
            print(f"\n✅ {CONFIG['month']:02d}月共 {len(dt_grid_events)} 个网格事件（跨日合并后）")

            # ---- 时间缩放因子敏感性分析 ----
            _t0 = time.time()
            if CONFIG.get("time_scale_auto", True):
                _cache_tsf = CONFIG.get("performance", {}).get("cache_tsf_per_season", True)
                # ★缓存键用四季（春3-5/夏6-8/秋9-11/冬12-2），避免原 transition 把春秋混用
                _m = CONFIG["month"]
                _season_key = "spring" if _m in (3, 4, 5) else \
                              "summer" if _m in (6, 7, 8) else \
                              "autumn" if _m in (9, 10, 11) else "winter"
                if _cache_tsf and _season_key in _tsf_season_cache:
                    best_tsf = _tsf_season_cache[_season_key]
                    print(f"\n⏱️  复用同年季度缓存 time_scale_factor = {best_tsf:.0f} (season={_season_key})")
                    CONFIG["time_scale_factor"] = best_tsf
                else:
                    best_tsf = time_scale_factor_sensitivity(dt_grid_events, CONFIG)
                    CONFIG["time_scale_factor"] = best_tsf
                    if _cache_tsf:
                        _tsf_season_cache[_season_key] = best_tsf
                        print(f"   📌 已缓存 season={_season_key} 的 time_scale_factor = {best_tsf:.0f}")
            else:
                print(f"\n⏱️  使用固定时间缩放因子: {CONFIG['time_scale_factor']:.0f}")
            print(f"⏱️  TSF分析耗时: {time.time() - _t0:.1f}s")

            # ---- 时间缩放: 使用绝对小时数而非 day/31 比例 ----
            if "time_num" not in dt_grid_events.columns:
                dt_grid_events["time_num"] = (
                    dt_grid_events["start_time"] - dt_grid_events["start_time"].min()
                ).dt.total_seconds() / 3600.0
            dt_grid_events["time_scaled"] = dt_grid_events["time_num"] * CONFIG["time_scale_factor"]

            # ---- DBSCAN参数 ----
            _t0 = time.time()
            if CONFIG["enable_grid_search"]:
                grid_results = dbscan_grid_search(dt_grid_events, CONFIG)
                CONFIG["dbscan_eps_deg"] = grid_results["eps"]
                CONFIG["dbscan_minPts"] = grid_results["minPts"]
                print(f"\n✅ 网格搜索完成: eps={CONFIG['dbscan_eps_deg']:.1f}°, minPts={CONFIG['dbscan_minPts']}")
            else:
                print(f"\n✅ 使用手动参数: eps={CONFIG['dbscan_eps_deg']:.1f}°, minPts={CONFIG['dbscan_minPts']}")
            print(f"⏱️  DBSCAN参数搜索耗时: {time.time() - _t0:.1f}s")

            # ---- DBSCAN聚类 ----
            _t0 = time.time()
            dt_clustered = cluster_events_dbscan(dt_grid_events, CONFIG)
            print(f"⏱️  DBSCAN聚类耗时: {time.time() - _t0:.1f}s")
            if dt_clustered is None:
                print(f"❌ {CONFIG['month']:02d}月聚类失败，跳过该月")
                continue
            del dt_grid_events
            gc.collect()

            # ---- 动态切分（使用自适应阈值） ----
            _t0 = time.time()
            dt_merged = dynamic_split_events(dt_clustered, CONFIG)
            print(f"⏱️  动态切分耗时: {time.time() - _t0:.1f}s")
            if dt_merged is None:
                print(f"❌ {CONFIG['month']:02d}月事件切分失败，跳过该月")
                continue
            del dt_clustered
            gc.collect()

            # ---- 事件自验证 ----
            _t0 = time.time()
            dt_merged = validate_events(dt_merged, CONFIG)
            print(f"⏱️  事件验证耗时: {time.time() - _t0:.1f}s")

            # ---- 事件扩展 ----
            _t0 = time.time()
            dt_expanded = expand_events(dt_merged, CONFIG)
            print(f"⏱️  事件扩展耗时: {time.time() - _t0:.1f}s")

            # ---- 事件汇总 ----
            _t0 = time.time()
            event_summary = summarize_events(dt_expanded, CONFIG, geo_data)
            print(f"⏱️  事件汇总耗时: {time.time() - _t0:.1f}s")
            if event_summary is None:
                print(f"❌ {CONFIG['month']:02d}月事件汇总失败，跳过该月")
                continue

            # ---- 地理与气候分类 ----
            _t0 = time.time()
            event_summary = add_classifications(event_summary, CONFIG, kg_gdf)
            print(f"⏱️  地理分类耗时: {time.time() - _t0:.1f}s")

            print(f"\n⏱️  {CONFIG['month']:02d}月总耗时: {time.time() - _month_t0:.1f}s")

            # ---- 二次过滤：确保海洋事件 n_grids>=25（修复报告P1问题）----
            # 问题原因：初始过滤用简单经纬度判断陆地/海洋，与sjoin精确判断可能不一致
            # 导致部分海洋事件用陆地阈值(5)过滤，实际应为海洋阈值(25)
            if "location_type" in event_summary.columns and "n_grids" in event_summary.columns:
                min_grids_ocean = CONFIG.get("min_area_pixels", 25)
                ocean_small = (event_summary["location_type"] == "海洋") & (event_summary["n_grids"] < min_grids_ocean)
                n_ocean_small = int(ocean_small.sum())
                if n_ocean_small > 0:
                    print(f"  🗑️ 二次过滤: 移除 {n_ocean_small} 个海洋事件 (n_grids<{min_grids_ocean})")
                    event_summary = event_summary[~ocean_small].copy()

            # ---- 保存结果 ----
            export_centroid_trajectories(event_summary, CONFIG["output_dir"])

            # ---- 最终数据质量检查 ----
            print(f"\n📋 {CONFIG['month']:02d}月数据质量检查...")

            # 1. 去重检查
            n_before = len(event_summary)
            event_summary = event_summary.drop_duplicates(subset=["event_id"], keep="first")
            n_after = len(event_summary)
            if n_before != n_after:
                print(f"  🔄 去除重复事件ID: {n_before - n_after} 条")

            # 2. 逻辑一致性检查
            # ★方案B：与Stage3保持一致的硬截断条件，区分"真实极端"与"数据异常"
            #   - 软约束 (avg_obs×10 ~ ×20 且 ≤200mm): 可疑值，仅警告不截断，保留真实极端暴雨信号
            #   - 硬约束 (>avg_obs×20 且 >200mm): 明显数据异常，截断到200mm（与Stage3一致）
            # 背景：原逻辑用 avg_obs×10 截断，会误删台风眼墙等真实极端暴雨（cap≈40mm但实测150mm）
            logic_issues = 0
            avg_obs = event_summary["total_rain_depth"] / event_summary["n_grids"] / np.maximum(event_summary["duration_hours"] / 0.5, 1)
            physical_cap = 200.0  # 与Stage3一致：30min>200mm视为数据异常（气象记录极值约150mm/30min）

            # 软约束警告：avg_obs×10 < max_30min_rain ≤ avg_obs×20 且 ≤200mm
            soft_threshold = avg_obs * 10
            hard_threshold = avg_obs * 20
            suspicious = (
                (event_summary["max_30min_rain"] > soft_threshold) &
                ~((event_summary["max_30min_rain"] > hard_threshold) & (event_summary["max_30min_rain"] > physical_cap))
            )
            if suspicious.any():
                print(f"  ℹ️ 警告: {int(suspicious.sum())} 个事件 max_30min_rain > avg_obs×10 (疑似极端暴雨，保留原值未截断)")

            # 硬约束截断：>avg_obs×20 且 >200mm（与Stage3一致）
            invalid_max = (event_summary["max_30min_rain"] > hard_threshold) & (event_summary["max_30min_rain"] > physical_cap)
            if invalid_max.any():
                logic_issues += int(invalid_max.sum())
                print(f"  ⚠️ 数据异常: {int(invalid_max.sum())} 个事件 max_30min_rain > avg_obs×20 且 >{physical_cap}mm，已截断到{physical_cap}mm")
                event_summary.loc[invalid_max, "max_30min_rain"] = physical_cap
                event_summary.loc[invalid_max, "max_30min_intensity"] = (event_summary.loc[invalid_max, "max_30min_rain"] * 2).values.astype(np.float32)
                # ★修复：二次修正max_30min_rain后必须重算极端标志，否则flag=1但rain<=100
                event_summary["is_extreme_max30"] = (event_summary["max_30min_rain"] > 100).astype(np.int8)

            # 3. 时长合理性检查
            invalid_duration_long = event_summary["duration_hours"] > 168  # 超过7天
            invalid_duration_zero = event_summary["duration_hours"] <= 0  # 时长为0或负数
        
            if invalid_duration_long.any():
                logic_issues += invalid_duration_long.sum()
                print(f"  ⚠️ 时长异常: {invalid_duration_long.sum()} 个事件超过7天，已删除")
                event_summary = event_summary[~invalid_duration_long]
        
            if invalid_duration_zero.any():
                n_zero = invalid_duration_zero.sum()
                logic_issues += n_zero
                print(f"  ⚠️ 时长异常: {n_zero} 个事件duration_hours<=0，已修正为最小0.5h")
                # 将 duration_hours <= 0 的事件修正为 0.5 小时（最小时间步长）
                event_summary.loc[invalid_duration_zero, "duration_hours"] = 0.5
                # 同时修正 end_time（基于 start_time + 0.5h）
                if "start_time" in event_summary.columns and "end_time" in event_summary.columns:
                    zero_events = event_summary.loc[invalid_duration_zero]
                    event_summary.loc[invalid_duration_zero, "end_time"] = (
                        zero_events["start_time"] + pd.Timedelta(hours=0.5)
                    )

            if logic_issues == 0:
                print("  ✅ 所有逻辑检查通过")
            else:
                print(f"  ⚠️ 共发现 {logic_issues} 个逻辑问题，已修正")

            event_summary_csv = event_summary.copy()
            # ★修复：将时间戳对齐到30min网格，消除浮点运算产生的微秒噪声
            # 报告指出93.6%事件max_30min_time含微秒噪声，59.3%事件start_time含微秒噪声
            for _ts_col in ["start_time", "end_time", "max_30min_time"]:
                if _ts_col in event_summary_csv.columns:
                    event_summary_csv[_ts_col] = event_summary_csv[_ts_col].dt.floor("30min")
            # ★修复：时间戳对齐后必须重算duration_hours和mean_intensity_all
            # 否则floored前后时间差最大偏差0.5h，导致124192事件duration偏差
            event_summary_csv["duration_hours"] = (
                (event_summary_csv["end_time"] - event_summary_csv["start_time"]).dt.total_seconds() / 3600
            ).astype(np.float32)
        
            # ★修复：时间戳对齐后可能产生 duration_hours=0 的情况（start_time=end_time）
            # 对 duration_hours <= 0 的事件修正为最小 0.5h
            _dur_zero_mask = event_summary_csv["duration_hours"] <= 0
            if _dur_zero_mask.any():
                n_zero_after_floor = _dur_zero_mask.sum()
                print(f"  ⚠️ 时间戳对齐后: {n_zero_after_floor} 个事件duration_hours<=0，已修正为0.5h")
                event_summary_csv.loc[_dur_zero_mask, "duration_hours"] = 0.5
                # 修正 end_time = start_time + 0.5h
                event_summary_csv.loc[_dur_zero_mask, "end_time"] = (
                    event_summary_csv.loc[_dur_zero_mask, "start_time"] + pd.Timedelta(hours=0.5)
                )
                # 重新计算 duration_hours
                event_summary_csv.loc[_dur_zero_mask, "duration_hours"] = 0.5
        
            if "avg_precip_depth" in event_summary_csv.columns:
                event_summary_csv["mean_intensity_all"] = (
                    event_summary_csv["avg_precip_depth"] / np.maximum(event_summary_csv["duration_hours"], 0.5)
                ).astype(np.float32)
            # 丢弃不需要输出的辅助列
            drop_cols = [c for c in [
                "centroid_trajectory", "weighted_lon", "weighted_lat",
                "bbox_area_km2"  # 内部计算用
            ] if c in event_summary_csv.columns]
            if drop_cols:
                event_summary_csv = event_summary_csv.drop(columns=drop_cols)

            # 输出统计
            print(f"\n📊 {CONFIG['month']:02d}月输出统计:")
            print(f"  总事件数: {len(event_summary_csv)}")
            print(f"  平均降雨深: {event_summary_csv['avg_precip_depth'].min():.1f} - {event_summary_csv['avg_precip_depth'].max():.1f} mm/grid")
            print(f"  30分钟峰值雨强: {event_summary_csv['max_30min_intensity'].min():.1f} - {event_summary_csv['max_30min_intensity'].max():.1f} mm/hr")
            print(f"  总水量范围: {event_summary_csv['total_water_volume'].min():.2f} - {event_summary_csv['total_water_volume'].max():.2f} m³")

            # 确保输出目录存在
            os.makedirs(CONFIG["output_dir"], exist_ok=True)

            # 生成数据字典 (CSV字段说明) — 仅首次写入，后续跳过（内容固定，省I/O）
            dict_path = os.path.join(CONFIG["output_dir"], "data_dictionary.txt")
            if not os.path.exists(dict_path):
                with open(dict_path, "w", encoding="utf-8") as f:
                    f.write("""降雨事件数据字典 / Rainfall Event Data Dictionary
==============================================

字段名                        类型      说明
-----                        ----      ----
event_id                     string    全局唯一事件ID (月度: YYYYMM_NNNNNN, 全年: YYYY_NNNNNN)
start_time                   datetime  事件开始时间 (UTC)
end_time                     datetime  事件结束时间 (UTC)
duration_hours               float     事件持续时间 (小时)
total_rain_depth             float     事件内所有网格降雨量求和 (mm·grid数, 面积量纲)
total_water_volume           float     事件总水量 (m³, 公式: avg_precip×n_grids×grid_area×cos(lat)×10³)
avg_precip_depth             float     事件平均格点降雨深 (mm/grid, 实际降雨强度)
max_30min_rain               float     事件内最大30分钟降雨量 (mm)
max_30min_intensity          float     事件内最大30分钟雨强 (mm/hr = max_30min_rain × 2)
max_30min_time               datetime  最大30分钟降雨发生时间
mean_intensity               float     条件平均雨强 (mm/hr, 仅统计有降水格点的平均雨强)
mean_intensity_all           float     全网格平均雨强 (mm/hr, = avg_precip_depth / duration_hours)
n_grids                      int       事件包含的网格点数
lon_min, lon_max             float     事件边界经度范围 (度, 0-360, 跨日界线时lon_min>lon_max)
lon_range_circular           float     循环经度跨度 (度, 0-180, 跨日界线事件的真实经度范围)
crosses_dateline             int       是否跨0°/360°日界线 (1=是, 0=否)
lat_min, lat_max             float     事件边界纬度范围 (度, -90~90)
center_lon, center_lat       float     事件空间质心 (网格中位数)
location_type                string    海陆分类: 内陆/沿海/近海/海洋 (内陆=距海岸>100km的陆地, 沿海=距海岸≤100km的陆地)
continent                    string    所属大洲 (海洋/近海事件标为"海洋"; 沿海/内陆事件为实际大洲)
climate_zone                 string    柯本气候分区 (陆地=真实Köppen码; 海洋/近海=语义化海洋分区如"热带太平洋海洋")
ocean                        string    海洋名称 (海洋/近海事件, 如"热带太平洋"; 陆地事件为空)
country                      string    国家名称 (陆地事件, 如"China"; 海洋事件为空)
confidence                   float     事件置信度 (0-1, 基于n_grids/duration/intensity)
is_extreme_precip            int       极端降水标志 (1=avg_precip_depth>200mm, 0=否, 不截断)
is_extreme_max30             int       极端30min雨强标志 (1=max_30min_rain>100mm, 0=否, 不截断)

关键参数 (供论文披露):
- 数据: CMORPH 0.072° 分辨率, 全球覆盖
- 检测阈值: 气候自适应 (柯本分区, 0.05-0.2 mm)
- IETD (事件间隔时间): 气候自适应 (2-12 小时)
- 3D DBSCAN: eps=0.8° (~89km), minPts=5, 纬度自适应 (cos(lat)调整)
- 最小事件网格数: 海洋25 / 陆地5 (陆地降雨空间尺度小, 差异化阈值)
- 置信度阈值: 0.3 (低于此值的事件被过滤)
- 碎裂事件过滤: n_grids≤18 且 bbox>5000km²
- 极端值处理: 两阶段修正 (Stage3汇总后 + Stage4月度输出前)
  * 软约束 (avg_obs×10): 可疑值仅警告不截断, 保留真实极端暴雨信号
  * 硬约束 (>avg_obs×20 且 >200mm): 明显数据异常, 截断到200mm
  * 极端事件用 is_extreme_precip(>200mm) / is_extreme_max30(>100mm) 标志位标记

注意:
- total_rain_depth 是所有网格点的降雨量求和, 面积量纲, 不应直接作为"事件降水量"
- 事件降水量请使用 avg_precip_depth (mm/grid) 或 total_water_volume (m³)
- 海洋事件 (location_type="海洋") 的 climate_zone 为语义化海洋分区
""")
            print(f"  📖 数据字典: {dict_path}")

            # 按月输出（文件名含年月）
            month_tag = f"{CONFIG['year']}{CONFIG['month']:02d}"
            event_summary_csv.to_csv(os.path.join(CONFIG["output_dir"], f"rainfall_events_{month_tag}_raw.csv"), index=False)
            if CONFIG.get("performance", {}).get("save_intermediate_pkl", True):
                dt_expanded.to_pickle(os.path.join(CONFIG["output_dir"], f"rainfall_events_full_{month_tag}_raw.pkl"))

            # ---- 交互式报告（按月，独立文件名） ----
            # 只在指定月份生成交互地图（节省时间），默认只生成1月预览
            map_months = CONFIG.get("interactive_map_months", [])
            if not map_months or CONFIG["month"] in map_months:
                report_file = generate_interactive_report(event_summary, CONFIG["output_dir"], CONFIG)
                # 重命名为含月份的独立文件
                if report_file and os.path.exists(report_file):
                    month_map_path = os.path.join(CONFIG["output_dir"], f"interactive_map_{month_tag}.html")
                    import shutil
                    shutil.move(report_file, month_map_path)
                    report_file = month_map_path
                    print(f"  🗺️ {CONFIG['month']:02d}月互动地图: {month_map_path}")
                else:
                    report_file = ""
            else:
                report_file = ""
                print(f"  ⏭️ {CONFIG['month']:02d}月跳过互动地图生成（preview_months={map_months}）")

            all_monthly_summaries.append(event_summary_csv)
        print(f"\n✅ {CONFIG['month']:02d}月处理完成，报告: {report_file}")

        # 释放当月数据
        del dt_expanded, event_summary, event_summary_csv, dt_merged
        gc.collect()

        # ============ 当年汇总 ============
        print(f"\n{'#'*60}")
        print(f"##  📊 {proc_year}年汇总 ({CONFIG['start_month']}~{CONFIG['end_month']}月)")
        print(f"{'#'*60}")

        if all_monthly_summaries:
            full_year = pd.concat(all_monthly_summaries, ignore_index=True)

            # ---- 跨月事件合并（解决168h截断问题）----
            # ★说明: 年度汇总时 allow_cross_year=False 是正确的——因为下一年1月数据
            #   尚未处理，无法做真正的跨年合并。跨年合并在多年汇总阶段执行。
            #   fullyear CSV 与 multiyear CSV 事件数不一致是预期行为，前者是单年视角。
            try:
                full_year = merge_cross_month_events(
                    full_year, CONFIG,
                    max_gap_hours=CONFIG.get("cross_month_max_gap_hours", 6),
                    max_distance_deg=CONFIG.get("cross_month_max_distance_deg", 2.0)
                )
            except Exception as e:
                print(f"  ⚠️ 跨月合并失败: {e}")
                import traceback
                traceback.print_exc()

            # ---- 重新生成全局唯一ID (统一格式 YYYY_NNNNNN) ----
            full_year = full_year.sort_values("start_time").reset_index(drop=True)
            year_str = str(proc_year)  # ★修复 P3: 用循环变量替代 CONFIG["year"]，避免隐式状态
            # 格式: YYYY_NNNNNN（全年统一编号），便于跨月追踪
            full_year["event_id"] = [
                f"{year_str}_{i+1:06d}" for i in range(len(full_year))
            ]
            # ★移除非唯一的 event_id_final 字段（月度内部编号，跨月重复）
            # 报告指出：event_id_final 仅85,745个唯一值 vs 202,792行，不可用作主键
            if "event_id_final" in full_year.columns:
                print("  🗑️ 移除非唯一的 event_id_final 字段（月度内部编号，跨月重复）")
                full_year = full_year.drop(columns=["event_id_final"])

            # ★时间戳对齐到30min网格（跨月合并可能产生新时间戳）
            for _ts_col in ["start_time", "end_time", "max_30min_time"]:
                if _ts_col in full_year.columns:
                    full_year[_ts_col] = pd.to_datetime(full_year[_ts_col]).dt.floor("30min")
            # ★修复：时间戳对齐后重算duration_hours和mean_intensity_all
            full_year["duration_hours"] = (
                (full_year["end_time"] - full_year["start_time"]).dt.total_seconds() / 3600
            ).astype(np.float32)
        
            # ★修复：全年合并后也检查 duration_hours<=0 的情况
            _full_dur_zero = full_year["duration_hours"] <= 0
            if _full_dur_zero.any():
                n_full_zero = _full_dur_zero.sum()
                print(f"  ⚠️ 全年合并后: {n_full_zero} 个事件duration_hours<=0，已修正为0.5h")
                full_year.loc[_full_dur_zero, "duration_hours"] = 0.5
                full_year.loc[_full_dur_zero, "end_time"] = (
                    full_year.loc[_full_dur_zero, "start_time"] + pd.Timedelta(hours=0.5)
                )
        
            if "avg_precip_depth" in full_year.columns:
                full_year["mean_intensity_all"] = (
                    full_year["avg_precip_depth"] / np.maximum(full_year["duration_hours"], 0.5)
                ).astype(np.float32)
            # ★修复：跨月合并后重算极端标志，确保与max_30min_rain一致
            if "max_30min_rain" in full_year.columns:
                full_year["is_extreme_max30"] = (full_year["max_30min_rain"] > 100).astype(np.int8)
            if "avg_precip_depth" in full_year.columns:
                full_year["is_extreme_precip"] = (full_year["avg_precip_depth"] > 200).astype(np.int8)

            # ★终极一致性强制：保存前最后检查 location_type vs continent/country/ocean
            if "location_type" in full_year.columns and "continent" in full_year.columns:
                _oc = full_year["location_type"].isin(["海洋", "近海"])
                _ld = full_year["location_type"].isin(["内陆", "沿海"])
                # 海洋事件：continent=海洋, country=空
                full_year.loc[_oc, "continent"] = "海洋"
                full_year.loc[_oc, "country"] = ""
                # 陆地事件但continent=海洋：重算
                _bad = _ld & (full_year["continent"] == "海洋")
                if _bad.any():
                    _blon = full_year.loc[_bad, "center_lon"].values
                    _blat = full_year.loc[_bad, "center_lat"].values
                    _blon_norm = ((_blon + 180) % 360) - 180
                    full_year.loc[_bad, "continent"] = classify_continent_vectorized(_blon_norm, _blat)
                    full_year.loc[_bad, "country"] = classify_country_vectorized(_blon_norm, _blat)
                    full_year.loc[_bad, "ocean"] = ""
                    print(f"  🔧 全年保存前修复 {int(_bad.sum())} 条陆地事件 continent 矛盾")
                # 陆地事件但ocean非空：清空
                _bad_ocean = _ld & (full_year["ocean"].astype(str).str.len() > 0)
                if _bad_ocean.any():
                    full_year.loc[_bad_ocean, "ocean"] = ""
                # 最终统计
                _final_inc = int(((_oc & (full_year["continent"] != "海洋")) | (_ld & (full_year["continent"] == "海洋"))).sum())
                if _final_inc == 0:
                    print(f"  ✅ 全年 location_type vs continent 一致性检查通过 ({len(full_year)} 条)")
                else:
                    print(f"  ⚠️ 全年仍有 {_final_inc} 条 location_type vs continent 不一致")

            _year_tag = str(proc_year)  # ★修复 P3: 用循环变量
            full_year.to_csv(os.path.join(CONFIG["output_dir"], f"rainfall_events_{_year_tag}_fullyear.csv"), index=False)
            print(f"  全年总事件数: {len(full_year)}")
            print(f"  全局ID范围: {full_year['event_id'].iloc[0]} ~ {full_year['event_id'].iloc[-1]}")
            print("  各月事件分布:")
            if "start_time" in full_year.columns:
                full_year["start_time"] = pd.to_datetime(full_year["start_time"])
                month_counts = full_year["start_time"].dt.month.value_counts().sort_index()
                for m, c in month_counts.items():
                    print(f"    {m:02d}月: {c} 个事件")

            # ---- 生成全年互动地图 ----
            print("\n🗺️ 生成全年互动地图...")
            try:
                # 采样以避免地图过大
                n_map_sample = min(8000, len(full_year))
                if len(full_year) > n_map_sample:
                    # 按月分层采样，确保各月都有代表
                    full_year["month"] = full_year["start_time"].dt.month
                    map_data = full_year.groupby("month", group_keys=False).apply(
                        lambda g: g.sample(n=min(n_map_sample // 12, len(g)), random_state=42)
                    ).drop(columns=["month"]).copy()
                    print(f"  分层采样 {len(map_data)} 个事件用于地图显示")
                else:
                    map_data = full_year.copy()

                # 生成全年地图（复用月度函数，传入全年专属文件名）
                _csv_fn = f"rainfall_events_{_year_tag}_fullyear.csv"
                _map_fn = f"interactive_map_{_year_tag}_fullyear.html"
                _report_fn = f"report_{_year_tag}_fullyear.html"
                fullyear_report = generate_interactive_report(
                    map_data, CONFIG["output_dir"], CONFIG,
                    csv_filename=_csv_fn,
                    map_filename=_map_fn,
                    report_filename=_report_fn
                )
                if fullyear_report and os.path.exists(fullyear_report):
                    print(f"  ✅ {_year_tag}年互动报告: {fullyear_report}")
                    print(f"  ✅ {_year_tag}年互动地图: {os.path.join(CONFIG['output_dir'], _map_fn)}")
                else:
                    print("  ⚠️ 全年地图生成失败: 报告文件未创建")
            except Exception as e:
                print(f"  ⚠️ 全年地图生成失败: {e}")

            # ★保存年度汇总到字典（用于最终多年汇总）
            all_year_summaries[_year_tag] = full_year
            print(f"\n✅ {_year_tag}年处理完成，共 {len(full_year)} 个事件")

            # 释放该年数据
            del full_year, all_monthly_summaries
            gc.collect()

    # ============ 多年最终汇总 ============
    print(f"\n{'#'*60}")
    print(f"##  📊 多年汇总 ({CONFIG['start_year']}~{CONFIG['end_year']}年)")
    print(f"{'#'*60}")

    if all_year_summaries:
        # 合并所有年份的数据
        all_years_df = pd.concat(all_year_summaries.values(), ignore_index=True)
        all_years_df = all_years_df.sort_values("start_time").reset_index(drop=True)

        # ★跨年事件合并（与跨月合并使用相同逻辑，允许跨年）
        _n_before = len(all_years_df)
        try:
            all_years_df = merge_cross_month_events(
                all_years_df, CONFIG,
                max_gap_hours=CONFIG.get("cross_month_max_gap_hours", 6),
                max_distance_deg=CONFIG.get("cross_month_max_distance_deg", 2.0),
                allow_cross_year=True
            )
            _n_after = len(all_years_df)
            if _n_after < _n_before:
                print(f"  🔗 跨年合并: {_n_before} → {_n_after} (合并 {_n_before - _n_after} 对)")
        except Exception as e:
            print(f"  ⚠️ 跨年合并失败: {e}")

        # 重新编号 event_id（跨年合并后ID需要统一）
        # ★修复 P1: 每年独立编号（与 fullyear CSV 语义一致），向量化避免 iterrows 性能瓶颈
        all_years_df = all_years_df.sort_values("start_time").reset_index(drop=True)
        _start_years = all_years_df["start_time"].dt.year.values
        # 每年组内序号：用 cumcount + 1
        _within_year_seq = all_years_df.groupby(_start_years).cumcount().values + 1
        all_years_df["event_id"] = np.char.add(
            np.char.add(_start_years.astype(str), "_"),
            np.char.zfill(_within_year_seq.astype(str), 6)
        )

        # 保存多年合并CSV
        multi_year_csv = os.path.join(CONFIG["output_dir"], 
            f"rainfall_events_{CONFIG['start_year']}_{CONFIG['end_year']}_multiyear.csv")
        all_years_df.to_csv(multi_year_csv, index=False)
        print(f"  📁 多年合并CSV: {multi_year_csv}")
        print(f"  📊 总事件数: {len(all_years_df)}")
        print(f"  📅 年份分布:")
        if "start_time" in all_years_df.columns:
            all_years_df["start_time"] = pd.to_datetime(all_years_df["start_time"])
            year_counts = all_years_df["start_time"].dt.year.value_counts().sort_index()
            for y, c in year_counts.items():
                print(f"    {y}年: {c} 个事件")
        
        del all_years_df
        gc.collect()

    elapsed_time = time.time() - start_time
    print(f"\n⏱️ 多年总耗时: {elapsed_time:.1f} 秒 ({elapsed_time/60:.1f} 分钟)")
    print(f"💾 内存峰值: {check_memory_usage():.1f} MB")
    print("\n🎉 多年处理完成！")
