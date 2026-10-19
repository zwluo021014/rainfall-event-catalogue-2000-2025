# 降雨事件划分规范（G1 闭合件）

> **用途**：闭合三盲审唯一的 P0 缺口 G1（事件分割脚本未定位），使事件表生成规则**可被第三方独立实现**。
> **权威源文件**：`C:\Users\Administrator\Desktop\脚本\rainfall_event_detection.py`
> —— **版本 v18.0**（"多年批量处理版 2000-2025"，6,077 行，294,190 B，2026-08-21 09:56）
> —— **SHA-256**：`78084f7f5c06992ebc66063f2e645d6dedad363c35528b0eff451fcc31017d57`
> **本规范依据**：源码逐函数核验（非文档转述），核验日期 2026-09-21。

---

## 0. 版本与副本差异（复现前必读）

| 项 | 内容 |
|---|---|
| **生产版本** | `Desktop\脚本\rainfall_event_detection.py`，**v18.0**，6,077 行，SHA-256 `78084f7f…17d57` |
| **另一副本** | `Desktop\MouseWithoutBorders\rainfall_event_detection.py`，**v17.0**，3,881 行，SHA-256 `9c90e6fa…224c9` |
| 两副本差异 | **9,958 行差异**（非同一文件的小改） |
| ⚠ **口径判定** | 事件表输出目录 `F:\output_CMORPH_2000_2025` 由 **v18.0** 的 CONFIG 指定（`data_dir=E:\date download\CMORPH`，`output_dir=F:\output_CMORPH_2000_2025`），故 **v18.0 为生产口径**。v17.0 副本是前代版本，**不得用于复现事件表**。 |
| **处置建议** | 归档 v18.0（含哈希与行数）；把 v17.0 副本重命名为 `.superseded_v17`，或在副本内加一行横幅注明"非生产版本，勿用" |

---

## 1. 十步流水线（源码级，逐步骤可执行）

```
[1] 数据读取      按天读 CMORPH NetCDF → (grid, time) 长表，缺失值线性插值
[2] 网格事件识别  逐网格 IETD 分析：降水 ≥ threshold 起，无雨间隔 > IETD 止
[3] 跨日合并      同一网格、间隔 ≤ IETD 的片段合并为一个网格事件
[4] DBSCAN 聚类   网格事件按 时空邻近 聚为天气尺度事件（球面距离 + 时间缩放）
[5] 动态切分      簇内分裂检测：面积骤减 / 强度骤降 / 质心跳跃 → 切分
[6] 事件汇总      聚合起止时间、总雨量、峰值、位移轨迹质心
[7] 地理分类      海陆（内陆/沿海/近海/海洋）+ 大洲 + 国家 + 海洋名 + 柯本气候区
[8] 跨月合并      并查集合并跨月边界的同一事件（≤6 h 且 ≤2°）
[9] 全年统一 ID   跨月合并后重编号 `YYYY_NNNNNN`
[10] 输出         CSV + 交互式 HTML 地图 + 静态报告 + 数据字典
```

---

## 2. 步骤 2：网格级事件识别（IETD 阈值法）

**核心函数** `RainfallEventDetector._extract_events_numba`（L811–873）与批量版 `_extract_all_events_batch`（L686、L877）。

**判定规则（源码逐行语义）**：

```
is_rain[t] = precip[t] >= threshold          # 严格大于等于
分段：连续 is_rain=True 的时段 → segment
合并：若 seg[i].start - seg[i-1].end <= ietd → 并入同一事件
      否则 → 结束当前事件，开启新事件
```

**每个网格事件记录 5 个量**：
`(start_time, end_time, total=Σprecip, max_rain=max(precip), peak_idx=argmax(precip))`

⚠ **两个易误读的实现细节**：

1. **无雨间隔的条件是"≤ IETD 则合并"**，即 `gap <= ietd` 合并。因此 IETD 是**最小无雨间隔（小时）**，不是最小间隔的下界排除。字面表述应为"无雨间隔超过 IETD 才判定为两个事件"。
2. `threshold` 与 `ietd` **逐网格不同**，来自气候区/纬度带查找（见 §3）。

---

## 3. 自适应阈值与 IETD（`get_koppen_params` / `get_adaptive_params`）

**优先级**：柯本气候分区（`use_koppen_climate=True`，需 shapefile 或 raster 就位）> 纬度带 > 全局兜底。

**柯本大区映射（`koppen_climate_params`，L303–323）**：

| 气候区 | threshold (mm/30min) | IETD (h) | 判读依据 |
|---|---|---|---|
| **A** 热带 | 0.05 | 4 | 对流雨多，阈值低 |
| **Af** 热带雨林 | 0.03 | 3 | 降水极频繁 |
| **Am** 热带季风 | 0.04 | 4 | |
| **Aw** 热带草原 | 0.06 | 5 | |
| **B** 干旱 | 0.20 | 12 | 降水少，阈值高 |
| **BSh** 半干旱 | 0.15 | 10 | |
| **BWh** 热带沙漠 | 0.25 | 15 | |
| **BSk** 半干旱（温带） | 0.18 | 12 | |
| **BWk** 温带沙漠 | 0.22 | 14 | |
| **C** 温带 | 0.10 | 6 | 适中 |
| **Cfa** 湿润亚热带 | 0.08 | 5 | |
| **Cwa** 亚热带季风 | 0.08 | 5 | |
| **Csa** 地中海气候 | 0.12 | 7 | |
| **D** 大陆性 | 0.15 | 8 | 冬季少、夏季多 |
| **E** 极地 | 0.20 | 12 | 降水极少 |
| **全局兜底** | **0.10** | **6** | `global_rain_threshold` / `global_IETD_hours` |

**纬度带回退（`lat_bands`，仅在无柯本数据时启用）**：

| 带 | 纬度 | threshold | IETD |
|---|---|---|---|
| tropical | \|lat\| < 23.5 | 0.05 | 4 |
| subtropical | 23.5–35 | 0.10 | 6 |
| temperate | 35–55 | 0.15 | 8 |
| polar | 55–90 | 0.20 | 12 |

**实现细节**：`build_climate_lut`（L477）预建 0.1° 分辨率气候区查找表（O(1) 查询）；`query_climate_lut`（L565）做数组化查询。若 shapefile 与 raster 均缺失，降级到 `_simple_climate_zone(lat)` 纬度带。

---

## 4. 步骤 3：跨日合并（`merge_cross_day_events`，L2817–2900）

**规则**：同一网格 `(x, y)` 内，按 `start_time` 排序，若 `gap = start_time[i] − end_time[i−1] > prev_ietd` 则视为新事件。

⚠ **关键实现细节（源码已标注的 P1 修复）**：跨日合并用的 IETD 是**前一事件所属气候区的 IETD**（`groupby(["x","y"])["ietd"].shift()`），而非全局 IETD。首事件无前值时用全局 IETD 兜底（`fillna`）。

**聚合规则**：

| 输出字段 | 聚合方式 |
|---|---|
| `start_time` / `end_time` | `min` / `max` |
| `total_rain` | `sum` |
| `max_30min_rain` | `max` |
| `max_30min_time` | **取对应 `max_30min_rain` 的时间点**（用 `idxmax()` 一次性取出，L2862/2882） |
| `mean_intensity` | **重算为 `total_rain / max(duration_hours, 0.5)`**（L2891） |
| `n_obs` | `sum` |
| **`duration_hours`** | **`(end_time − start_time)`，含段间间隔**（L2886） |

⚠ **源码已标注的 P0 修复（这是台账 `duration_hours` 口径的权威来源）**：`duration_hours` 原用 `sum`（累加子事件历时），会**漏掉段间间隔**并虚高 `mean_intensity`；现改为 `end − start`。**这一条直接决定 API 中 `duration_hours` 的含义与 clamp 行为。**

**时间缩放**：`time_num` = 相对首事件的小时数；`time_scaled = time_num × time_scale_factor`（默认 **10.0**）。

---

## 5. 步骤 4：DBSCAN 时空聚类（`cluster_events_dbscan`，L3118+）

**输入**：`[lon, lat, time_scaled]` 三维；经度先归一（`lon > 180 → lon − 360`，L3129）。

**参数**：
- `dbscan_eps_deg = 0.8`（度）、`dbscan_minPts = 10`
- 备选网格搜索：`eps ∈ arange(0.5, 1.3, 0.15)`、`minPts ∈ {6, 8, 10, 12, 15}`（选择准则 = 噪声比 < 0.3 且在候选中 silhouette 最大；`dbscan_grid_search` L2907，`np.random.seed(123)`）
- **`enable_grid_search = False`**（当前生产口径直接用 0.8 / 10，不做搜索）

**纬度自适应 eps（L3136–3140）**：
```
eps_base_km = dbscan_eps_deg × 111
eps_km = eps_base_km × max(cos(median|lat|), 0.3)
```
即在中位纬度上把"度"换算为等效地面公里数，高纬自动缩小，下限 0.3 防极地过小。

**minPts 纬度自适应（L3171–3177，源码标为"修复 P2"）**：高纬降雨稀疏，按 `cos(lat)` 放宽，下限 3。

**预分区优化**（`dbscan_pre_partition=True`，`n_events > 5000` 时启用）：10 经 × 6 纬 = 60 块独立聚类，块边界割裂的簇由 `_merge_cross_block_clusters`（L2975）回并。

⚠ **块边界回并规则（`_merge_cross_block_clusters`）**：相邻簇若 **质心距离 < 2°**（deg）且 **时间重叠或间隔 < 6 h**，则合并。**这是一条容易在复现时遗漏的步骤**——它意味着聚类结果依赖分块方案，不完全是"纯 DBSCAN"。

**网格搜索采样**：`len(dt_events) > 50000` 时随机抽 50,000（seed=123）。

---

## 6. 步骤 5：动态切分（`dynamic_split_events`，L3288–3394）

**只在** `spatial_cluster != −1` 的成员上进行。`split_method = "dynamic"`（另可选 `"fixed"` 配 `fixed_split_hours = 48`，生产未用）。

**三个切分点判据**（在 cluster 内按 `time_bin = time_num / 0.5` 聚合后逐 bin 计算）：

| 判据 | 公式 | 默认阈值 |
|---|---|---|
| **cp1 面积骤减** | `area_chg = area.pct_change() < −area_threshold` | `area_change_threshold = 0.4` |
| **cp2 强度骤降** | `int_drop = (prev_int − mean_int)/prev_int > intensity_threshold` **且** `int_drop < 1` | `intensity_drop_threshold = 0.5` |
| **cp3 质心跳跃** | `sqrt(dlon² + dlat²) > 5.0°`，`dlon` 循环归一化到 [−180, 180] | 硬编码 **5.0°**（≈500 km） |

**后处理**：
1. 首点与末点不可作切分点；`cut_points` 去重排序。
2. **最小历时约束**：`min_split_duration_hours = 2.0`——仅当切分点**前后两段都 ≥ 2.0 h** 才保留该切分点。
3. cluster 内成员 < 10 个则**不切分**（`len(sub_df) < 10 → 全归 1 组`）。

**月度自适应阈值（`split_params_by_month`，`adaptive_split = True`）**：

| 月份 | area_change_threshold | intensity_drop_threshold | 依据 |
|---|---|---|---|
| 1、2 | 0.50 | 0.60 | 冬季层状雨，变化慢 |
| 3 | 0.45 | 0.55 | |
| 4、5 | 0.40 | 0.50 | 春季对流雨增多 |
| **6、7、8** | **0.35** | **0.45** | **夏季放宽，避免对流正常波动被误切** |
| 9 | 0.40 | 0.50 | |
| 10 | 0.45 | 0.55 | |
| 11、12 | 0.50 | 0.60 | |

⚠ **注意**：`adaptive_split=True` 时，`area_change_threshold` / `intensity_drop_threshold` 的**全局值（0.4 / 0.5）被月度表覆盖**（按事件所在月取）。复现时必须使用月度表，否则与事件表不符。

**切分后**：`temp_event_id` 按 `(spatial_cluster, temp_group)` 重新编号 → `event_duration = groupby(temp_event_id)["duration_hours"].sum()` → `final_event_id`。

---

## 7. 步骤 6：事件汇总与质心（`summarize_events` L3523 / `expand_events` L3430）

**位移轨迹**：加权线性回归拟合质心位移，**含经度解缠绕**（`unwrap_lon` L3499）。

**质心经度**：**循环加权平均**（`circular_weighted_lon` L3474），处理跨 0° 经线。

**极端值**：**只加标志位，不截断**，保留原始值。（这解释了为何台账 `total_rain_depth` 可达 1.72×10⁵。）

---

## 8. 步骤 7：地理分类

| 分类 | 方法 |
|---|---|
| **海陆类型** | `sjoin + buffer`，4 类：内陆 / 沿海 / 近海 / 海洋；`coastal_buffer_km = 100`（源码注明"扩大以避免陆地事件被误判为海洋"） |
| **大洲 / 国家** | `reverse_geocoder`（离线，优先级高于 sjoin，L83+） |
| **海洋名称** | `classify_ocean_vectorized`（L4793） |
| **柯本气候区** | 查找表（同 §3）或 raster 查询 |

对应台账字段：`location_type`（海洋 207,527 / 内陆 48,189）、`climate_zone`、国家/大洲。

---

## 9. 步骤 8：跨月合并（`merge_cross_month_events`，L4884）

**算法**：**并查集（union-find）** 合并跨月边界的同一事件。

| 参数 | 值 |
|---|---|
| `cross_month_max_gap_hours` | **6 h** |
| `cross_month_max_distance_deg` | **2.0°**（循环经度归一化） |
| `max_gap_hours` / `max_distance_deg`（函数签名默认） | 6 / 2.0 |
| `allow_cross_year` | 参数存在，生产默认**不跨年** |

**⚠ 关键口径**：`avg_precip_depth` 用各片段**加权平均**（源码注明"避免虚高"）。**这一条直接影响台账 API 公式中 `avg_precip_depth` 的含义**——它是跨月合并后的加权平均日降水深度，不是单段值。

**步骤 9**：跨月合并后重新编号为 `YYYY_NNNNNN`。

---

## 10. 事件过滤器（决定事件表的样本边界）

| 参数 | 值 | 作用 |
|---|---|---|
| `min_rain_per_event` | 0.5 | 事件最小总雨量 (mm) |
| `min_area_pixels`（海洋） | **25** | 海洋事件最小网格数（滤碎裂噪声） |
| `min_area_pixels_land`（陆地） | **5** | 陆地降雨空间尺度小，阈值更低 |
| `min_confidence` | 0.3 | 最低置信度 |
| `min_duration_hours` | 0.5 | 最短历时 |
| `time_merge_hours` | 6 | 时间合并阈值 |
| `expand_search_radius` | 3 | 邻域搜索半径（网格） |
| `enable_validation` / `filter_invalid_events` | True / True | 事件校验与无效事件过滤 |

⚠ **`min_area_pixels` 海陆不对称（25 vs 5）是事件表 `location_type` 分布的重要成因**，须在方法节声明。

---

## 11. 与下游台账字段的对应关系（闭合"字段从哪来"）

| 台账字段 | 生成位置 | 口径要点 |
|---|---|---|
| `start_time` / `end_time` | 步骤 2 + 3 | UTC；跨日/跨月合并后为整段起止 |
| `duration_hours` | **步骤 3（L2886）** | **`end − start`，含段间间隔**（非 `sum`） |
| `total_rain_depth` | 步骤 3 `sum` | 不截断，极端值保留 |
| `avg_precip_depth` | **步骤 8** | **跨月合并后片段加权平均**（避免虚高） |
| `max_30min_rain` | 步骤 3 `max` | 峰值；`max_30min_time` 为其对应时刻 |
| `n_grids` | 步骤 4/6 | 事件规模（观测 5–3,370） |
| `center_lon` / `center_lat` | 步骤 6 | **循环加权平均**经度 |
| `location_type` | 步骤 7 | 4 类；`coastal_buffer_km = 100` |
| `climate_zone` | 步骤 7 | 柯本；与 §3 阈值表同源 |
| `confidence` | 步骤 2/6 | 观测 0.3–1（受 `min_confidence` 下截） |
| `crosses_dateline` | 步骤 6 | 经度解缠绕标记（观测 2,559） |
| `event_id` | **步骤 9** | `YYYY_NNNNNN` |

---

## 12. G1 闭合声明

**闭合结论**：事件分割脚本已定位、版本已锁定、哈希已登记、十步流水线与全部阈值/自适应规则已按源码逐条记录。

**因此，台账 §7.3.3 的复现级别 R3 由"未具备"升为"具备（条件性）"**：

> R3（全流程级）**已具备**——前提是归档 v18.0 并排除 v17.0 副本；且 `kg_shapefile_path` /
> `kg_raster_path`（柯本气候区）就位，否则阈值降级为纬度带（会产生不同的事件表）。

**投稿声明措辞可升级为**：

> "Rainfall events were delineated with an IETD-based per-grid segmentation followed by
> spatiotemporal DBSCAN clustering and intra-cluster dynamic splitting (script
> `rainfall_event_detection.py`, version 18.0, SHA-256 78084f7f…; all thresholds,
> climate-adaptive IETD values, and splitting criteria are listed in Methods Table SX).
> Full end-to-end regeneration from raw CMORPH fields is therefore supported."

**尚需补做的一项（真正的残余缺口）**：**替代切分规则的敏感性重跑**——
用不同 `ietd_candidates`（3/6/9/12）、不同 `area_change_threshold`、以及 `split_method="fixed"`（48 h）
各重跑事件表，报告事件计数、`duration_hours` 中位数、极端事件总量与池化中位的变化。
**这是三盲审要求的"≥2 组替代切分规则重跑"的落地清单**，也是把 G1 从"可复现性合规"
升级为"已量化误差通道"的最后一步。

---

## 13. 归档建议

| 动作 | 目标 |
|---|---|
| 归档 v18.0 本体 | `F:\rainfall_sediment_results\b1_pipeline\code_archive\rainfall_event_detection_v18.0.py`（含本规范） |
| 登记哈希 | `code_archive\MANIFEST.md`：文件名 / 版本 / 行数 / SHA-256 / 日期 |
| 隔离旧副本 | `MouseWithoutBorders\rainfall_event_detection.py` → 改名加 `.superseded_v17` 或加横幅 |
| 登记依赖 | numpy、pandas、xarray、scikit-learn、tqdm、**numba**（JIT，缺失则降级纯 Python 慢 5–15×）、geopandas（可选）、reverse_geocoder（可选） |
| 登记外部依赖 | 柯本 shapefile `E:\全球柯本气候区划\8ed04-main\气候区划\气候区划.shp`、raster `world_koppen1.tif` |
