# 美国租房价格预测（Apartment Rent Price Prediction）

本项目旨在根据房源的结构、地理位置、文本描述与设施信息，预测公寓月租金（`price`）。项目包含一条**端到端的可复现实验管线**：数据清洗 → 特征工程 → 模型选择与调参 → 评估与导出结果。

---

## 📦 数据与任务

* **任务类型**：回归（目标变量：`price`，单位 USD）
* **核心字段（字段释义）**：

  * `amenities`：可用设施列表（例如空调、篮球场、有线/宽带、健身房、泳池、冰箱等）。
  * `bathrooms`：公寓中的卫生间数量。
  * `bedrooms`：公寓中的卧室数量。
  * `currency`：价格最初使用的货币类型。
  * `fee`：与公寓相关的任何额外费用。
  * `has_photo`：布尔值，指示该房源是否包含照片。
  * `pets_allowed`：允许的宠物类型（如狗、猫等）。
  * `price_type`：以USD标准化的价格类型。
  * `square_feet`：公寓面积（平方英尺）。
  * `address`：详细地址。
  * `cityname`：公寓所在的城市。
  * `state`：公寓所在的州。
  * `latitude`：公寓位置的地理纬度。
  * `longitude`：公寓位置的地理经度。
  * `source`：获取该房源的平台或网站。
  * `time`：房源创建的时间戳。
  * `price`：数值型租金价格（目标变量）。

> 说明：仓库中提供了拆分后的 `train.csv` 与 `test.csv`（两者均含 `price` 以便最终测试指标计算）。

---

## 🧹 数据清洗与特征工程（Design Choices）

> 下述思路与实现与 `notebooks/predict_house_price.ipynb` 中保持一致。

### 1) 列删减（Column Pruning）

* **删除 `currency` 与 `price_type`**：这两列在数据中**取值恒定**，对模型回归无信息增益，保留只会增加无效维度。
* **删除 `address`**：该列**缺失值极多且难以可靠填充**；同时我们已拥有 `cityname / state / latitude / longitude`，能更稳定地承载地理信息，故删去以降低噪声。

### 2) 设施与宠物（Multi-hot Encoding）

* **`amenities` 的缺失值 → `Nothing`**：

  * 原因：若直接空值将被独热/多热编码忽略，等价于**丢失一种“无设施”的语义**。用占位符保留“无”的信息，有助于模型识别“缺省=无”。
  * 实现：`SimpleImputer(fill_value='Nothing')` → 字符串按逗号切分 → `MultiLabelBinarizer`（自定义 `MultiHotEncoder`）。
* **`pets_allowed` 的缺失值 → `No`**：

  * 原因同上：保留“默认不允许宠物”的含义，避免因为缺失而被编码器忽略。
  * 实现：`SimpleImputer(fill_value='No')` → 逗号切分 → 多热编码。

### 3) 类别降噪与稳健独热（Low-Card & Rare Bucketing）

* **`category` 清洗与映射**：将原始路径式类别清洗并**归并为少量语义稳定的类**（如 `apartment`, `home`, `condo`, `short_term`, `commercial/retail`, `others`）。
* **`fee / has_photo / source`**：

  * 用众数填补缺失。
  * 使用 `OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=0.05)`，自动将**低频类别合并为“稀有桶”**，抑制高维稀疏与过拟合。

### 4) 文本建模（Title & Body）

* **清洗**：

  * 统一小写，去 URL / 邮箱，标准化空白。
  * **价格遮蔽**：用占位符 `__PRICE__` 屏蔽文案里的货币/数额模式（避免把“要价”泄漏给特征，造成目标泄漏）。
* **表示**：

  * `TfidfVectorizer`（词 n-gram + 适度的字符 n-gram），`min_df≥3`、`sublinear_tf`、`smooth_idf`、`strip_accents='unicode'`。
  * **标题加权融合**：通过 `ColumnTransformer` 将 `title` 与 `body` 按权重**3:1**线性池化（`title_weight=3.0`），以强调标题的强信号。

### 5) 结构三件套（bedrooms/bathrooms/square_feet）

* 统一转数值；使用**自定义相似度填充器** `BedBathSqftSimilarityImputer`：

  * `square_feet`：优先在**相同 (bedrooms, bathrooms)** 组内用均值填充。
  * `bedrooms`：以**面积相近 + 相同 bathrooms** 的样本做 KNN 式均值；兜底用 `bathrooms` 分组均值与全局均值。
  * `bathrooms`：以**面积相近 + 相同 bedrooms** 的样本做 KNN 式均值；兜底与上同。

### 6) 地理特征（City/State/Lat/Lon）

* 自定义 `GeoCityStateImputer`：

  * `state` 缺失：先用 `(cityname, lat, lon)` 估计；不行则用 `cityname` 众数；再不行用 (lat, lon) 最近邻或全局众数。
  * `cityname` 缺失：保证 `state` 已填后，在同州内按 (lat, lon) 最近邻；兜底用该州众数或全局众数。
  * `latitude/longitude` 缺失：在同 `(state, cityname)` 组内用均值，最终兜底全局均值。
* 构造 `city_state = cityname + ', ' + state` 并做稳健独热；经纬度数值特征**直通**。

### 7) 时间编码（Time）

* 将时间戳映射为：`sin(month)`、`cos(month)` 与 `days_since_first`（相对项目内最早时间的天数），同时具备**季节周期性与时间距信息**。

---

## 🧱 特征管线（sklearn `ColumnTransformer`）

* 子管线：`category_pipe`, `title_body_pipeline`, `amenities_pipe`, `pets_allowed_pipe`, `fee_photo_source_pipe`, `bed_bath_sqft_pipe`, `geo_pipe`, `time_pipe`。
* 组合后作为 `preprocess` 接到最终模型之前。

---

## 🤖 模型与调参

* **模型**：`XGBRegressor`

  * 目标：`reg:absoluteerror`（直接以 MAE 优化）
  * 训练：`tree_method='hist'`，`max_bin=128`，`min_child_weight=6.0`，`subsample=0.8`
  * 正则：`reg_alpha=0.1`，`reg_lambda=5.0`
  * 设备：默认 `device='cuda'`（如无 GPU，可改为 `device='cpu'` 或删除该参数）
* **搜索策略**：`HalvingGridSearchCV`（5 折，评分 `neg_mean_absolute_error`）

  * 网格：

    * `max_depth ∈ {2, 3, 4}`
    * `n_estimators ∈ {2000, 2200, 2400, 2600, 2800}`
    * `colsample_bytree ∈ {0.25, 0.26, 0.27}`

---

## 📈 实验结果（在仓库提供的 `train.csv`/`test.csv` 上）

* **交叉验证（5-Fold CV）最佳**：

  * 最优参数：`{max_depth=2, n_estimators=2800, colsample_bytree=0.26}`
  * 最佳 CV **MAE ≈ 203.56**
* **独立测试集**：

  * **MAE ≈ 196.94**
  * **MAPE ≈ 12.10%**

> 注：以上指标来自 `notebooks/predict_house_price.ipynb` 的实际运行输出。由于随机性与环境差异，数值可能有轻微波动（`RANDOM_STATE=42`）。

---

## 🗂️ 仓库结构

```
.
├── notebooks/
│   └── predict_house_price.ipynb      # 主实验笔记本（端到端管线）
├── data/                              
│   ├── train.csv                      # 本项目训练集（含 price）
│   └── test.csv                       # 本项目测试集（含 price）
├── requirements.txt                   # 复现实验环境
├── README.md                          # 项目说明（本文档）
└── LICENSE                            # 许可证
```

---

## 🚀 快速开始

### 1) 环境准备

```bash
# Python ≥ 3.10 建议
python -m venv .venv && source .venv/bin/activate  # Windows 使用 .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

> 关键依赖：`pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`。

### 2) 运行实验

* 将 `train.csv` 与 `test.csv` 放至仓库根目录或 `data/`。
* 打开并依次运行 `notebooks/predict_house_price.ipynb`。
* 运行结束将输出最优参数、CV 结果与测试集指标，同时可导出预测。

---

## 🧠 设计取舍与复盘

* **缺失值不“消失”**：`amenities` 与 `pets_allowed` 的空值显式编码为 `Nothing/No`，保证“无/未知”被模型看见；实践中常优于把缺失当作“样本缺少特征”而忽略。
* **恒定列清理**：`currency/price_type` 恒为单一值，早删减可避免无效维度与数据泄漏风险。
* **地址删除**：`address` 稀缺且难填，且与 `city/state/lat/lon` 存在强替代关系，保留反而引入噪声与不稳定性。
* **文本防泄漏**：遮蔽描述中的显性价格，防止模型“抄答案”。
* **稳健独热**：对高基数类别启用低频折叠（`infrequent_if_exist`）以控维度爆炸与过拟合。
* **地理与时间**：同时捕获季节周期与相对时间，地理则融合离散位置（`city_state`）与连续坐标（lat/lon）。

---

## 🔭 可进一步优化的方向

* **地理编码**：引入 H3/Geohash 或基于 POI/学区/通勤的外部特征。
* **文本表示**：用 `fastText`/`Sentence-BERT`/小型 `Transformer` 替代/补充 TF‑IDF。
* **类别编码**：对高基数列尝试目标编码（带 K-Fold 与正则化）或 CatBoost。
* **模型集成**：与 `LightGBM`/`CatBoost`/线性模型混合、Stacking/Blending。
* **稳健性评估**：时序切分、城市间迁移验证、误差分布与异常值分析。
* **业务约束**：对不合理预测施加单调性约束或分段校准。
