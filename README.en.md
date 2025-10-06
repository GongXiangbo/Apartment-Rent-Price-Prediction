# US Rental Price Prediction

**Data Download**: Google Drive — [Click to Access Dataset](https://drive.google.com/drive/folders/1igJiDJNReEFYZuoK4TCpq8Ee6z6t-_Sp?usp=sharing)

This project aims to predict the monthly rent (`price`) of apartments based on their structural attributes, geographical location, text descriptions, and amenities. It features an **end-to-end reproducible experimental pipeline**: Data Cleaning → Feature Engineering → Model Selection & Tuning → Evaluation & Result Export.

---

## 📦 Data & Task

* **Task Type**: Regression (Target Variable: `price`, in USD)
* **Core Fields (Field Descriptions)**:
    * `amenities`: List of available amenities (e.g., A/C, basketball court, cable/broadband, gym, pool, refrigerator).
    * `bathrooms`: Number of bathrooms in the apartment.
    * `bedrooms`: Number of bedrooms in the apartment.
    * `currency`: The original currency of the price.
    * `fee`: Any additional fees associated with the apartment.
    * `has_photo`: Boolean indicating if the listing includes photos.
    * `pets_allowed`: Types of pets allowed (e.g., dogs, cats).
    * `price_type`: The type of price, standardized to USD.
    * `square_feet`: The area of the apartment in square feet.
    * `address`: The detailed street address.
    * `cityname`: The city where the apartment is located.
    * `state`: The state where the apartment is located.
    * `latitude`: The geographical latitude of the apartment's location.
    * `longitude`: The geographical longitude of the apartment's location.
    * `source`: The platform or website where the listing was sourced.
    * `time`: The timestamp when the listing was created.
    * `price`: The numerical rental price (target variable).

---

## 🧹 Data Cleaning & Feature Engineering (Design Choices)

> The following rationale and implementation are consistent with `notebooks/predict_house_price.ipynb`.

### 1) Column Pruning

* **Drop `currency` & `price_type`**: These two columns have constant values across the dataset, offering no informational gain for the regression model. Keeping them would only add useless dimensions.
* **Drop `address`**: This column has a **very high number of missing values and is difficult to fill reliably**. More importantly, we already have `cityname / state / latitude / longitude`, which provide a more stable and structured representation of geographical information. Therefore, `address` is removed to reduce noise and avoid information redundancy.

### 2) Amenities & Pets (Multi-hot Encoding)

* **`amenities` NaN → `Nothing`**:
    * **Rationale**: If missing values are left as `NaN`, most encoders will ignore them, which is equivalent to losing the crucial information that "no amenities are listed". By filling with a specific placeholder, we enable the model to explicitly distinguish between "no amenities list provided" and "an empty list was provided," preserving the "none" semantic.
    * **Implementation**: `SimpleImputer(fill_value='Nothing')` → Split string by comma → `MultiLabelBinarizer` (implemented as a custom `MultiHotEncoder`).
* **`pets_allowed` NaN → `No`**:
    * **Rationale**: Same logic as above. In the rental market, missing information usually implies "not allowed" rather than "all types allowed." Filling with `No` aligns better with business logic and prevents the encoder from ignoring the missing entries.
    * **Implementation**: `SimpleImputer(fill_value='No')` → Split string by comma → Multi-hot encoding.

### 3) Categorical Noise Reduction & Robust One-Hot (Low-Cardinality & Rare Bucketing)

* **`category` Cleaning & Mapping**: The original categories are in a verbose, path-like format with significant noise and high cardinality. We consolidate them into a few core semantic classes (e.g., `apartment`, `home`, `condo`), allowing the model to focus on the essential differences between types.
* **`fee / has_photo / source`**:
    * Impute missing values with the mode.
    * Use `OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=0.05)`. **Why merge infrequent categories?** Categories that appear very rarely in the dataset (e.g., a niche listing website) make it difficult for the model to learn stable patterns and can lead to overfitting. This setting automatically groups all categories with a frequency below 5% into a single "infrequent" feature, enhancing the model's generalization ability.

### 4) Text Modeling (Title & Body)

* **Cleaning**:
    * Convert to lowercase, remove URLs/emails, and normalize whitespace.
    * **Price Masking**: **Why mask prices?** Listing descriptions often contain the rent amount directly (e.g., "Rent is $1500"), which constitutes target leakage. Without removal, the model would learn to "copy the answer," leading to artificially high scores in cross-validation but poor performance in real-world scenarios where the price is unknown. We replace currency/numeric patterns in the text with a placeholder `__PRICE__` to force the model to learn from the true semantic content.
* **Representation**:
    * `TfidfVectorizer` (with word n-grams + moderate char n-grams), `min_df≥3`, `sublinear_tf`, `smooth_idf`, `strip_accents='unicode'`.
    * **Weighted Title Fusion**: Use a `ColumnTransformer` to create a weighted linear combination of the TF-IDF vectors from `title` and `body` with a 3:1 ratio. **Why give the title more weight?** The title is typically a highly condensed summary of the property's key selling points, possessing a much higher signal-to-noise ratio than the body. A higher weight helps the model capture critical information more effectively.

### 5) The Structural Trio (bedrooms/bathrooms/square_feet)

* Convert to numeric types; use a **custom `BedBathSqftSimilarityImputer`**:
* **Why not use simple mean/median imputation?** There is a strong physical correlation between a property's area, number of bedrooms, and number of bathrooms. Using a global mean could generate illogical data points like a "5-bedroom apartment with only 500 square feet," which would mislead the model. Our method performs local imputation by finding "similar" listings, resulting in more accurate and realistic estimates:
    * **`square_feet`**: Impute using the mean `square_feet` from listings with the **same (`bedrooms`, `bathrooms`) combination**.
    * **`bedrooms`**: Prioritize finding K-nearest neighbors with similar `square_feet` and the same `bathrooms`, then use the mean of their `bedrooms`. Fallback to using the mean from all listings with the same `bathrooms`, and finally, the global mean.
    * **`bathrooms`**: Logic is similar to `bedrooms`, finding neighbors with similar `square_feet` and the same `bedrooms`.

### 6) Geographical Features (City/State/Lat/Lon)

* Custom `GeoCityStateImputer`:
    * **Missing `state`**: Estimate using `(cityname, lat, lon)`; if that fails, use the mode from `cityname`; fallback to the nearest neighbor by `(lat, lon)` or the global mode.
    * **Missing `cityname`**: After ensuring `state` is filled, find the nearest neighbor by `(lat, lon)` within the same state. Fallback to the state's mode or the global mode.
    * **Missing `latitude/longitude`**: Impute with the mean within the same `(state, cityname)` group, with a final fallback to the global mean.
* Construct `city_state = cityname + ', ' + state` and apply robust one-hot encoding. Latitude and longitude are passed through as **direct numerical features**.

### 7) Time Encoding

* Map the timestamp to three features: `sin(month)`, `cos(month)`, and `days_since_first`.
* **Why this encoding?** A simple month number (1-12) is linear, not cyclical, so the model wouldn't understand that December (12) is adjacent to January (1). By using sine/cosine transformations, we map the months onto a circle, preserving seasonal periodicity. Meanwhile, `days_since_first` acts as a linearly increasing feature, helping the model capture long-term trends in rental prices over time.

---

## 🧱 Feature Pipeline (`sklearn ColumnTransformer`)
* **Sub-pipelines**: `category_pipe`, `title_body_pipeline`, `amenities_pipe`, `pets_allowed_pipe`, `fee_photo_source_pipe`, `bed_bath_sqft_pipe`, `geo_pipe`, `time_pipe`.
* These are combined into a single `preprocess` transformer that feeds into the final model.

---

## 🤖 Model & Hyperparameter Tuning

* **Model**: `XGBRegressor`
    * **Objective**: `reg:absoluteerror` (optimizing directly for MAE)
    * **Training**: `tree_method='hist'`, `max_bin=128`, `min_child_weight=6.0`, `subsample=0.8`
    * **Regularization**: `reg_alpha=0.1`, `reg_lambda=5.0`
    * **Device**: Default `device='cuda'` (if no GPU is available, change to `device='cpu'` or remove the parameter)
* **Search Strategy**: `HalvingGridSearchCV` (5-fold, scoring with `neg_mean_absolute_error`)
    * **Grid**:
        * `max_depth ∈ {2, 3, 4}`
        * `n_estimators ∈ {2000, 2200, 2400, 2600, 2800}`
        * `colsample_bytree ∈ {0.25, 0.26, 0.27}`

---

## 📈 Experiment Results (on the provided `train.csv`/`test.csv`)

* **Best 5-Fold Cross-Validation**:
    * **Best Params**: `{max_depth=2, n_estimators=2800, colsample_bytree=0.26}`
    * **Best CV MAE ≈ 203.56**
* **Independent Test Set**:
    * **MAE ≈ 196.94**
    * **MAPE ≈ 12.10%**

> Note: These metrics are from the actual output of `notebooks/predict_house_price.ipynb`. Values may fluctuate slightly due to randomness and environmental differences (`RANDOM_STATE=42`).

---

## 🚀 Quick Start

### 1) Environment Setup

```bash
# Python ≥ 3.10 is recommended
python -m venv .venv && source .venv/bin/activate  # On Windows, use .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```
> Key Dependencies: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`.

### 2) Running the Experiment

* Place `train.csv` and `test.csv` in the repository's root directory or under the `data/` folder.
* Open and run the cells sequentially in `notebooks/predict_house_price.ipynb`.
* Upon completion, the script will output the optimal parameters, cross-validation (CV) results, and test set metrics. You can also export the predictions.

---

## 🧠 Design Choices & Retrospective

* **Explicit Missing Value Handling**: Null values in `amenities` and `pets_allowed` are explicitly encoded as `Nothing` or `No`. This ensures the model "sees" the absence or unknown state, which often outperforms treating them as simply missing features.
* **Constant Column Removal**: The `currency` and `price_type` columns contain only a single unique value. Removing them early on prevents useless dimensions and potential data leakage.
* **Address Column Removal**: The `address` feature is sparse and difficult to impute. It is also highly substitutable by `city`, `state`, `lat`, and `lon`. Keeping it would introduce noise and instability.
* **Preventing Target Leakage from Text**: Explicit price mentions in the text descriptions are masked to prevent the model from "cheating" by finding the answer directly.
* **Robust One-Hot Encoding**: For high-cardinality categorical features, infrequent categories are collapsed (e.g., using an `infrequent_if_exist` strategy) to control dimensional explosion and mitigate overfitting.
* **Geographical & Temporal Features**: The model captures both seasonal cycles and relative time. For geography, it combines discrete locations (`city_state`) with continuous coordinates (`lat`/`lon`).

---

## 🔭 Future Improvements

* **Geospatial Encoding**: Incorporate H3/Geohash indexing or external features based on Points of Interest (POI), school districts, or commute times.
* **Text Representation**: Replace or supplement TF-IDF with more advanced embeddings like `fastText`, `Sentence-BERT`, or a compact `Transformer` model.
* **Categorical Encoding**: For high-cardinality features, experiment with Target Encoding (with K-Fold cross-validation and regularization) or leverage CatBoost's native handling.
* **Model Ensembling**: Create a hybrid model by stacking or blending with `LightGBM`, `CatBoost`, or linear models.
* **Robustness Evaluation**: Implement more rigorous validation, such as time-series splits, inter-city transfer validation, and in-depth analysis of error distribution and outliers.
* **Business-Driven Constraints**: Apply monotonic constraints or piecewise calibration to ensure predictions are aligned with real-world business logic and avoid unreasonable outputs.
