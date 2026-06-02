```markdown
# 🛒 Walmart M5 Forecasting - High-Performance Big Data Pipeline

## 🎯 Project Overview
Predicting sales for 3,049 products across 10 stores in 3 US states (California, Texas, and Wisconsin) represents one of the most complex hierarchical time-series problems. This project implements a production-grade, highly optimized **Data Engineering & Feature Engineering pipeline** capable of processing over **57 Million rows** within limited cloud container memory (RAM).

The pipeline addresses extreme memory constraints through strategic downcasting and structural data transformation, preparing the dataset for high-performance gradient boosting models (LightGBM).

---

## 🛠️ Memory Optimization & Data Engineering Architecture
Processing massive datasets requires strict computing optimization. This pipeline integrates four core data engineering practices to prevent Out-Of-Memory (OOM) crashes:

1. **Vertical Serialization (`pd.melt`):** Pivots the wide daily sales tracking structure into a normalized long format, expanding the row count vertically to map continuous timeline histories cleanly.
2. **Strict Numerical Downcasting:** Optimizes data storage constraints by aggressively converting standard 64-bit types into lower bit-depths (`int16`, `float32`) and swapping heavy object strings for structural `pandas.category` blocks.
3. **Garbage Collection Isolation (`gc.collect()`):** Manually purges unlinked intermediate frames from the heap memory instantly after merging catalog layers (Prices & Calendar).
4. **Categorical Label Encoding:** Encodes complex textual identifiers (e.g., `event_name_1`, `item_id`) directly into compressed integer category codes (`cat.codes`).

---

## ⏱️ Feature Engineering & Safe Horizon Lags
To predict a forward-looking 28-day sequence without causing target data leakage, the historical features are offset by a protective temporal boundary:

* **Safe Historical Lags:** Shifts target horizons by a strict **28-day and 35-day window**, anchoring the feature baseline exactly on the edge of the blind prediction horizon.
* **Lagged Rolling Statistics:** Computes short-term (7-day) and medium-term (30-day) rolling means exclusively *on top* of the 28-day shifted lag matrix rather than active target histories.
* **Artifact Cleansing:** Drops the first 35 days of historical initialization timelines to eliminate empty rolling computation rows (`NaN`), feeding only high-fidelity signals to the training matrix.

---

## 📊 Pipeline Success & Final Metrics
Upon execution inside the cloud environment, the optimization pipeline yielded a highly structured training dataset:

* **Initial State:** Isolated horizontal relational tables.
* **Final Transformed Shape:** **`(57,290,710 rows, 18 columns)`**
* **Memory Status:** Stable and fully scalable within standard cloud container boundaries.

---

## 🚀 Optimized Data Ingestion Code Snippet
```python
# Downcasting structural identifiers and dates to prevent OOM errors
df['d'] = df['d'].apply(lambda x: int(x.split('_')[1])).astype(np.int16)
df['sales'] = df['sales'].astype(np.int16)

# Creating rolling metrics strictly on top of the 28-day safe lag window
df["sales_lag_28"] = df.groupby("id")["sales"].shift(28).astype(np.float32)
df["rolling_mean_7"] = df.groupby("id")["sales_lag_28"].transform(lambda x: x.rolling(7).mean()).astype(np.float32)

# Purging stale data references from active memory
del sales, calendar, prices
gc.collect()
