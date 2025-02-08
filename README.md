# NDVI Analysis Project

## Overview
This project analyzes **Normalized Difference Vegetation Index (NDVI)** trends across Israel from **1982 to 2025** using satellite data from **MODIS, Sentinel, and Landsat**. NDVI is a key indicator of vegetation health, calculated from satellite imagery, where higher values indicate denser and healthier vegetation.  

We conducted three main experiments to examine:
1. **Long-term NDVI trends** across different climate zones.
2. **Feature importance analysis** to determine which environmental and climate factors most affect NDVI values.
3. **Timing analysis** to identify the factors influencing when NDVI starts rising and when the largest NDVI change occurs each year.  

By analyzing these patterns, we aim to understand how climate variability, temperature, and precipitation influence vegetation cycles and whether climate change is affecting seasonal plant growth.

---

## Dataset
All datasets used in this project are available on **Google Drive**.  

📂 **Google Drive Link:** `[INSERT_YOUR_GOOGLE_DRIVE_LINK_HERE]`  

To run the project successfully, download the datasets and place the two folders from Google Drive inside the **main directory** of the project.

---

## How to Run the Notebooks  
1. **Download the dataset** from Google Drive and place the folders in the project’s main directory.
2. Install required dependencies:
   ```sh
   pip install -r requirements.txt

---

## Experiment 1: Long-Term NDVI Trends  
📌 **Notebook:** `NVDI_analysis_across_years_1984-2025.ipynb`  

- This experiment examines **NDVI trends over the years** across different climate zones.
- It does not incorporate temperature or rainfall data, as those are only available from **2000 to 2022**.
- We also analyze the **timing of NDVI increase** and **the month with the biggest NDVI change** without considering climate factors.

---

## Experiment 2 & 3: Feature Importance & Timing Analysis  
📌 **Notebook:** `statistical_analysis_NVDI.ipynb`  

- In **Experiment 2**, we analyze how **various climate and environmental factors** influence NDVI values.
  - We use **XGBoost and Shapley values** to determine feature importance.
  - The key factors affecting NDVI include **temperature, rainfall, and vegetation type**.

- In **Experiment 3**, we investigate the **timing of NDVI changes** by identifying:
  - When NDVI **begins to rise** each year.
  - When the **largest NDVI change** occurs.
  - The climate and environmental features most responsible for these changes.

- The dataset used in this experiment contains **monthly averages** for temperature and cumulative rainfall from **August to January** to focus on the transition period into the growing season.

## Other Python Files in the Project  
The project also includes several additional Python scripts that provide functionalities for data extraction, processing, and model computation:

### `api.py`
This file contains **API calls** to the **Israel Meteorological Service** to fetch climate data.

### `compute_shap_values.py`
This script handles the computation of **Shapley values** for feature importance analysis. Since computing these values takes a long time, this script allows running it separately to avoid rerunning the entire analysis.

### `extract_data.py`
This script retrieves **NDVI and climate data** from satellite sources using **Google Earth Engine (GEE)**.

### `marge_data.py`
This file contains functionality for **merging multiple datasets** (NDVI, climate, and environmental data) into a unified dataset for analysis.

### `see_data.py`
A utility script for **exploring the datasets**, displaying basic information and statistics.

### `utils.py`
This file contains various helper functions used across the project, including:
- **Plotting functions** for visualizing NDVI trends.
- **Statistical calculations** for data analysis.
- **Machine learning model utilities** for training and evaluating models.
