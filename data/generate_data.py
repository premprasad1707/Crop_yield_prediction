"""
Agriculture Crop Yield Dataset Generator
Mimics the Kaggle Agriculture Crop Yield dataset structure
"""
import numpy as np
import pandas as pd
import os

np.random.seed(42)

CROPS = ['Rice', 'Wheat', 'Maize', 'Sugarcane', 'Cotton', 'Soybean',
         'Barley', 'Sorghum', 'Millet', 'Groundnut', 'Sunflower', 'Potato']

REGIONS = ['Punjab', 'Haryana', 'Uttar Pradesh', 'Maharashtra', 'Karnataka',
           'Andhra Pradesh', 'Tamil Nadu', 'West Bengal', 'Rajasthan', 'Gujarat',
           'Madhya Pradesh', 'Bihar']

SEASONS = ['Kharif', 'Rabi', 'Zaid', 'Whole Year']

CROP_BASE_YIELD = {
    'Rice': 2500, 'Wheat': 3000, 'Maize': 2800, 'Sugarcane': 70000,
    'Cotton': 500, 'Soybean': 1200, 'Barley': 2200, 'Sorghum': 1000,
    'Millet': 800, 'Groundnut': 1500, 'Sunflower': 1100, 'Potato': 20000
}

n = 5000
years = np.random.randint(2005, 2024, n)
crops = np.random.choice(CROPS, n)
regions = np.random.choice(REGIONS, n)
seasons = np.random.choice(SEASONS, n)
area = np.random.uniform(1, 500, n).round(2)
rainfall = np.random.uniform(300, 2500, n).round(1)
temperature = np.random.uniform(15, 42, n).round(1)
fertilizer = np.random.uniform(50, 400, n).round(1)
pesticide = np.random.uniform(0.5, 20, n).round(2)
humidity = np.random.uniform(30, 95, n).round(1)

production = np.array([
    CROP_BASE_YIELD[c] * a * (
        1 + 0.0003 * r - 0.01 * abs(t - 25)
        + 0.002 * f - 0.00001 * f**2
        - 0.02 * p + np.random.normal(0, 0.15)
    )
    for c, a, r, t, f, p in zip(crops, area, rainfall, temperature, fertilizer, pesticide)
])
production = np.abs(production).round(2)
yield_per_ha = (production / area).round(2)

df = pd.DataFrame({
    'Year': years,
    'Crop': crops,
    'Region': regions,
    'Season': seasons,
    'Area': area,
    'Annual_Rainfall': rainfall,
    'Average_Temperature': temperature,
    'Fertilizer_Used': fertilizer,
    'Pesticide_Used': pesticide,
    'Humidity': humidity,
    'Production': production,
    'Yield_Per_Ha': yield_per_ha
})

out = os.path.join(os.path.dirname(__file__), 'crop_yield.csv')
df.to_csv(out, index=False)
print(f"Dataset generated: {out} ({len(df)} rows)")
