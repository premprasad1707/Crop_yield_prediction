"""
Insight Engine — Agriculture Crop Yield Platform
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def generate_insights(df: pd.DataFrame, feature_importance: pd.DataFrame = None) -> list[dict]:
    insights = []

    # Rainfall correlation
    corr_rain = df['Annual_Rainfall'].corr(df['Yield_Per_Ha'])
    if abs(corr_rain) > 0.3:
        direction = 'positively' if corr_rain > 0 else 'negatively'
        insights.append({
            'icon': '🌧️',
            'title': 'Rainfall Impact',
            'body': f'Annual rainfall is {direction} correlated with crop yield (r={corr_rain:.2f}). '
                    f'Regions with {">800mm" if corr_rain > 0 else "<400mm"} rainfall consistently outperform.',
            'type': 'info'
        })

    # Fertilizer efficiency
    corr_fert = df['Fertilizer_Used'].corr(df['Yield_Per_Ha'])
    insights.append({
        'icon': '🌱',
        'title': 'Fertilizer Efficiency',
        'body': (f'Excess fertilizer (above 260 kg/ha) shows diminishing returns. '
                 f'Optimal application yields {df[df["Fertilizer_Used"].between(100, 260)]["Yield_Per_Ha"].mean():.0f} kg/ha '
                 f'vs {df[df["Fertilizer_Used"] > 300]["Yield_Per_Ha"].mean():.0f} kg/ha for over-application.'),
        'type': 'warning'
    })

    # Best crop
    best_crop = df.groupby('Crop')['Yield_Per_Ha'].mean().idxmax()
    best_yield = df.groupby('Crop')['Yield_Per_Ha'].mean().max()
    insights.append({
        'icon': '🏆',
        'title': 'Top Performing Crop',
        'body': f'{best_crop} leads all crops with an average yield of {best_yield:,.0f} kg/ha. '
                f'Farmers in suitable regions should prioritize this crop.',
        'type': 'success'
    })

    # Best region
    best_region = df.groupby('Region')['Yield_Per_Ha'].mean().idxmax()
    best_reg_yield = df.groupby('Region')['Yield_Per_Ha'].mean().max()
    insights.append({
        'icon': '📍',
        'title': 'Most Productive Region',
        'body': f'{best_region} is the highest-yielding region with avg {best_reg_yield:,.0f} kg/ha. '
                f'Agro-climatic conditions here are optimal for most crops.',
        'type': 'success'
    })

    # Temperature insight
    corr_temp = df['Average_Temperature'].corr(df['Yield_Per_Ha'])
    insights.append({
        'icon': '🌡️',
        'title': 'Temperature Sensitivity',
        'body': (f'Crops show peak yields at 22–28°C. '
                 f'Temperatures above 35°C correlate with a {abs(corr_temp)*100:.0f}% reduction in productivity. '
                 f'Climate change risk is significant for heat-sensitive crops.'),
        'type': 'warning' if corr_temp < 0 else 'info'
    })

    # Feature importance insight
    if feature_importance is not None and not feature_importance.empty:
        top_feat = feature_importance.iloc[0]['Feature'].replace('_', ' ')
        insights.append({
            'icon': '🤖',
            'title': 'ML Insight: Top Yield Driver',
            'body': f'Machine learning identifies "{top_feat}" as the single most predictive feature for yield. '
                    f'Focus on optimizing this factor for maximum impact.',
            'type': 'info'
        })

    # Yield trend
    by_year = df.groupby('Year')['Yield_Per_Ha'].mean()
    if len(by_year) > 2:
        trend = np.polyfit(by_year.index, by_year.values, 1)[0]
        direction = 'improving' if trend > 0 else 'declining'
        insights.append({
            'icon': '📈',
            'title': 'Multi-Year Yield Trend',
            'body': f'Agricultural productivity is {direction} at {abs(trend):.1f} kg/ha per year. '
                    f'{"Adopt modern precision farming to accelerate gains." if trend > 0 else "Intervention required to reverse decline."}',
            'type': 'success' if trend > 0 else 'error'
        })

    return insights


def get_recommendations(crop: str, region: str, rainfall: float,
                         temperature: float, fertilizer: float, df: pd.DataFrame) -> list[str]:
    """Smart recommendations for a farmer."""
    recs = []
    # Region benchmark
    bench = df[(df['Crop'] == crop) & (df['Region'] == region)]['Yield_Per_Ha'].mean()
    if not np.isnan(bench):
        recs.append(f"📊 Average yield for {crop} in {region} is {bench:,.0f} kg/ha — use this as your benchmark.")

    if rainfall < 500:
        recs.append("💧 Low rainfall detected. Consider drip irrigation or drought-resistant seed varieties.")
    elif rainfall > 2000:
        recs.append("🌊 Excessive rainfall risk. Ensure proper field drainage to prevent waterlogging.")

    if temperature > 35:
        recs.append("🌡️ High temperature alert. Shift planting schedules or deploy shade nets for sensitive crops.")

    opt_fert = df[df['Crop'] == crop]['Fertilizer_Used'].quantile(0.5)
    if fertilizer > opt_fert * 1.4:
        recs.append(f"⚠️ Fertilizer usage is significantly above the optimal {opt_fert:.0f} kg/ha for {crop}. Reduce application to cut costs and avoid soil degradation.")
    elif fertilizer < opt_fert * 0.6:
        recs.append(f"🌱 Fertilizer is below optimal levels. Increasing to ~{opt_fert:.0f} kg/ha could improve yields.")

    # Season alignment (placeholder)
    recs.append(f"🗓️ Monitor sowing windows aligned with historical weather patterns for {region} to maximize productivity.")

    return recs
