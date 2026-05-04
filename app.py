import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os
import requests
from src.data_loader import load_data, get_summary
from src.forecasting import forecast_yield
from src.insights import generate_insights, get_recommendations
from src.predict import predict_yield
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="AgriSense AI", page_icon="🌾", layout="wide")

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# Glassmorphism CSS
st.markdown("""
<style>
    .main { background: #0f172a; color: #e2e8f0; }
    .stMetric { background: rgba(30, 41, 59, 0.7); border-radius: 12px; padding: 15px; border: 1px solid rgba(255,255,255,0.1); }
    .insight-card { background: rgba(30, 41, 59, 0.5); padding: 20px; border-radius: 15px; border-left: 5px solid #4ade80; margin-bottom: 10px; }
    .stSidebar { background-color: #1e293b !important; }
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#e8f5e9'),
    margin=dict(l=20, r=20, t=40, b=20)
)

def safe_year_to_datetime(df, col='Year'):
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=[col])
    df[col] = df[col].astype('Int64')
    df['ds'] = pd.to_datetime(df[col], format='%Y', errors='coerce')
    return df.dropna(subset=['ds'])

def main():
    st.sidebar.title("🌾 AgriSense AI")
    page = st.sidebar.radio("Navigation", ['🏠 Dashboard', '🔮 Yield Prediction', '📉 Forecast', '🌦 Live Weather'])

    try:
        df = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    if page == '🏠 Dashboard':
        st.title("🏠 Agricultural Intelligence Dashboard")
        
        summary = get_summary(df)
        
        # KPIs
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🌾 Total Records", f"{summary['total_records']:,}")
        m2.metric("🏆 Best Crop", summary['best_crop'])
        m3.metric("📍 Top Region", summary['best_region'])
        m4.metric("📈 Avg Yield", f"{summary['avg_yield']:,} kg/ha")

        st.divider()

        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.subheader("📊 Yield Trends & Distribution")
            tab1, tab2 = st.tabs(["Yield by Crop", "Regional Performance"])
            
            with tab1:
                fig_crop = px.bar(df.groupby('Crop')['Yield_Per_Ha'].mean().reset_index(), 
                                 x='Crop', y='Yield_Per_Ha', color='Yield_Per_Ha',
                                 color_continuous_scale='Greens', template='plotly_dark')
                fig_crop.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig_crop, use_container_width=True)
            
            with tab2:
                fig_reg = px.pie(df.groupby('Region')['Production'].sum().reset_index(), 
                                values='Production', names='Region', hole=0.4,
                                template='plotly_dark')
                fig_reg.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig_reg, use_container_width=True)

        with c2:
            st.subheader("🧠 Automated Insights")
            insights = generate_insights(df)
            for ins in insights:
                st.markdown(f"""
                <div class='insight-card'>
                    <h4>{ins['icon']} {ins['title']}</h4>
                    <p style='font-size: 0.9rem; opacity: 0.8;'>{ins['body']}</p>
                </div>
                """, unsafe_allow_html=True)

        st.subheader("📋 Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

    elif page == '🔮 Yield Prediction':
        st.title("🔮 AI Yield Prediction")
        st.markdown("Enter farming parameters below to predict crop yield per hectare.")

        # Load resources
        try:
            model = joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))
            scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
            
            # Regenerate encoders
            encoders = {}
            for col in ['Crop', 'Region', 'Season']:
                le = LabelEncoder()
                le.fit(df[col].astype(str))
                encoders[col] = le
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return

        with st.form("prediction_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                crop = st.selectbox("Crop", sorted(df['Crop'].unique()))
                region = st.selectbox("Region", sorted(df['Region'].unique()))
                season = st.selectbox("Season", sorted(df['Season'].unique()))
            with c2:
                area = st.number_input("Area (Hectares)", min_value=1.0, value=100.0)
                rainfall = st.number_input("Annual Rainfall (mm)", min_value=0.0, value=1000.0)
                temp = st.number_input("Avg Temperature (°C)", min_value=0.0, value=25.0)
            with c3:
                fert = st.number_input("Fertilizer (kg)", min_value=0.0, value=200.0)
                pest = st.number_input("Pesticide (kg)", min_value=0.0, value=10.0)
                hum = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
            
            submit = st.form_submit_button("🚀 Predict Yield")

        if submit:
            input_data = {
                'crop': crop, 'region': region, 'season': season,
                'area': area, 'rainfall': rainfall, 'temperature': temp,
                'fertilizer': fert, 'pesticide': pest, 'humidity': hum
            }
            
            with st.spinner("Analyzing data..."):
                res = predict_yield(model, scaler, encoders, input_data)
            
            st.success("Analysis Complete!")
            
            r1, r2 = st.columns([1, 1])
            with r1:
                st.metric("📊 Predicted Yield", f"{res['predicted_yield']:,} kg/ha")
                st.metric("📦 Est. Total Production", f"{res['estimated_production']:,} kg")
            with r2:
                st.metric("🎯 Confidence Score", f"{res['confidence']}%")
                st.metric("⚡ Productivity Level", res['productivity_level'])

            st.divider()
            st.subheader("💡 Expert Recommendations")
            recs = get_recommendations(crop, region, rainfall, temp, fert, df)
            for r in recs:
                st.info(r)

    elif page == '📉 Forecast':
        st.markdown("<div class='section-header'>📉 Yield Forecasting Engine</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-sub'>Time-series yield forecasting for next 1–10 seasons using trend models</div>", unsafe_allow_html=True)

        crops = sorted(df['Crop'].unique().tolist())
        fc1, fc2 = st.columns([2,1])

        with fc1:
            sel_crop = st.selectbox('Select Crop', crops)
        with fc2:
            periods = st.slider('Forecast Periods (Years)', 1, 10, 5)

        # ✅ SAFE FORECAST CALL
        try:
            with st.spinner('Generating forecast...'):
                forecast_df = forecast_yield(df, sel_crop, periods)
        except Exception as e:
            st.error(f"Forecast failed: {e}")
            st.stop()

        # ✅ FIXED HISTORICAL DATA PROCESSING
        hist = df[df['Crop'] == sel_crop].groupby('Year')['Yield_Per_Ha'].mean().reset_index()

        # 🔥 MAIN FIX APPLIED HERE
        hist = safe_year_to_datetime(hist)

        hist.rename(columns={'Yield_Per_Ha': 'y'}, inplace=True)

        # ── Plot ─────────────────────────────────────────────
        fig_f = go.Figure()

        fig_f.add_trace(go.Scatter(
            x=hist['ds'], y=hist['y'],
            name='Historical',
            mode='lines+markers',
            line=dict(color='#4ade80', width=3),
            marker=dict(size=7, color='#86efac')
        ))

        fig_f.add_trace(go.Scatter(
            x=forecast_df['ds'], y=forecast_df['yhat'],
            name='Forecast',
            mode='lines+markers',
            line=dict(color='#38bdf8', width=2.5, dash='dot'),
            marker=dict(size=7, color='#38bdf8', symbol='diamond')
        ))

        fig_f.add_trace(go.Scatter(
            x=list(forecast_df['ds']) + list(forecast_df['ds'][::-1]),
            y=list(forecast_df['yhat_upper']) + list(forecast_df['yhat_lower'][::-1]),
            fill='toself',
            fillcolor='rgba(56,189,248,0.08)',
            line=dict(color='rgba(0,0,0,0)'),
            name='Confidence Band'
        ))

        fig_f.update_layout(
            title=f'📉 {sel_crop} Yield Forecast — Next {periods} Years',
            **PLOTLY_LAYOUT,
            height=420
        )

        st.plotly_chart(fig_f, width="stretch")

        # ── Forecast Table ───────────────────────────────────
        st.markdown("**📋 Forecast Data Table**")

        future_rows = forecast_df.tail(periods).copy()
        future_rows['Year'] = future_rows['ds'].dt.year

        future_rows = future_rows[['Year','yhat','yhat_lower','yhat_upper']].rename(columns={
            'yhat': 'Predicted Yield (kg/ha)',
            'yhat_lower': 'Lower Bound',
            'yhat_upper': 'Upper Bound'
        }).round(2)

        st.dataframe(
            future_rows.style.background_gradient(
                subset=['Predicted Yield (kg/ha)'],
                cmap='Greens'
            ),
            width="stretch"
        )

        st.download_button(
            '📥 Download Forecast CSV',
            data=future_rows.to_csv(index=False).encode(),
            file_name=f'{sel_crop}_forecast_{periods}yr.csv',
            mime='text/csv'
        )

    elif page == '🌦 Live Weather':
        st.title("🌦 Live Weather Intelligence")
        st.markdown("Monitor real-time weather conditions and their impact on agricultural yield.")
        
        api_key = st.text_input("OpenWeatherMap API Key", type="password", help="Free key from openweathermap.org")
        city = st.text_input("City Name", value="Ludhiana")
        
        if st.button("🔍 Analyze Weather"):
            if not api_key:
                st.warning("Please enter an API Key to fetch live data. Showing demonstration data below.")
                # Demo Data
                w_data = {"main": {"temp": 32, "humidity": 45}, "weather": [{"description": "clear sky"}], "wind": {"speed": 12}}
            else:
                try:
                    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
                    resp = requests.get(url).json()
                    if resp.get("cod") != 200:
                        st.error(f"Error: {resp.get('message', 'Unknown error')}")
                        return
                    w_data = resp
                except Exception as e:
                    st.error(f"Failed to connect: {e}")
                    return

            st.success(f"Weather data for {city} retrieved!")
            
            c1, c2, c3 = st.columns(3)
            temp = w_data['main']['temp']
            hum = w_data['main']['humidity']
            wind = w_data['wind']['speed']
            
            c1.metric("🌡️ Temperature", f"{temp}°C")
            c2.metric("💧 Humidity", f"{hum}%")
            c3.metric("💨 Wind Speed", f"{wind} m/s")
            
            st.divider()
            st.subheader("🌾 Agricultural Impact Assessment")
            
            if temp > 35:
                st.error("🔥 High Heat Stress: Shift irrigation to early morning/late evening. Monitor for leaf wilt.")
            elif temp < 15:
                st.warning("❄️ Low Temp Alert: Growth may slow for tropical crops. Check for frost risk.")
            else:
                st.success("✅ Optimal Temperature: Conditions are favorable for most local crops.")
                
            if hum > 80:
                st.warning("🌧️ High Humidity: Increased risk of fungal infections and pests. Ensure proper aeration.")
            elif hum < 30:
                st.info("🏜️ Low Humidity: Increased evaporation. Increase irrigation frequency.")

            st.info(f"Summary: Current conditions ({w_data['weather'][0]['description']}) are generally {'stable' if temp < 30 else 'challenging'} for high-yield farming in this region.")

if __name__ == "__main__":
    main()