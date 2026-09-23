import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
import plotly.express as px
import folium
from streamlit_folium import st_folium
from datetime import datetime

from config import DB_PATH, REGION_MAPPING, CITY_COORDINATES, REGION_CENTER_COORDINATES
from db_manager import (
    get_distinct_regions, query_regional_forecasts, query_city_forecasts
)
from fetch_cwa_data import fetch_and_store_weather_data

# Page Configuration
st.set_page_config(
    page_title="Taiwan Weather Forecast Dashboard",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics and Google Maps style integration
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main-title {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #6c757d;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #2a5298 0%, #1e3c72 100%);
        color: white;
        border-radius: 8px;
        font-weight: 600;
        border: none;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Main Header & Banner
st.markdown('<div class="main-title">🌤️ 臺灣天氣預報儀表板 (Taiwan Weather Forecast)</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AIoT L3 CWA Open Data 專案 | 中央氣象署 Weather Data Integration & Visualization</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://opendata.cwa.gov.tw/assets/img/logo.png", width=180)
st.sidebar.title("🎛️ 預報控制台")

# Refresh Button
if st.sidebar.button("🔄 重新向 CWA API 抓取資料"):
    with st.spinner("正在向中央氣象署 CWA API 獲取最新資料..."):
        success = fetch_and_store_weather_data()
        if success:
            st.sidebar.success("資料更新成功！")
            st.rerun()
        else:
            st.sidebar.error("資料更新失敗，請檢查 API 金鑰。")

# Data loading function
@st.cache_data(ttl=3600)
def load_data():
    df_regional = query_regional_forecasts()
    df_city = query_city_forecasts()
    return df_regional, df_city

df_regional, df_city = load_data()

if df_regional.empty and df_city.empty:
    st.warning("⚠️ 資料庫中尚無氣象資料。點擊左側「重新向 CWA API 抓取資料」即可初始化！")
    if st.button("🚀 立即初始化資料庫"):
        fetch_and_store_weather_data()
        st.rerun()
    st.stop()

# Region Selection
available_regions = ["全部地區"] + list(REGION_MAPPING.keys())
selected_region = st.sidebar.selectbox(
    "📍 選擇地區 (Select Region):",
    options=available_regions,
    index=2  # Default to 中部地區
)

# City Filter within selected region
if selected_region != "全部地區":
    available_cities = ["全部縣市"] + REGION_MAPPING.get(selected_region, [])
else:
    available_cities = ["全部縣市"] + list(CITY_COORDINATES.keys())

selected_city = st.sidebar.selectbox("🏙️ 選擇縣市 (Select City):", options=available_cities)

# Date Filter for Map
available_dates = sorted(list(df_city["dataDate"].unique())) if not df_city.empty else []
selected_date = st.sidebar.selectbox("📅 選擇地圖日期 (Select Date):", options=available_dates, index=0 if available_dates else None)

# Map Style Selector (Google Maps Options)
map_style = st.sidebar.radio(
    "🗺️ 地圖樣式 (Map Style):",
    options=["Google 地圖 (標準)", "Google 衛星混合圖 (Satellite)", "Google 地形圖 (Terrain)"],
    index=0
)

# App Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 氣溫折線圖與表格", "🗺️ 台灣天氣地圖 (Google Maps)", "💡 AI 穿搭與生活建議", "📚 專案架構與重點"])

with tab1:
    st.subheader(f"📌 {selected_region} - {selected_city} 氣溫預報 trend")

    if selected_city != "全部縣市":
        df_view = df_city[df_city["cityName"] == selected_city].sort_values("dataDate")
    elif selected_region != "全部地區":
        df_view = df_regional[df_regional["regionName"] == selected_region].sort_values("dataDate")
    else:
        df_view = df_regional.sort_values("dataDate")

    if not df_view.empty:
        latest_row = df_view.iloc[0]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🔥 最高氣溫 MaxT", f"{latest_row['maxt']} °C")
        with col2:
            st.metric("❄️ 最低氣溫 MinT", f"{latest_row['mint']} °C")
        with col3:
            avg_temp = round((latest_row['maxt'] + latest_row['mint']) / 2, 1)
            st.metric("🌡️ 平均氣溫 AvgT", f"{avg_temp} °C")
        with col4:
            st.metric("☁️ 天氣現象 Wx", f"{latest_row.get('wx', '多雲')}")

        st.divider()

        st.markdown("### 📈 一週最高與最低氣溫 (MaxT vs MinT)")
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_view["dataDate"],
            y=df_view["maxt"],
            mode="lines+markers",
            name="最高氣溫 (MaxT)",
            line=dict(color="#FF4B4B", width=3),
            marker=dict(size=8)
        ))

        fig.add_trace(go.Scatter(
            x=df_view["dataDate"],
            y=df_view["mint"],
            mode="lines+markers",
            name="最低氣溫 (MinT)",
            line=dict(color="#1C83E5", width=3),
            marker=dict(size=8)
        ))

        fig.update_layout(
            title="一週氣溫變化趨勢圖",
            xaxis_title="日期 (Date)",
            yaxis_title="氣溫 (°C)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white",
            height=420
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📋 清楚呈現預報資料表格")
        display_df = df_view[["dataDate", "mint", "maxt", "wx"]].rename(columns={
            "dataDate": "日期 (Date)",
            "mint": "最低氣溫 (°C)",
            "maxt": "最高氣溫 (°C)",
            "wx": "天氣現象"
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("🗺️ 台灣天氣地圖 (Google Maps Style)")
    st.caption("結合 **Google Maps 底圖** 與全台縣市即時預報標籤。標籤顏色代表平均溫度：🔵<20°C | 🟢 20-25°C | 🟡 25-30°C | 🔴 >30°C")

    if selected_date and not df_city.empty:
        df_map = df_city[df_city["dataDate"] == selected_date]

        # Determine center coordinates
        if selected_region != "全部地區" and selected_region in REGION_CENTER_COORDINATES:
            center_lat, center_lon = REGION_CENTER_COORDINATES[selected_region]
            zoom_level = 9
        else:
            center_lat, center_lon = 23.8, 120.9
            zoom_level = 7.5

        # Configure Google Maps Tile URLs
        if map_style == "Google 衛星混合圖 (Satellite)":
            tile_url = "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}"
            tile_attr = "Google Maps Satellite"
        elif map_style == "Google 地形圖 (Terrain)":
            tile_url = "https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}"
            tile_attr = "Google Maps Terrain"
        else: # Google Standard Roadmap
            tile_url = "https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}"
            tile_attr = "Google Maps Roadmap"

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_level,
            tiles=tile_url,
            attr=tile_attr
        )

        for _, row in df_map.iterrows():
            avg_temp = (row["maxt"] + row["mint"]) / 2.0
            
            # Color coding
            if avg_temp < 20:
                bg_color = "#1E88E5" # Blue
            elif 20 <= avg_temp < 25:
                bg_color = "#43A047" # Green
            elif 25 <= avg_temp < 30:
                bg_color = "#FB8C00" # Orange
            else:
                bg_color = "#E53935" # Red

            popup_html = f"""
            <div style="font-family: 'Inter', sans-serif; width: 170px; padding: 4px;">
                <h4 style="margin:0 0 6px 0; color:#1a73e8; border-bottom:1px solid #eee; padding-bottom:4px;">{row['cityName']}</h4>
                <b>日期:</b> {row['dataDate']}<br>
                <b>最高溫:</b> <span style="color:#d93025; font-weight:bold;">{row['maxt']}°C</span><br>
                <b>最低溫:</b> <span style="color:#1a73e8; font-weight:bold;">{row['mint']}°C</span><br>
                <b>天氣狀況:</b> {row.get('wx', '多雲')}
            </div>
            """

            # Google Maps Styled Pill Badge Marker
            icon_html = f"""
            <div style="
                background-color: {bg_color};
                color: white;
                padding: 4px 10px;
                border-radius: 16px;
                font-size: 12px;
                font-weight: 700;
                border: 2px solid white;
                box-shadow: 0 3px 8px rgba(0,0,0,0.35);
                white-space: nowrap;
                text-align: center;
                font-family: 'Inter', sans-serif;
            ">
                📍 {row['cityName']} | {row['mint']}~{row['maxt']}°C
            </div>
            """

            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                icon=folium.DivIcon(
                    html=icon_html,
                    icon_size=(110, 32),
                    icon_anchor=(55, 16)
                ),
                popup=folium.Popup(popup_html, max_width=220),
                tooltip=f"{row['cityName']}: {row['mint']}°C ~ {row['maxt']}°C ({row.get('wx', '')})"
            ).add_to(m)

        st_folium(m, width="100%", height=550)

with tab3:
    st.subheader("💡 AI 氣象分析與穿搭生活建議 (Poster Step 22)")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("🧥 **穿搭與防曬建議**")
        st.markdown("""
        - **高溫特報 (>30°C)**: 建議穿著透氣排汗衣物，外出請備妥遮陽帽與太陽眼鏡，並隨時補充水分。
        - **舒適溫度 (20~25°C)**: 適合穿著薄長袖或短袖搭配薄外套，體感極為舒適。
        - **涼爽低溫 (<20°C)**: 早晚溫差較大，建議攜帶風衣或保暖外套以免受涼。
        """)

    with col2:
        st.success("🤖 **智能旅遊與農情推播**")
        st.markdown("""
        - **旅遊行程建議**: 若預報顯示多雲到晴，適合戶外踏青與踏浪；若有局部陣雨，建議安排室內展館行程。
        - **農業防災推播**: 針對中南部蔬果種植區，密切關注降雨概率與最高溫變化，做好防溫差與防雨措施。
        """)

with tab4:
    st.subheader("📚 24 步驟 AIoT Weather App 開發架構全覽")
    st.markdown("""
    | 步驟編號 | 階段名稱 | 實作內容與模組 |
    | :--- | :--- | :--- |
    | **Step 1 - 3** | 平台與金鑰 | 註冊 CWA Open Data 平台並設定 API Key `CWA-009C2096...` |
    | **Step 4 - 6** | API 與 JSON 解析 | 使用 `requests` 抓取 JSON，解析縣市與地區之 `MinT`/`MaxT` |
    | **Step 7 - 10**| SQLite 資料庫設計 | 建立 `data.db`，設計 `TemperatureForecasts` 表並使用 SQL 查詢驗證 |
    | **Step 11 - 16**| Streamlit Web App | 整合互動下拉選單、Plotly 折線圖與自訂 Dataframe 表格 |
    | **Step 17 - 18**| 台灣地圖視覺化 | 結合 **Google Maps 底圖** 與日期篩選，呈現 Google 風格溫度地標 |
    | **Step 19 - 21**| 程式碼品質與 Git | 模組化架構 (`config`, `fetch_cwa_data`, `db_manager`, `app`) 並託管於 GitHub |
    """)

st.markdown("---")
st.caption("Powered by CWA Open Data API & Streamlit | AIoT L3 HW1 專案開發")
