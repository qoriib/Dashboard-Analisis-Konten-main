import os
import json
from pathlib import Path
import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet


st.set_page_config(
    page_title="Dashboard Analisis Konten",
    layout="wide"
)

st.title("Dashboard Analisis & Forecast Konten")
st.caption("Data otomatis dari Google Sheets")

st.sidebar.divider()
st.sidebar.subheader("Data")

if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()

TEMPLATE = "plotly_white"
WARNA_BIRU = "#4C78A8"

def get_service_account_credentials():
    env_map = {
        "type": os.getenv("GCP_SA_TYPE"),
        "project_id": os.getenv("GCP_SA_PROJECT_ID"),
        "private_key_id": os.getenv("GCP_SA_PRIVATE_KEY_ID"),
        "private_key": os.getenv("GCP_SA_PRIVATE_KEY"),
        "client_email": os.getenv("GCP_SA_CLIENT_EMAIL"),
        "client_id": os.getenv("GCP_SA_CLIENT_ID"),
        "auth_uri": os.getenv("GCP_SA_AUTH_URI"),
        "token_uri": os.getenv("GCP_SA_TOKEN_URI"),
        "auth_provider_x509_cert_url": os.getenv("GCP_SA_AUTH_PROVIDER_X509_CERT_URL"),
        "client_x509_cert_url": os.getenv("GCP_SA_CLIENT_X509_CERT_URL"),
        "universe_domain": os.getenv("GCP_SA_UNIVERSE_DOMAIN"),
    }

    required_keys = [
        "type",
        "project_id",
        "private_key_id",
        "private_key",
        "client_email",
        "client_id",
        "auth_uri",
        "token_uri",
        "auth_provider_x509_cert_url",
        "client_x509_cert_url",
    ]

    if all(env_map[k] for k in required_keys):
        # Replace escaped newlines so the key is usable
        env_map["private_key"] = env_map["private_key"].replace("\\n", "\n")
        return env_map

    try:
        return dict(st.secrets["gcp_service_account"])
    except Exception:
        st.error(
            "Credential service account tidak ditemukan. "
            "Set env GCP_SERVICE_ACCOUNT_JSON atau set GCP_SA_* di .env / environment, "
            "atau tambahkan gcp_service_account ke st.secrets."
        )
        st.stop()


@st.cache_data(ttl=300)
def load_data():
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]

    credentials = get_service_account_credentials()
    creds = ServiceAccountCredentials.from_json_keyfile_dict(credentials, scopes=scope)

    
    client = gspread.authorize(creds)
    sheet = client.open("Data Konten Awal").worksheet("Sheet1")
    data = sheet.get_all_values()
    return pd.DataFrame(data[1:], columns=data[0])

df = load_data()


df["Tanggal"] = df["Tanggal"].astype(str).str.strip().replace(["", " "], pd.NA).ffill()

bulan_map = {
    "Januari": "January", "Februari": "February", "Maret": "March",
    "April": "April", "Mei": "May", "Juni": "June",
    "Juli": "July", "Agustus": "August", "September": "September",
    "Oktober": "October", "November": "November", "Desember": "December"
}

def convert_bulan(x):
    for i, e in bulan_map.items():
        x = x.replace(i, e)
    return x

df["Tanggal"] = df["Tanggal"].apply(convert_bulan)
df["Tanggal"] = pd.to_datetime(df["Tanggal"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["Tanggal"])

df["Tahun"] = df["Tanggal"].dt.year
df["Nama_Bulan"] = df["Tanggal"].dt.month_name()

df["Konten Kanwil/Kanca"] = (
    df["Konten Kanwil/Kanca"]
    .astype(str)
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

df["Konten Kanwil/Kanca"] = df["Konten Kanwil/Kanca"].replace({
    "Kanpus": "Pusat",
    "Jatim": "Kanwil",
    "kanwil": "Kanwil"
})

df.columns = df.columns.str.replace(" ", "_").str.replace("/", "_")

total_konten = len(df)
total_kanwil = df["Konten_Kanwil_Kanca"].nunique()

konten_bulanan = df.groupby("Nama_Bulan").size()
avg_konten = round(konten_bulanan.mean(), 1)

top_bulan = konten_bulanan.idxmax()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Konten", total_konten)
c2.metric("Kanwil/Kanca Aktif", total_kanwil)
c3.metric("Rata-rata / Bulan", avg_konten)
c4.metric("Bulan Terproduktif", top_bulan)

st.divider()


tab1, tab2, tab3 = st.tabs(["📌 Overview", "📊 Analisis Detail", "🔮 Forecast"])


with tab1:
    st.subheader("Overview Jumlah Konten per Bulan")

    df_overview = (
        df.groupby("Nama_Bulan")
        .size()
        .reset_index(name="Jumlah Konten")
    )

    fig = px.bar(
        df_overview,
        x="Nama_Bulan",
        y="Jumlah Konten",
        text="Jumlah Konten",
        color_discrete_sequence=[WARNA_BIRU]
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(template=TEMPLATE)

    st.plotly_chart(fig, use_container_width=True, key="overview")


with tab2:
    st.sidebar.header("Filter Analisis")

    tahun = st.sidebar.selectbox(
        "Pilih Tahun",
        ["Semua"] + sorted(df["Tahun"].unique().tolist())
    )

    kanwil = st.sidebar.selectbox(
        "Pilih Kanwil / Kanca",
        ["Semua"] + sorted(df["Konten_Kanwil_Kanca"].unique().tolist())
    )

    df_filter = df.copy()
    if tahun != "Semua":
        df_filter = df_filter[df_filter["Tahun"] == tahun]
    if kanwil != "Semua":
        df_filter = df_filter[df_filter["Konten_Kanwil_Kanca"] == kanwil]

    st.subheader("Jumlah Konten per Bulan")

    df_bulan = (
        df_filter
        .groupby("Nama_Bulan")
        .size()
        .reset_index(name="Jumlah Konten")
    )

    fig1 = px.bar(
        df_bulan,
        x="Nama_Bulan",
        y="Jumlah Konten",
        text="Jumlah Konten",
        color_discrete_sequence=[WARNA_BIRU]
    )
    fig1.update_traces(textposition="outside")
    fig1.update_layout(template=TEMPLATE)

    st.plotly_chart(fig1, use_container_width=True, key="detail_bulanan")

    st.divider()


    st.subheader("Perbandingan Konten per Bulan (Antar Tahun)")

    df_compare = (
        df
        .groupby(["Tahun", "Nama_Bulan"])
        .size()
        .reset_index(name="Jumlah Konten")
    )

    fig_compare = px.bar(
        df_compare,
        x="Nama_Bulan",
        y="Jumlah Konten",
        color="Tahun",
        barmode="group",
        text="Jumlah Konten",
        title=None
    )

    fig_compare.update_traces(textposition="outside")
    fig_compare.update_layout(template=TEMPLATE)

    st.plotly_chart(fig_compare, use_container_width=True, key="compare_tahun")

    st.divider()

    st.subheader("Jumlah Konten per Kanwil / Kanca")

    df_kanwil = (
        df_filter
        .groupby("Konten_Kanwil_Kanca")
        .size()
        .reset_index(name="Jumlah Konten")
    )

    fig2 = px.bar(
        df_kanwil,
        x="Konten_Kanwil_Kanca",
        y="Jumlah Konten",
        text="Jumlah Konten",
        color_discrete_sequence=[WARNA_BIRU]
    )

    fig2.update_layout(template=TEMPLATE, xaxis_tickangle=45)
    st.plotly_chart(fig2, use_container_width=True, key="kanwil_chart")

    st.divider()


    st.subheader("Top 5 Kanca Paling Produktif")

    df_top5 = (
        df_filter[~df_filter["Konten_Kanwil_Kanca"].isin(["Kanwil", "Pusat"])]
        .groupby("Konten_Kanwil_Kanca")
        .size()
        .nlargest(5)
        .reset_index(name="Jumlah Konten")
    )

    fig_top5 = px.bar(
        df_top5,
        x="Jumlah Konten",
        y="Konten_Kanwil_Kanca",
        orientation="h",
        text="Jumlah Konten",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    fig_top5.update_layout(
        template=TEMPLATE,
        yaxis={"categoryorder": "total ascending"}
    )

    st.plotly_chart(fig_top5, use_container_width=True, key="top5_kanca")

    st.divider()

    st.subheader("Distribusi Platform Konten")

    platforms = ["IG", "Tiktok", "FB", "X", "Youtube"]
    platform_counts = {p: df_filter[p].notna().sum() for p in platforms}

    df_platform = pd.DataFrame({
        "Platform": platform_counts.keys(),
        "Jumlah Konten": platform_counts.values()
    })

    fig_platform = px.pie(
        df_platform,
        names="Platform",
        values="Jumlah Konten",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3
    )

    fig_platform.update_layout(template=TEMPLATE)

    st.plotly_chart(fig_platform, use_container_width=True, key="platform_pie")


with tab3:
    st.subheader("Forecast Jumlah Konten Bulanan")

    df_monthly = (
        df.groupby(pd.Grouper(key="Tanggal", freq="M"))
        .size()
        .reset_index(name="y")
        .rename(columns={"Tanggal": "ds"})
    )

    df_monthly = df_monthly[df_monthly["y"] > 0]

    if len(df_monthly) >= 6:
        model = Prophet(yearly_seasonality=True)
        model.fit(df_monthly)

        future = model.make_future_dataframe(periods=6, freq="M")
        forecast = model.predict(future)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_monthly["ds"], y=df_monthly["y"],
            mode="lines+markers", name="Aktual"
        ))

        fig.add_trace(go.Scatter(
            x=forecast["ds"], y=forecast["yhat"],
            mode="lines", name="Forecast", line=dict(dash="dash")
        ))

        fig.add_trace(go.Scatter(
            x=forecast["ds"], y=forecast["yhat_upper"],
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=forecast["ds"], y=forecast["yhat_lower"],
            fill="tonexty", opacity=0.2, name="Confidence Interval"
        ))

        fig.update_layout(template=TEMPLATE)
        st.plotly_chart(fig, use_container_width=True, key="forecast")

    else:
        st.warning("Data tidak cukup untuk forecast (minimal 6 bulan)")
