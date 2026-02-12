# =========================
# IMPORTS
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_ind, levene

# =========================
# CONFIGURACIÓN GENERAL
# =========================
st.set_page_config(
    page_title="Reporte Telecomunicaciones",
    page_icon="📡",
    layout="wide"
)

sns.set_style("darkgrid")

# =========================
# CARGA Y LIMPIEZA DE DATOS
# =========================
@st.cache_data
def load_data():
    df_calls = pd.read_csv("datasets/megaline_calls.csv")
    df_internet = pd.read_csv("datasets/megaline_internet.csv")
    df_messages = pd.read_csv("datasets/megaline_messages.csv")
    df_plans = pd.read_csv("datasets/megaline_plans.csv")
    df_users = pd.read_csv("datasets/megaline_users.csv")

    # Normalización
    df_plans['plan_name'] = df_plans['plan_name'].str.capitalize()
    df_users['plan'] = df_users['plan'].str.capitalize()
    df_users['churn_date'] = df_users['churn_date'].fillna('Usuario activo')

    # Tipos
    df_calls['call_date'] = pd.to_datetime(df_calls['call_date'])
    df_messages['message_date'] = pd.to_datetime(df_messages['message_date'])
    df_internet['session_date'] = pd.to_datetime(df_internet['session_date'])
    df_users['reg_date'] = pd.to_datetime(df_users['reg_date'])

    # MB → GB
    df_internet['mb_used'] = df_internet['mb_used'] / 1024

    return df_calls, df_messages, df_internet, df_plans, df_users


# =========================
# AGREGACIÓN MENSUAL
# =========================
@st.cache_data
def build_monthly_usage(df_calls, df_messages, df_internet):
    df_calls['year_month'] = df_calls['call_date'].dt.to_period('M')
    df_calls['duration'] = np.ceil(df_calls['duration'])

    calls = df_calls.groupby(['user_id', 'year_month']).agg(
        calls_count=('duration', 'size'),
        minutes_used=('duration', 'sum')
    ).reset_index()

    df_messages['year_month'] = df_messages['message_date'].dt.to_period('M')
    messages = df_messages.groupby(['user_id', 'year_month']).size().reset_index(
        name='messages_count'
    )

    df_internet['year_month'] = df_internet['session_date'].dt.to_period('M')
    internet = df_internet.groupby(['user_id', 'year_month'])['mb_used'].sum().reset_index(
        name='vol_count'
    )

    df = calls.merge(messages, how='outer')
    df = df.merge(internet, how='outer')
    df.fillna(0, inplace=True)

    return df


# =========================
# CÁLCULO DE INGRESOS
# =========================
def calculate_revenue(df, df_users, df_plans):
    df = df.merge(df_users, on='user_id', how='inner')
    plans = df_plans.set_index('plan_name')

    def revenue(row):
        plan = row['plan']
        r = plans.loc[plan, 'usd_monthly_pay']

        if row['minutes_used'] > plans.loc[plan, 'minutes_included']:
            r += (row['minutes_used'] - plans.loc[plan, 'minutes_included']) * plans.loc[plan, 'usd_per_minute']

        if row['messages_count'] > plans.loc[plan, 'messages_included']:
            r += (row['messages_count'] - plans.loc[plan, 'messages_included']) * plans.loc[plan, 'usd_per_message']

        if row['vol_count'] > plans.loc[plan, 'mb_per_month_included'] / 1024:
            r += np.ceil(
                row['vol_count'] - plans.loc[plan, 'mb_per_month_included'] / 1024
            ) * plans.loc[plan, 'usd_per_gb']

        return r

    df['monthly_revenue'] = df.apply(revenue, axis=1)
    return df


# =========================
# CARGA DATA
# =========================
df_calls, df_messages, df_internet, df_plans, df_users = load_data()
df_usage = build_monthly_usage(df_calls, df_messages, df_internet)
df_final = calculate_revenue(df_usage, df_users, df_plans)

# =========================
# SIDEBAR – FILTROS
# =========================
st.sidebar.title("🎛️ Filtros de Análisis")

plans = ['Todos'] + sorted(df_final['plan'].unique().tolist())
selected_plan = st.sidebar.selectbox("Selecciona el plan", plans)

user_status = st.sidebar.radio(
    "Estado del usuario",
    ["Todos", "Activos", "Churn"]
)

min_rev, max_rev = st.sidebar.slider(
    "Rango de ingresos mensuales ($)",
    float(df_final['monthly_revenue'].min()),
    float(df_final['monthly_revenue'].max()),
    (
        float(df_final['monthly_revenue'].min()),
        float(df_final['monthly_revenue'].max())
    )
)

# =========================
# APLICAR FILTROS
# =========================
df_filtered = df_final.copy()

if selected_plan != "Todos":
    df_filtered = df_filtered[df_filtered['plan'] == selected_plan]

if user_status == "Activos":
    df_filtered = df_filtered[df_filtered['churn_date'] == 'Usuario activo']
elif user_status == "Churn":
    df_filtered = df_filtered[df_filtered['churn_date'] != 'Usuario activo']

df_filtered = df_filtered[
    (df_filtered['monthly_revenue'] >= min_rev) &
    (df_filtered['monthly_revenue'] <= max_rev)
]

# =========================
# UI
# =========================
st.title("📡 Reporte de Investigación – Telecomunicaciones")
st.markdown("Análisis exploratorio, ingresos y pruebas estadísticas")

# =========================
# KPIs
# =========================
st.subheader("📊 KPIs Clave")

c1, c2, c3 = st.columns(3)
c1.metric("Usuarios", df_filtered['user_id'].nunique())
c2.metric("Ingreso total", f"${df_filtered['monthly_revenue'].sum():,.0f}")
c3.metric("Ingreso promedio", f"${df_filtered['monthly_revenue'].mean():.2f}")

# =========================
# DISTRIBUCIÓN DE MINUTOS
# =========================
st.subheader("📞 Distribución de minutos mensuales")

fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(
    data=df_filtered,
    x='minutes_used',
    hue='plan',
    bins=30,
    element="step",
    stat="density",
    common_norm=False,
    ax=ax
)
st.pyplot(fig)
plt.close()

# =========================
# CONSUMO DE INTERNET
# =========================
st.subheader("🌐 Consumo de Internet (GB)")

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(
    data=df_filtered,
    x='plan',
    y='vol_count',
    ax=ax
)
st.pyplot(fig)
plt.close()

# =========================
# INGRESOS
# =========================
st.subheader("💰 Ingresos mensuales por plan")

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(
    data=df_filtered,
    x='plan',
    y='monthly_revenue',
    ax=ax
)
st.pyplot(fig)
plt.close()

# =========================
# PRUEBA DE HIPÓTESIS
# =========================
st.subheader("🧪 Prueba de hipótesis: Ingresos por plan")

surf = df_filtered[df_filtered['plan'] == 'Surf']['monthly_revenue']
ultimate = df_filtered[df_filtered['plan'] == 'Ultimate']['monthly_revenue']

if len(surf) >= 2 and len(ultimate) >= 2:
    _, p_levene = levene(surf, ultimate)
    equal_var = p_levene > 0.05
    stat, p_value = ttest_ind(surf, ultimate, equal_var=equal_var)

    st.write(f"P-valor: **{p_value:.15f}**")

    if p_value < 0.05:
        st.success("Existe diferencia estadísticamente significativa entre los planes")
    else:
        st.info("No se detecta diferencia estadísticamente significativa")
else:
    st.warning("Datos insuficientes para realizar la prueba.")

# =========================
# CONCLUSIONES
# =========================
st.header("📌 Conclusiones")

st.markdown("""
- El plan **Ultimate** genera mayor ingreso promedio por usuario.
- El consumo de internet es similar entre planes.
- La diferencia de ingresos resulta **estadísticamente significativa**.
""")
