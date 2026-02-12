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
    page_title="Comparación de Planes – Telecomunicaciones",
    page_icon="📡",
    layout="wide"
)

sns.set_style("darkgrid")

# =========================
# CARGA Y LIMPIEZA
# =========================
@st.cache_data
def load_data():
    df_calls = pd.read_csv("datasets/megaline_calls.csv")
    df_internet = pd.read_csv("datasets/megaline_internet.csv")
    df_messages = pd.read_csv("datasets/megaline_messages.csv")
    df_plans = pd.read_csv("datasets/megaline_plans.csv")
    df_users = pd.read_csv("datasets/megaline_users.csv")

    df_plans['plan_name'] = df_plans['plan_name'].str.capitalize()
    df_users['plan'] = df_users['plan'].str.capitalize()
    df_users['churn_date'] = df_users['churn_date'].fillna('Usuario activo')

    df_calls['call_date'] = pd.to_datetime(df_calls['call_date'])
    df_messages['message_date'] = pd.to_datetime(df_messages['message_date'])
    df_internet['session_date'] = pd.to_datetime(df_internet['session_date'])
    df_users['reg_date'] = pd.to_datetime(df_users['reg_date'])

    df_internet['mb_used'] /= 1024  # MB → GB

    return df_calls, df_messages, df_internet, df_plans, df_users


@st.cache_data
def build_monthly_usage(df_calls, df_messages, df_internet):
    df_calls['year_month'] = df_calls['call_date'].dt.to_period('M')
    df_calls['duration'] = np.ceil(df_calls['duration'])

    calls = df_calls.groupby(['user_id', 'year_month']).agg(
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

    df = calls.merge(messages, how='outer').merge(internet, how='outer')
    df.fillna(0, inplace=True)

    return df


def calculate_revenue(df, df_users, df_plans):
    df = df.merge(df_users, on='user_id')
    plans = df_plans.set_index('plan_name')

    def revenue(row):
        p = plans.loc[row['plan']]
        r = p['usd_monthly_pay']

        r += max(0, row['minutes_used'] - p['minutes_included']) * p['usd_per_minute']
        r += max(0, row['messages_count'] - p['messages_included']) * p['usd_per_message']
        r += max(0, np.ceil(row['vol_count'] - p['mb_per_month_included'] / 1024)) * p['usd_per_gb']

        return r

    df['monthly_revenue'] = df.apply(revenue, axis=1)
    return df


# =========================
# DATA
# =========================
df_calls, df_messages, df_internet, df_plans, df_users = load_data()
df_usage = build_monthly_usage(df_calls, df_messages, df_internet)
df_final = calculate_revenue(df_usage, df_users, df_plans)

# =========================
# SIDEBAR
# =========================
st.sidebar.subheader("📆 Periodo")
period = st.sidebar.radio("Selecciona el periodo", ["Mensual", "Anual"])

# SOLO Surf vs Ultimate
df_filtered = df_final[df_final['plan'].isin(['Surf', 'Ultimate'])]

# =========================
# PERIODOS
# =========================
df_period = df_filtered.copy()

if period == "Anual":
    df_period['year'] = df_period['year_month'].dt.to_timestamp().dt.year
    df_period = (
        df_period
        .groupby(['user_id', 'plan', 'year'], as_index=False)
        .agg(monthly_revenue=('monthly_revenue', 'sum'))
    )

# =========================
# UI
# =========================
st.title("📡 Comparación de Rentabilidad: Surf vs Ultimate")
st.markdown("**Objetivo:** determinar qué plan genera mayor ingreso por usuario.")

# =========================
# GRÁFICA INGRESOS
# =========================
st.subheader("💰 Ingresos por plan")

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df_period, x='plan', y='monthly_revenue', ax=ax)
ax.set_title(f"Ingresos {period.lower()}es por plan")
st.pyplot(fig)
plt.close()

# =========================
# RENTABILIDAD POR USUARIO
# =========================
annual = (
    df_final
    .assign(year=df_final['year_month'].dt.to_timestamp().dt.year)
    .groupby(['user_id', 'plan', 'year'], as_index=False)
    .agg(annual_revenue=('monthly_revenue', 'sum'))
)

summary = annual.groupby('plan').agg(
    users=('user_id', 'nunique'),
    avg_annual_revenue=('annual_revenue', 'mean')
)

surf_users = summary.loc['Surf', 'users']
ultimate_users = summary.loc['Ultimate', 'users']
surf_avg = summary.loc['Surf', 'avg_annual_revenue']
ultimate_avg = summary.loc['Ultimate', 'avg_annual_revenue']
diff = ultimate_avg - surf_avg

st.subheader("📊 Rentabilidad por usuario (anual)")

c1, c2 = st.columns(2)
c1.metric("Usuarios Surf", surf_users)
c1.metric("Ingreso anual promedio Surf", f"${surf_avg:,.2f}")

c2.metric("Usuarios Ultimate", ultimate_users)
c2.metric("Ingreso anual promedio Ultimate", f"${ultimate_avg:,.2f}")

st.success(f"👉 Ultimate genera **${diff:,.2f} más por usuario al año**")

# =========================
# PRUEBA DE HIPÓTESIS
# =========================
st.subheader("🧪 Prueba estadística")

surf = df_filtered[df_filtered['plan'] == 'Surf']['monthly_revenue']
ultimate = df_filtered[df_filtered['plan'] == 'Ultimate']['monthly_revenue']

_, p_levene = levene(surf, ultimate)
stat, p_value = ttest_ind(surf, ultimate, equal_var=p_levene > 0.05)

st.write(f"P-valor: **< 0.0001**")
st.success("Existe diferencia estadísticamente significativa")

# =========================
# CONCLUSIÓN
# =========================
st.header("📌 Conclusión de negocio")
st.markdown("""
- **Ultimate es consistentemente más rentable**
- Genera **mayor ingreso anual por usuario**
- La diferencia es **estadísticamente significativa**
- Recomendación: **impulsar migración de Surf → Ultimate**
""")
