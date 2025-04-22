import streamlit as st
import pandas as pd
import glob
import plotly.express as px
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile, urllib.request
from pathlib import Path

from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

logo = Image.open('7038fb25-82d5-478f-9b43-a19ac46cb9ed.png')
st.set_page_config(
    page_title="Анализ данных системы отправки рейсов",
    page_icon=logo,
    layout="wide"
)

@st.cache_data
def load_data():
    url = "https://drive.google.com/uc?id=1kXz-DCE2jgKuAfyvlAtri0fOYvRdz1J_&export=download"
    zip_path = "data.zip"
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall("data")
    dfs = []
    for fn in glob.glob("data/**/*.csv", recursive=True):
        df0 = pd.read_csv(fn, sep=";", encoding="cp1251")
        df0 = df0.loc[:, ~df0.columns.str.contains("^Unnamed")]
        df0["passengers"] = pd.to_numeric(df0["Кол-во пасс."], errors="coerce").fillna(0).astype(int)
        df0["contract_short"] = df0["№ договора"].fillna("Без договора").str.extract(r'([^\\s]+)').fillna("Без договора")
        df0 = df0.rename(columns={"Код а/к": "airline", "Код а/п": "airport", "Номер рейса": "flight_no"})
        dfs.append(df0)
    if not dfs:
        st.error("Не найдено ни одного CSV в папке data/")
        return pd.DataFrame()
    df_all = pd.concat(dfs, ignore_index=True)
    df_all["dep_date"] = pd.to_datetime(df_all["Дата вылета"], dayfirst=True, errors="coerce")
    return df_all

df = load_data()

custom_palette = ["#1f77b4", "#5fa2dd", "#a3c9f7", "#cce4ff", "#e6f2ff"]
def format_russian_number(x):
    if x >= 1_000_000:
        return f"{x / 1_000_000:.1f} млн."
    elif x >= 1_000:
        return f"{x / 1_000:.0f} тыс."
    else:
        return str(int(x))

def render_chart(df_filt, group_col, title, kind="bar", showlegend=False):
    agg = df_filt.groupby(group_col)["passengers"].sum().reset_index(name="value")
    if agg.empty:
        agg = pd.DataFrame({group_col: ["Нет данных"], "value": [0]})
    top5 = agg.nlargest(5, "value")
    top5["pct"] = top5["value"] / top5["value"].sum()
    if kind == "pie":
        fig = px.pie(top5, values="value", names=group_col, hole=0.3,
                     color_discrete_sequence=custom_palette)
        fig.update_traces(textposition='inside', textinfo='label+percent+value',
                          hovertemplate="<b>%{label}</b><br>Пассажиры: %{value:,}<br>%{percent:.1%}<extra></extra>")
    else:
        if group_col == "contract_short":
            text_labels = top5.apply(lambda r: f"{format_russian_number(r['value'])} ({r['pct']:.1%})", axis=1)
        else:
            text_labels = top5["value"].apply(format_russian_number)
        fig = px.bar(top5, x="value", y=group_col, orientation="h", text=text_labels,
                     color=group_col, color_discrete_sequence=custom_palette,
                     labels={"value": "Пассажиры", group_col: title})
        fig.update_traces(textposition="inside", insidetextanchor="start", textfont=dict(color="black"),
                          hovertemplate="<b>%{y}</b><br>Пассажиры: %{x:,}<extra></extra>")
    fig.update_layout(title=title, xaxis_tickformat=",", xaxis_title=None, yaxis_title=None,
                      margin=dict(t=50,l=50,r=50,b=50), hovermode="closest",
                      showlegend=showlegend, font=dict(family="Arial", size=14, color="black"))
    return fig

section = st.sidebar.radio(
    "Выберите раздел:",
    ["Основные диаграммы", "Дополнительная аналитика", "Прогноз", "Кластеры", "Аномалии"]
)

if section == "Основные диаграммы":
    st.title("Анализ данных системы отправки рейсов")
    charts = [("Топ‑5 авиакомпаний","airline","bar"),("Топ‑5 направлений","flight_no","bar"),
              ("Топ‑5 аэропортов","airport","bar"),("Договоры (круговая)","contract_short","pie"),
              ("Договоры (столбчатая)","contract_short","bar")]
    for i,(t,col,k) in enumerate(charts):
        st.subheader(t)
        sd = st.date_input("Дата начала", value=df["dep_date"].min(), key=f"sd{i}")
        ed = st.date_input("Дата окончания", value=df["dep_date"].max(), key=f"ed{i}")
        mask = (df.dep_date >= pd.to_datetime(sd)) & (df.dep_date <= pd.to_datetime(ed))
        st.plotly_chart(render_chart(df.loc[mask],col,t,kind=k,showlegend=(k=="pie")), use_container_width=True)

elif section == "Дополнительная аналитика":
    st.title("Дополнительная аналитика рейсов")
    add = [("Рейсы по месяцам","flight_no","flights"),("Пассажиры по месяцам","passengers","passengers"),
           ("Средняя загрузка","avg_passengers","avg_passengers"),("Тепловая карта","heatmap","heatmap")]
    for title,ylabel,key in add:
        with st.expander(title, expanded=True):
            sd = st.date_input("Дата начала", value=df["dep_date"].min(), key=f"a_sd_{key}")
            ed = st.date_input("Дата окончания", value=df["dep_date"].max(), key=f"a_ed_{key}")
            df_f = df[(df.dep_date >= pd.to_datetime(sd)) & (df.dep_date <= pd.to_datetime(ed))]
            if key=="flights":
                df_f["month"] = df_f.dep_date.dt.to_period("M").astype(str)
                data = df_f.groupby("month")["flight_no"].nunique().reset_index()
                fig=px.line(data,x="month",y="flight_no",markers=True,title=title,labels={"flight_no":ylabel})
            elif key=="passengers":
                df_f["month"]=df_f.dep_date.dt.to_period("M").astype(str)
                data=df_f.groupby("month")["passengers"].sum().reset_index()
                fig=px.line(data,x="month",y="passengers",markers=True,title=title,labels={"passengers":ylabel})
            elif key=="avg_passengers":
                data=df_f.groupby("flight_no")["passengers"].mean().reset_index()
                data=data[data.passengers>0]
                fig=px.histogram(data,x="passengers",nbins=20,title=title,labels={"passengers":ylabel})
            else:

                df_f["month_num"] = df_f.dep_date.dt.month
                df_f["dow"] = df_f.dep_date.dt.dayofweek
                month_map={i:m for i,m in enumerate(
                    ['Январь','Февраль','Март','Апрель','Май','Июнь','Июль','Август','Сентябрь','Октябрь','Ноябрь','Декабрь'],1)}
                day_map={i:d for i,d in enumerate(
                    ['Понедельник','Вторник','Среда','Четверг','Пятница','Суббота','Воскресенье'])}
                df_f['Месяц']=df_f.month_num.map(month_map)
                df_f['День недели']=df_f.dow.map(day_map)
                df_f['Месяц']=pd.Categorical(df_f['Месяц'],categories=list(month_map.values()),ordered=True)
                df_f['День недели']=pd.Categorical(df_f['День недели'],categories=list(day_map.values()),ordered=True)
                heat= df_f.groupby(['Месяц','День недели']).size().unstack(fill_value=0)
                fig, ax = plt.subplots(figsize=(12,6))
                sns.heatmap(heat,cmap="YlOrRd",annot=True,fmt="d",linewidths=.5,ax=ax)
                ax.set_xlabel("День недели")
                ax.set_ylabel("Месяц")
            st.plotly_chart(fig,use_container_width=True) if key!="heatmap" else st.pyplot(fig)

elif section == "Прогноз":
    st.title("Прогноз пассажиропотока на 6 месяцев")
    hist = df.groupby(df.dep_date.dt.to_period('M'))['passengers'].sum().reset_index()
    hist['ds'] = hist.dep_date.dt.to_timestamp()
    hist['y'] = hist.passengers
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(hist[['ds','y']])
    future = m.make_future_dataframe(periods=6, freq='M')
    forecast = m.predict(future)
    fig = plot_plotly(m, forecast)
    st.plotly_chart(fig, use_container_width=True)

elif section == "Кластеры":
    st.title("Кластеризация маршрутов по средней загрузке")
    agg = df.groupby('flight_no')['passengers'].mean().reset_index()
    scaler = StandardScaler()
    X = scaler.fit_transform(agg[['passengers']])
    agg['cluster'] = KMeans(n_clusters=3, random_state=42).fit_predict(X)
    fig = px.scatter(agg, x='flight_no', y='passengers', color='cluster',
                     title="3 кластера по средней загрузке")
    st.plotly_chart(fig, use_container_width=True)

elif section == "Аномалии":
    st.title("Аномалии в пассажиропотоке (Z-score > 2)")
    daily = df.groupby('dep_date')['passengers'].sum().reset_index()
    daily['zscore'] = (daily['passengers'] - daily['passengers'].mean())/daily['passengers'].std()
    anomalies = daily[np.abs(daily['zscore']) > 2]
    st.line_chart(daily.set_index('dep_date')['passengers'])
    st.subheader("Аномальные даты и значения")
    st.table(anomalies[['dep_date','passengers','zscore']])