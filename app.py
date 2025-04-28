import streamlit as st
import pandas as pd
import glob
import plotly.express as px
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import zipfile, urllib.request
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

logo = Image.open('logo.png')
st.set_page_config(
    page_title="Анализ данных системы отправки рейсов",
    page_icon=logo,
    layout="wide"
)

@st.cache_data
def load_data(uploaded_file_new=None):
    if uploaded_file_new is not None:
        if uploaded_file_new.name.endswith('.zip'):
            with zipfile.ZipFile(uploaded_file_new, "r") as z:
                z.extractall("data_custom")
            search_path = "data_custom/**/*.csv"
        elif uploaded_file_new.name.endswith('.csv'):
            df0 = pd.read_csv(uploaded_file_new, sep=";", encoding="cp1251")
            df0 = df0.loc[:, ~df0.columns.str.contains("^Unnamed")]
            df0["passengers"] = pd.to_numeric(df0["Кол-во пасс."], errors="coerce").fillna(0).astype(int)
            df0["contract_short"] = df0["№ договора"].fillna("Без договора").str.extract(r'([^\\s]+)').fillna(
                "Без договора")
            df0 = df0.rename(columns={"Код а/к": "airline", "Код а/п": "airport", "Номер рейса": "flight_no"})
            df0["dep_date"] = pd.to_datetime(df0["Дата вылета"], dayfirst=True, errors="coerce")
            return df0
        else:
            st.error("Поддерживаются только файлы .zip или .csv!")
            return pd.DataFrame()
    else:
        url = "https://drive.google.com/uc?id=1kXz-DCE2jgKuAfyvlAtri0fOYvRdz1J_&export=download"
        zip_path = "data.zip"
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall("data")
        search_path = "data/**/*.csv"

    dfs = []
    for fn in glob.glob(search_path, recursive=True):
        df0 = pd.read_csv(fn, sep=";", encoding="cp1251")
        df0 = df0.loc[:, ~df0.columns.str.contains("^Unnamed")]
        df0["passengers"] = pd.to_numeric(df0["Кол-во пасс."], errors="coerce").fillna(0).astype(int)
        df0["contract_short"] = df0["№ договора"].fillna("Без договора").str.extract(r'([^\\s]+)').fillna(
            "Без договора")
        df0 = df0.rename(columns={"Код а/к": "airline", "Код а/п": "airport", "Номер рейса": "flight_no"})
        dfs.append(df0)

    if not dfs:
        st.error("Не найдено ни одного CSV!")
        return pd.DataFrame()

    df_all = pd.concat(dfs, ignore_index=True)
    df_all["dep_date"] = pd.to_datetime(df_all["Дата вылета"], dayfirst=True, errors="coerce")
    return df_all

st.sidebar.title("Загрузка данных")
uploaded_file = st.sidebar.file_uploader("Загрузите ZIP или CSV файл", type=["zip", "csv"])

df = load_data(uploaded_file)

custom_palette = ["#1f77b4", "#5fa2dd", "#a3c9f7", "#cce4ff", "#e6f2ff"]

def format_russian_number(x):
    if x >= 1_000_000:
        return f"{x / 1_000_000:.1f} млн."
    elif x >= 1_000:
        return f"{x / 1_000:.0f} тыс."
    else:
        return str(int(x))

def render_chart(df_filt, group_col, chart_title, kind="bar", showlegend=False):
    agg_data = df_filt.groupby(group_col)["passengers"].sum().reset_index(name="value")
    if agg_data.empty:
        agg_data = pd.DataFrame({group_col: ["Нет данных"], "value": [0]})
    top5 = agg_data.nlargest(5, "value")
    top5["pct"] = top5["value"] / top5["value"].sum()
    if kind == "pie":
        chart = px.pie(top5, values="value", names=group_col, hole=0.3,
                       color_discrete_sequence=custom_palette)
        chart.update_traces(textposition='inside', textinfo='label+percent+value',
                            hovertemplate="<b>%{label}</b><br>Пассажиры: %{value:,}<br>%{percent:.1%}<extra></extra>")
    else:
        if group_col == "contract_short":
            text_labels = top5.apply(lambda r: f"{format_russian_number(r['value'])} ({r['pct']:.1%})", axis=1)
        else:
            text_labels = top5["value"].apply(format_russian_number)
        chart = px.bar(top5, x="value", y=group_col, orientation="h", text=text_labels,
                       color=group_col, color_discrete_sequence=custom_palette,
                       labels={"value": "Пассажиры", group_col: chart_title})
        chart.update_traces(textposition="inside", insidetextanchor="start", textfont=dict(color="black"),
                            hovertemplate="<b>%{y}</b><br>Пассажиры: %{x:,}<extra></extra>")
    chart.update_layout(title=chart_title, xaxis_tickformat=",", xaxis_title=None, yaxis_title=None,
                        margin=dict(t=50, l=50, r=50, b=50), hovermode="closest",
                        showlegend=showlegend, font=dict(family="Arial", size=14, color="black"))
    return chart

section = st.sidebar.radio(
    "Выберите раздел:",
    ["Основные диаграммы", "Дополнительная аналитика", "Прогноз", "Кластеры", "Аномалии"]
)

if section == "Основные диаграммы":
    st.title("Анализ данных системы отправки рейсов")
    charts = [("Топ‑5 авиакомпаний","airline","bar"),("Топ‑5 направлений","flight_no","bar"),
              ("Топ‑5 аэропортов","airport","bar"),("Договоры (круговая)","contract_short","pie"),
              ("Договоры (столбчатая)","contract_short","bar")]
    for i, (t, col, k) in enumerate(charts):
        st.subheader(t)

        col1, col2 = st.columns([1, 1])

        with col1:
            sd = st.date_input("Дата начала", value=df["dep_date"].min(), key=f"sd{i}")

        with col2:
            ed = st.date_input("Дата окончания", value=df["dep_date"].max(), key=f"ed{i}")

        sd = pd.to_datetime(sd)
        ed = pd.to_datetime(ed)

        mask = (df.dep_date >= sd) & (df.dep_date <= ed)

        st.plotly_chart(render_chart(df.loc[mask], col, t, kind=k, showlegend=(k == "pie")), use_container_width=True)

elif section == "Дополнительная аналитика":
    st.title("Дополнительная аналитика рейсов")

    add = [("Рейсы по месяцам", "flight_no", "flights"),
           ("Пассажиры по месяцам", "passengers", "passengers"),
           ("Средняя загрузка", "avg_passengers", "avg_passengers"),
           ("Тепловая карта", "heatmap", "heatmap")]

    for title, ylabel, key in add:
        with st.expander(title, expanded=True):
            col1, col2 = st.columns([1, 1])

            with col1:
                sd = st.date_input("Дата начала", value=df["dep_date"].min(), key=f"start_date_{key}")

            with col2:
                ed = st.date_input("Дата окончания", value=df["dep_date"].max(), key=f"end_date_{key}")

            sd = pd.to_datetime(sd)
            ed = pd.to_datetime(ed)

            df_f = df[(df.dep_date >= sd) & (df.dep_date <= ed)]

            if key == "flights":
                df_f["month"] = df_f.dep_date.dt.to_period("M").astype(str)
                data = df_f.groupby("month")["flight_no"].nunique().reset_index()
                fig = px.line(data, x="month", y="flight_no", markers=True, title=title, labels={"flight_no": ylabel})
            elif key == "passengers":
                df_f["month"] = df_f.dep_date.dt.to_period("M").astype(str)
                data = df_f.groupby("month")["passengers"].sum().reset_index()
                fig = px.line(data, x="month", y="passengers", markers=True, title=title, labels={"passengers": ylabel})
            elif key == "avg_passengers":
                data = df_f.groupby("flight_no")["passengers"].mean().reset_index()
                data = data[data.passengers > 0]
                fig = px.histogram(data, x="passengers", nbins=20, title=title, labels={"passengers": ylabel})
            else:
                df_f["month_num"] = df_f.dep_date.dt.month
                df_f["dow"] = df_f.dep_date.dt.dayofweek
                month_map = {i: m for i, m in enumerate(
                    ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь', 'Июль', 'Август', 'Сентябрь', 'Октябрь',
                     'Ноябрь', 'Декабрь'], 1)}
                day_map = {i: d for i, d in enumerate(
                    ['Понедельник', 'Вторник', 'Среда', 'Четверг', 'Пятница', 'Суббота', 'Воскресенье'])}
                df_f['Месяц'] = df_f.month_num.map(month_map)
                df_f['День недели'] = df_f.dow.map(day_map)
                df_f['Месяц'] = pd.Categorical(df_f['Месяц'], categories=list(month_map.values()), ordered=True)
                df_f['День недели'] = pd.Categorical(df_f['День недели'], categories=list(day_map.values()),
                                                     ordered=True)
                heat = df_f.groupby(['Месяц', 'День недели']).size().unstack(fill_value=0)
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.heatmap(heat, cmap="YlOrRd", annot=True, fmt="d", linewidths=.5, ax=ax)
                ax.set_xlabel("День недели")
                ax.set_ylabel("Месяц")

            if key != "heatmap":
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.pyplot(fig)

elif section == "Прогноз":
    st.title("Прогноз пассажиропотока на 6 месяцев")

    hist = df.groupby(df.dep_date.dt.to_period('M'))['passengers'].sum().reset_index()
    hist['ds'] = hist.dep_date.dt.to_timestamp()
    hist['y'] = hist.passengers

    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(hist[['ds', 'y']])

    future = m.make_future_dataframe(periods=6, freq='M')
    forecast = m.predict(future)

    fig = plot_plotly(m, forecast)
    st.plotly_chart(fig, use_container_width=True)

    comp_fig = m.plot_components(forecast)
    st.pyplot(comp_fig)

    st.markdown("**Прогноз на следующие 6 месяцев:**")

    next6 = (forecast[["ds", "yhat"]]
             .tail(6)
             .rename(columns={"ds": "Месяц", "yhat": "Прогноз пассажиров"})
             )

    next6["Прогноз пассажиров"] = (
        next6["Прогноз пассажиров"]
        .clip(lower=0)
        .round(0)
        .astype(int)
    )
    st.dataframe(next6.set_index("Месяц"))

    st.markdown(
        """
        **Интерпретация прогноза:**
        - **Тренд** показывает общее направление изменения пассажиропотока:  
          – Если линия растёт, спрос на рейсы увеличивается.  
          – Если падает — возможен спад или сезонный спад.
        - **Сезонность (годовая)** выявляет повторяющиеся колебания в течение года:  
          – Пики обычно приходятся на летние месяцы (июнь–август).  
          – Спады — на зимние месяцы (январь–февраль).
        - Выделенные **компоненты** помогают понять, какие эффекты (долгосрочный тренд или сезонные флуктуации) доминируют.
        """
    )

elif section == "Кластеры":
    st.title("Кластеризация маршрутов по средней загрузке")

    agg = df.groupby('flight_no')['passengers'].mean().reset_index()

    scaler = StandardScaler()
    X = scaler.fit_transform(agg[['passengers']])

    kmeans = KMeans(n_clusters=3, random_state=42)
    agg['cluster'] = kmeans.fit_predict(X)

    centers = scaler.inverse_transform(kmeans.cluster_centers_).flatten()

    order = centers.argsort()
    labels = {
        order[0]: "Низкая загрузка",
        order[1]: "Средняя загрузка",
        order[2]: "Высокая загрузка"
    }
    agg['cluster_label'] = agg['cluster'].map(labels)

    fig = px.scatter(
        agg,
        x='flight_no',
        y='passengers',
        color='cluster_label',
        title="Кластеры маршрутов по средней загрузке",
        labels={"cluster_label": "Тип кластера"}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Центры кластеров (средняя загрузка пассажиров):**")
    for i, c in enumerate(centers):
        st.write(f"- Кластер «{labels[i]}»: {c:.1f} пассажиров в среднем")

    st.markdown(
        """
        **Интерпретация кластеров:**
        - **Низкая загрузка**: маршруты, где в среднем менее {:.0f} пассажиров.
        - **Средняя загрузка**: маршруты с примерно {:.0f}–{:.0f} пассажирами.
        - **Высокая загрузка**: маршруты, где более {:.0f} пассажиров.

        Такие группы помогают:
        1. Выявить маршруты, которые стоит стимулировать (низкая загрузка).
        2. Отследить «рабочие лошадки» (средняя загрузка).
        3. Определить сверхпопулярные направления (высокая загрузка).
        """
        .format(
            centers[order[0]],
            centers[order[0]],
            centers[order[2]],
            centers[order[2]]
        )
    )

elif section == "Аномалии":
    st.title("Обнаружение аномалий в пассажиропотоке")

    mean_p = df['passengers'].mean()
    std_p = df['passengers'].std()
    threshold = mean_p + 3 * std_p

    st.markdown(f"Порог выбросов (mean + 3·std): **{threshold:.0f}** пассажиров")

    X = df[['passengers']].values

    df['threshold_anomaly'] = df['passengers'] > threshold

    iso = IsolationForest(
        contamination=0.01,  # предполагаем, что ~1% точек — аномалии
        random_state=42
    )
    df['iso_label'] = iso.fit_predict(X)  # -1 = аномалия, 1 = нормальная точка
    df['iso_anomaly'] = df['iso_label'] == -1

    lof = LocalOutlierFactor(
        n_neighbors=20,  # число соседей для оценки плотности
        contamination=0.01  # доля аномалий
    )
    df['lof_label'] = lof.fit_predict(X)  # -1 = аномалия, 1 = нормальная точка
    df['lof_anomaly'] = df['lof_label'] == -1

    tabs = st.tabs(["Порог", "IsolationForest", "LocalOutlierFactor"])
    with tabs[0]:
        anom = df[df['threshold_anomaly']]
        fig0 = px.scatter(
            anom, x='dep_date', y='passengers', color='flight_no',
            title=f"Аномалии по порогу (> {threshold:.0f})"
        )
        st.plotly_chart(fig0, use_container_width=True)
        st.dataframe(
            anom[['flight_no', 'dep_date', 'passengers']]
            .rename(columns={'flight_no': 'Рейс', 'dep_date': 'Дата', 'passengers': 'Пассажиров'})
            .reset_index(drop=True)
        )

    with tabs[1]:
        anom_iso = df[df['iso_anomaly']]
        fig1 = px.scatter(
            anom_iso, x='dep_date', y='passengers', color='flight_no',
            title="Аномалии IsolationForest"
        )
        st.plotly_chart(fig1, use_container_width=True)
        st.dataframe(
            anom_iso[['flight_no', 'dep_date', 'passengers']]
            .rename(columns={'flight_no': 'Рейс', 'dep_date': 'Дата', 'passengers': 'Пассажиров'})
            .reset_index(drop=True)
        )

    with tabs[2]:
        anom_lof = df[df['lof_anomaly']]
        fig2 = px.scatter(
            anom_lof, x='dep_date', y='passengers', color='flight_no',
            title="Аномалии LocalOutlierFactor"
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(
            anom_lof[['flight_no', 'dep_date', 'passengers']]
            .rename(columns={'flight_no': 'Рейс', 'dep_date': 'Дата', 'passengers': 'Пассажиров'})
            .reset_index(drop=True)
        )

    st.markdown(
        """
        Интерпретация графиков:
    - **Порог**: отмечаем рейсы с очень большим числом пассажиров — тут всё просто, это самые «выдающиеся» случаи.
    - **IsolationForest**: алгоритм сам ищет «редкие» точки в данных и отмечает их как аномалии.
    - **LocalOutlierFactor**: сравнивает каждый рейс с его ближайшими «соседями» и помечает те, что сильно отличаются.

    Таким образом, вы сразу увидите три разных списка «подозрительных» рейсов и сможете выбрать тот метод, который понятнее или удобнее.
    """
    )