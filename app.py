import streamlit as st
import pandas as pd
import glob
import plotly.express as px
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

logo = Image.open('7038fb25-82d5-478f-9b43-a19ac46cb9ed.png')

st.set_page_config(
    page_title="Анализ данных системы отправки рейсов",
    page_icon=logo,
    layout="wide"
)

@st.cache_data
def load_data():
    csv_files = glob.glob("W&&B_Libra-Particularized.*.csv")
    dfs = []
    for fn in csv_files:
        df = pd.read_csv(fn, sep=";", encoding="cp1251")
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        df["dep_date"] = pd.to_datetime(df["Дата вылета"], dayfirst=True, errors="coerce")
        df["passengers"] = pd.to_numeric(df["Кол-во пасс."], errors="coerce").fillna(0).astype(int)
        df["contract_short"] = df["№ договора"].fillna("Без договора").str.extract(r'([^\\s]+)')
        df["contract_short"] = df["contract_short"].fillna("Без договора")
        df = df.rename(columns={"Код а/к": "airline", "Код а/п": "airport", "Номер рейса": "flight_no"})
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

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
        fig = px.pie(
            top5,
            values="value",
            names=group_col,
            hole=0.3,
            color_discrete_sequence=custom_palette
        )
        fig.update_traces(
            textposition='inside',
            textinfo='label+percent+value',
            hovertemplate="<b>%{label}</b><br>Пассажиры: %{value:,}<br>%{percent:.1%}<extra></extra>",
            insidetextfont=dict(color="black")
        )
    else:

        if group_col == "contract_short":
            text_labels = top5.apply(
                lambda row: f"{format_russian_number(row['value'])} ({row['pct']:.1%})",
                axis=1
            )
        else:
            text_labels = top5["value"].apply(format_russian_number)

        fig = px.bar(
            top5,
            x="value",
            y=group_col,
            orientation="h",
            text=text_labels,
            color=group_col,
            color_discrete_sequence=custom_palette,
            labels={"value": "Пассажиры", group_col: title}
        )
        fig.update_traces(
            textposition="inside",
            insidetextanchor="start",
            textfont=dict(color="black"),
            hovertemplate="<b>%{y}</b><br>Пассажиры: %{x:,}<extra></extra>"
        )

    fig.update_layout(
        title=title,
        xaxis_tickformat=",",
        xaxis_title=None,
        yaxis_title=None,
        margin=dict(t=50, l=50, r=50, b=50),
        hovermode="closest",
        showlegend=showlegend,
        font=dict(family="Arial", size=14, color="black"),
    )

    return fig

st.sidebar.header("Навигация")
section = st.sidebar.radio(
    "Выберите раздел:",
    ("Основные диаграммы", "Дополнительная аналитика")
)

if section == "Основные диаграммы":
    st.title("Анализ данных системы отправки рейсов")

    sections = [
        ("Топ‑5 авиакомпаний", "airline", "bar"),
        ("Топ‑5 направлений", "flight_no", "bar"),
        ("Топ‑5 аэропортов", "airport", "bar"),
        ("Договоры (круговая)", "contract_short", "pie"),
        ("Договоры (столбчатая)", "contract_short", "bar"),
    ]

    for i, (title, group_col, chart_type) in enumerate(sections):
        st.subheader(title)
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Дата начала",
                value=df["dep_date"].min(),
                key=f"{group_col}_start_{i}"
            )
        with col2:
            end_date = st.date_input(
                "Дата окончания",
                value=df["dep_date"].max(),
                key=f"{group_col}_end_{i}"
            )

        mask = (df.dep_date >= pd.to_datetime(start_date)) & (df.dep_date <= pd.to_datetime(end_date))
        df_filtered = df.loc[mask]

        fig = render_chart(
            df_filtered,
            group_col,
            title,
            kind=chart_type,
            showlegend=True if chart_type == "pie" else False
        )
        st.plotly_chart(fig, use_container_width=True)

elif section == "Дополнительная аналитика":
    st.title("Дополнительная аналитика рейсов")

    additional_charts = [
        ("📅 Динамика количества рейсов по месяцам", "Количество рейсов", "flights"),
        ("👥 Динамика количества пассажиров по месяцам", "Количество пассажиров", "passengers"),
        ("🛫 Средняя загрузка пассажиров на рейс", "Средняя загрузка", "avg_passengers"),
        ("🔥 Тепловая карта активности", "Активность по дням недели", "heatmap")
    ]

    for i, (chart_title, y_label, chart_key) in enumerate(additional_charts):
        with st.expander(chart_title, expanded=(i == 0)):
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Дата начала",
                    value=df["dep_date"].min(),
                    key=f"additional_start_{chart_key}"
                )
            with col2:
                end_date = st.date_input(
                    "Дата окончания",
                    value=df["dep_date"].max(),
                    key=f"additional_end_{chart_key}"
                )

            mask = (df.dep_date >= pd.to_datetime(start_date)) & (df.dep_date <= pd.to_datetime(end_date))
            df_filtered = df.loc[mask]

            if chart_key == "flights":
                df_filtered["month"] = df_filtered["dep_date"].dt.to_period("M").astype(str)
                flights_by_month = df_filtered.groupby("month")["flight_no"].nunique().reset_index()

                fig = px.line(
                    flights_by_month,
                    x="month",
                    y="flight_no",
                    markers=True,
                    title="Количество рейсов по месяцам",
                    labels={"month": "Месяц", "flight_no": y_label},
                    color_discrete_sequence=["#1f77b4"]
                )
                fig.update_traces(line_shape="linear", marker=dict(size=8))
                fig.update_layout(xaxis_title="Месяц", yaxis_title=y_label)
                st.plotly_chart(fig, use_container_width=True)

            elif chart_key == "passengers":
                df_filtered["month"] = df_filtered["dep_date"].dt.to_period("M").astype(str)
                passengers_by_month = df_filtered.groupby("month")["passengers"].sum().reset_index()

                fig = px.line(
                    passengers_by_month,
                    x="month",
                    y="passengers",
                    markers=True,
                    title="Количество пассажиров по месяцам",
                    labels={"month": "Месяц", "passengers": y_label},
                    color_discrete_sequence=["#5fa2dd"]
                )
                fig.update_traces(line_shape="linear", marker=dict(size=8))
                fig.update_layout(xaxis_title="Месяц", yaxis_title=y_label)
                st.plotly_chart(fig, use_container_width=True)

            elif chart_key == "avg_passengers":
                avg_passengers = df_filtered.groupby("flight_no")["passengers"].mean().reset_index()
                avg_passengers = avg_passengers[avg_passengers["passengers"] > 0]

                fig = px.histogram(
                    avg_passengers,
                    x="passengers",
                    nbins=20,
                    title="Средняя загрузка пассажиров на рейс",
                    labels={"passengers": y_label},
                    color_discrete_sequence=["#1f77b4"]
                )
                fig.update_layout(xaxis_title="Пассажиры на рейс", yaxis_title="Количество рейсов")
                st.plotly_chart(fig, use_container_width=True)

            elif chart_key == "heatmap":
                weekday_mapping = {
                    'Monday': 'Понедельник',
                    'Tuesday': 'Вторник',
                    'Wednesday': 'Среда',
                    'Thursday': 'Четверг',
                    'Friday': 'Пятница',
                    'Saturday': 'Суббота',
                    'Sunday': 'Воскресенье'
                }

                df_filtered["weekday"] = df_filtered["dep_date"].dt.day_name().map(weekday_mapping)

                month_mapping = {
                    'January': 'Январь',
                    'February': 'Февраль',
                    'March': 'Март',
                    'April': 'Апрель',
                    'May': 'Май',
                    'June': 'Июнь',
                    'July': 'Июль',
                    'August': 'Август',
                    'September': 'Сентябрь',
                    'October': 'Октябрь',
                    'November': 'Ноябрь',
                    'December': 'Декабрь'
                }

                df_filtered["month_name"] = df_filtered["dep_date"].dt.month_name().map(month_mapping)

                if 'Месяц' not in df.columns or 'День недели' not in df.columns:
                    df['Дата вылета'] = pd.to_datetime(df['Дата вылета'], errors='coerce')
                    df['Месяц'] = df['Дата вылета'].dt.month
                    df['День недели'] = df['Дата вылета'].dt.dayofweek

                month_map = {
                    1: 'Январь', 2: 'Февраль', 3: 'Март', 4: 'Апрель',
                    5: 'Май', 6: 'Июнь', 7: 'Июль', 8: 'Август',
                    9: 'Сентябрь', 10: 'Октябрь', 11: 'Ноябрь', 12: 'Декабрь'
                }

                day_map = {
                    0: 'Понедельник', 1: 'Вторник', 2: 'Среда',
                    3: 'Четверг', 4: 'Пятница', 5: 'Суббота', 6: 'Воскресенье'
                }

                df['Месяц'] = df['Месяц'].map(month_map)
                df['День недели'] = df['День недели'].map(day_map)

                month_order = [
                    'Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь',
                    'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь'
                ]
                day_order = [
                    'Понедельник', 'Вторник', 'Среда', 'Четверг',
                    'Пятница', 'Суббота', 'Воскресенье'
                ]

                df['Месяц'] = pd.Categorical(df['Месяц'], categories=month_order, ordered=True)
                df['День недели'] = pd.Categorical(df['День недели'], categories=day_order, ordered=True)

                min_date = df['Дата вылета'].min()
                max_date = df['Дата вылета'].max()

                start_date, end_date = st.date_input(
                    "Выберите диапазон дат для тепловой карты:",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )

                filtered_df = df[
                    (df['Дата вылета'] >= pd.to_datetime(start_date)) & (df['Дата вылета'] <= pd.to_datetime(end_date))]

                heatmap_data = filtered_df.groupby(['Месяц', 'День недели']).size().unstack(fill_value=0)

                st.subheader("🔥 Тепловая карта активности по месяцам и дням недели")
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.heatmap(heatmap_data, cmap="YlOrRd", annot=True, fmt="d", linewidths=.5, ax=ax)
                ax.set_xlabel("День недели")
                ax.set_ylabel("Месяц")
                st.pyplot(fig)