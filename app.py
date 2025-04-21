import streamlit as st
import pandas as pd
import glob
import plotly.express as px
from PIL import Image

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
                df_filtered["weekday"] = df_filtered["dep_date"].dt.day_name(locale="Russian")
                df_filtered["month_name"] = df_filtered["dep_date"].dt.month_name(locale="Russian")

                pivot = df_filtered.pivot_table(index="weekday", columns="month_name", values="flight_no", aggfunc="count", fill_value=0)
                pivot = pivot.reindex(["Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота", "Воскресенье"])

                fig = px.imshow(
                    pivot,
                    labels=dict(x="Месяц", y="День недели", color="Количество рейсов"),
                    color_continuous_scale="Blues"
                )
                fig.update_layout(title="Тепловая карта активности")
                st.plotly_chart(fig, use_container_width=True)