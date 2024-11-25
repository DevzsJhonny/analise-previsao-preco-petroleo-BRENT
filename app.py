import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from prophet import Prophet
from datetime import date
from streamlit_option_menu import option_menu

# Configurações gerais
st.set_page_config(layout="wide", page_title="Análise do Preço do Petróleo")

dados = pd.read_excel("Base_Tech.xlsx", parse_dates=["data"])
dados.rename(columns={"data": "ds", "preco": "y"}, inplace=True)
dados["ds"] = pd.to_datetime(dados["ds"])
dados["ano"] = dados["ds"].dt.year
dados["media"] = dados.groupby("ano")["y"].transform("mean").round(2)
dados["y"] = dados["y"].round(2)
dados.sort_values(by="ds", ascending=True, inplace=True)


# Inicializar valores padrão no session_state
if "data_inicio" not in st.session_state:
    st.session_state.data_inicio = date(2018, 1, 1)
if "data_fim" not in st.session_state:
    st.session_state.data_fim = date(2023, 12, 31)


# Menu lateral
with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options=["Participantes", "Introdução", "Desenvolvimento", "Paineis", "Referências"],
        icons=["people-fill", "book-fill", "book-fill", "bar-chart-fill", "list-columns-reverse"],
        menu_icon="grid-fill",
        default_index=0,
    )

    # Submenu para Paineis
    if selected == "Paineis":
        submenu = option_menu(
            menu_title="SubMenu Paineis",
            options=["Apresentação Geral", "Previsão de Dados", "Derivados"],
            default_index=0,
        )

# Seção: Participantes
if selected == "Participantes":

    st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="margin-bottom: 1rem;">FIAP - PÓS TECH - DATA ANALYTICS</h1>
        <h2 style="margin-bottom: 1rem;">FASE 4 - TECH CHALLENGE</h2>
        <h3 style="margin-bottom: 1rem;">TURMA 5DTAT - GRUPO 44</h3>
        <h4 style="margin-bottom: 2rem;">INTEGRANTES</h4>
        <p style="font-size: 20px; line-height: 1.5;">
            Gabriel Silva Ferreira<br>
            Gustavo Duran Domingues<br>
            Jhonny Amorim Silva<br>
            Lucas Alexander dos Santos<br>
            Sandro Semmer
        </p>
    </div>
    """,
    unsafe_allow_html=True,
    )

# Seção: Introdução
if selected == "Introdução":
    st.title("Introdução")
    st.write("Descrição e objetivos do projeto.")

# Seção: Desenvolvimento
if selected == "Desenvolvimento":
    st.title("Desenvolvimento")
    st.write("Análise do projeto.")

# Seção: Paineis
if selected == "Paineis":
    if submenu == "Apresentação Geral":
        st.title("Visualização dos Dados")

        # Intervalo de datas
        st.write("### Selecione um intervalo de datas:")
        intervalo_datas = st.date_input(
            "Intervalo de datas",
            value=(st.session_state.data_inicio, st.session_state.data_fim)
        )
        
        st.session_state.data_inicio, st.session_state.data_fim = intervalo_datas

        # Filtrar os dados com base no intervalo
        df_filtrado = dados[
            (dados["ds"] >= pd.to_datetime(st.session_state.data_inicio)) &
            (dados["ds"] <= pd.to_datetime(st.session_state.data_fim))
        ]

        # Gerar o gráfico de preço
        fig_preco = px.line(df_filtrado, x="ds", y="y", title="Variação do Preço do Petróleo")
        fig_preco.update_layout(
            xaxis_title="Data de Referência",
            yaxis_title="Preço (em US$)",
            template="plotly_white"
        )

        df_filtrado_ano = dados[
            (dados["ds"] >= pd.to_datetime(st.session_state.data_inicio)) &
            (dados["ds"] <= pd.to_datetime(st.session_state.data_fim))
        ]

        # Gerar o gráfico de média anual
        fig_media = px.bar(df_filtrado_ano, x="ano", y="media", title="Média Anual do Preço do Petróleo")
        fig_media.update_layout(
            xaxis_title="Ano",
            yaxis_title="Média de Preço (em US$)",
            template="plotly_white"
        )
        fig_media.update_traces(text=dados["media"], textposition="outside")

        # Armazenar o gráfico no session_state
        st.session_state.grafico_preco = fig_preco
        st.session_state.grafico_media = fig_media

        # Mostrar o gráfico de preço se disponível
        if st.session_state.grafico_preco:
            st.plotly_chart(st.session_state.grafico_preco)

        # Mostrar o gráfico de média anual se disponível
        if st.session_state.grafico_media:
            st.plotly_chart(st.session_state.grafico_media)
            

    if submenu == "Previsão de Dados":
        st.title("Previsão dos Dados")

        # Gráfico histórico de preços
        fig = px.line(dados, x="ds", y="y", title="Histórico do Preço do Petróleo")
        fig.update_layout(xaxis_title="Data de Referência", yaxis_title="Preço (em US$)", template="plotly_white")
        st.plotly_chart(fig)

        df_1 = dados.copy().drop(columns=["ano", "media"])
        train_size = int(len(df_1) * 0.8)
        train, test = df_1[:train_size], df_1[train_size:]

        # Exibindo treino vs teste
        fig_train_test = go.Figure()
        fig_train_test.add_trace(go.Scatter(x=train["ds"], y=train["y"], mode="lines", name="Treinamento", line=dict(color="blue")))
        fig_train_test.add_trace(go.Scatter(x=test["ds"], y=test["y"], mode="lines", name="Teste", line=dict(color="red")))
        fig_train_test.update_layout(title="Treinamento vs Teste", xaxis_title="Data", yaxis_title="Preço (em US$)")
        st.plotly_chart(fig_train_test)

    # Seção: Desenvolvimento
    if submenu == "Derivados":
        st.title("Derivados")
        st.write("Produtos Derivados do Petróleo.")

# Seção: Referências
if selected == "Referências":
    st.title("Referências")
    st.write("Liste aqui as referências usadas no projeto.")