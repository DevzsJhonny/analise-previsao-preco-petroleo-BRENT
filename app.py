# Importando as Bibliotecas
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import io
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet import Prophet
from datetime import date
from streamlit_option_menu import option_menu

# Configurações gerais da pagina do streamlit
st.set_page_config(layout="centered", page_title="Análise do Preço do Petróleo")

# Carregando dados
dados = pd.read_excel("Base_Tech.xlsx", parse_dates=["data"])
dados.rename(columns={"data": "ds", "preco": "y"}, inplace=True)
dados["ds"] = pd.to_datetime(dados["ds"])
dados["ano"] = dados["ds"].dt.year
dados["media"] = dados.groupby("ano")["y"].transform("mean").round(2)
dados["y"] = dados["y"].round(2)
dados.sort_values(by="ds", ascending=True, inplace=True)

dados_s = pd.read_excel("Base_Tech.xlsx", parse_dates=["data"])
dados_s.rename(columns={"data": "ds", "preco": "y"}, inplace=True)
dados_s = dados_s.set_index('ds', inplace=True)

# Carregando os dados dos derivados do petróleo
dados_dv = pd.read_excel("Base_Dv.xlsx", parse_dates=["ds"])
dados_dv.sort_values(by="ds", ascending=True, inplace=True)

# Adicionando as colunas de média e ano  para cada derivado
dados_dv_ano = dados_dv.copy()
dados_dv_ano["ano"] = dados_dv["ds"].dt.year
dados_dv_ano = dados_dv_ano.drop(columns=['ds'])
dados_dv_ano["Media_Diesel"] = dados_dv_ano.groupby("ano")["Diesel"].transform("mean").round(2)
dados_dv_ano["Media_Diesel_S10"] = dados_dv_ano.groupby("ano")["Diesel S10"].transform("mean").round(2)
dados_dv_ano["Media_Etanol"] = dados_dv_ano.groupby("ano")["Etanol"].transform("mean").round(2)
dados_dv_ano["Media_Gasolina"] = dados_dv_ano.groupby("ano")["Gasolina"].transform("mean").round(2)
dados_dv_ano["Media_Gasolina_Aditivada"] = dados_dv_ano.groupby("ano")["Gasolina Aditivada"].transform("mean").round(2)
dados_dv_ano["Media_GNV"] = dados_dv_ano.groupby("ano")["GNV"].transform("mean").round(2)
dados_dv_ano = dados_dv_ano.drop(columns=["Diesel","Diesel S10","Etanol","Gasolina","Gasolina Aditivada","GNV"])
dados_dv_ano = dados_dv_ano.groupby(["ano"]).mean()

# Inicializar valores padrão no session_state
if "data_inicio" not in st.session_state:
    st.session_state.data_inicio = date(2018, 1, 1)
if "data_fim" not in st.session_state:
    st.session_state.data_fim = date(2024, 10, 31)


# Menu lateral
with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options=["Participantes", "Introdução", "Desenvolvimento", "Paineis Interativos"],
        icons=["people-fill", "book-fill", "book-fill", "bar-chart-fill", "list-columns-reverse"],
        menu_icon="grid-fill",
        default_index=0,
    )

    # Submenu para Paineis
    if selected == "Paineis Interativos":
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
    unsafe_allow_html=True
    )

# Seção: Introdução
if selected == "Introdução":
    st.markdown("""


        # Análise e Previsão do Petróleo
        #### Utilizamos o modelo de Predição Prophet para tentar prever o preço diário do petróleo em Dólares (U$D)
                    
        _Este projeto é para fins educativos, não recomendamos como investimento de qualquer natureza._
        
        # O petróleo Brent 

        O Petróleo Brent é denominado assim por ser extraído de uma plataforma da Shell no Mar do Norte e que leva este mesmo nome.
        A Shell costumava nomear seus campos de exploração fazendo referência a aves aquáticas como o ganso-bravo (brent goose). <br>
        O barril *Brent* é o valor de referência mundial, usado inclusive pela política de preços da Petrobras. É um tipo de petróleo 
        bruto leve e doce e que possui baixo teor de enxofre e densidade, características que o tornam ideal para refino 
        de combustíveis como gasolina e diesel, e é extraído do Mar do Norte, entre a Noruega e o Reino Unido. <br>
        Ele serve como referência global para precificação de mais de dois terços do petróleo comercializado no mundo, sendo fundamental 
        para a economia global e influenciando diversos setores, desde combustíveis até produtos petroquímicos. 
        É padrão também para outros tipos de petróleo como o WTI (West Texas Intermediate) dos Estados Unidos. <br>
        
        """, unsafe_allow_html=True)
    
    # Apresentação da imagem do navio
    st.image("petroleo.jpg", caption="Navio Plataforma", use_column_width=True)

# Seção: Desenvolvimento
if selected == "Desenvolvimento":
    st.title("Análise do Preço do Petróleo")
                
    st.markdown("""
                
    Realizamos a utilização dos dados que foram disponibilizados pelo IPEA (Instituto de Pesquisa Econômica Aplicada). <br>
    <br>
    A base de dados conta com o histórico desde 1987 até outubro de 2024.<br>
    <br>
    Para avaliar o modelo de machine learning desenvolvido para prever o valor diário (utilizando séries temporais), empregamos dados de 2018 a 2024. Apesar da base de dados possuir mais de 30 anos, esse período foi selecionado para nossa análise, permitindo que nosso modelo tenha resultados mais concisos e uma boa performance no fechamento diário.<br>
    <br>
    A utilização do período de 2018 à outubro de 2024 se justifica pelos fatores da crise econômica como por exemplo a Grande Recessão de 2008, ou fatores geopolíticos como a Guerra do Iraque de 2003.<br>
    <br>
    Esses fatos ocorreram em décadas anteriores e causaram uma grande volatilidade no preço do petróleo. <br>
    <br>
    Tendo isso em mente, consideramos que um curto período já surja efeito em nossa análise, visto que tivemos grandes acontecimentos como a pandemia da COVID-19 em 2020 que impactou bastante no preço do petróleo, e em 2022 o conflito entre Ucrânia e Rússia. Essa guerra que ainda não chegou ao fim continua impactando nos preços.<br>
                            
    """, unsafe_allow_html=True)

    fig = px.line(dados, x="ds", y="y", title="Histórico do Preço do Petróleo")
    fig.update_layout(xaxis_title="Data de Referência", yaxis_title="Preço (em US$)", template="plotly_white")
    st.plotly_chart(fig)
    
    st.markdown("""

                Alguns dos fatores que afetaram o preço do petróleo nos últimos anos: <br>
                - *Conflitos Geopolíticos, sanções, crescimento econômico global e sazonalidades* <br>
                Muitos dos principais países produtores de petróleo estão em áreas politicamente instáveis, como a Venezuela e o Irã, e isso pode impactar a cotação do petróleo, principalmente quando os governos desses países tomam ações inesperadas envolvendo a política global.<br>

                - *Catástrofes ambientais* <br>
                Catástrofes como furacões em países que são produtores de petróleo também impactam negativamente o preço do petróleo. Por exemplo os furações no Golfo do México, algumas dessas catástrofes ambientais fazem com que seja necessário interromper as extrações de petróleo e gás natural (que também é uma fonte de combustível). A região é responsável por uma grande parte da produção de petróleo e gás natural dos Estados Unidos, com centenas de plataformas de perfuração operando lá.<br>

                - *Guerras*<br>
                A guerra entre Ucrânia e Rússia, iniciada em 02/2022 já impactou bastante na variação de preços do petróleo. A produção de petróleo na Ucrânia é muito pequena em comparação com a Rússia, que continua a ser uma potência energética global, mas embora a produção da Ucrânia esteja por volta de 50 a 70 mil barris por dia, ela desempenha um papel importante no transporte de energia, o que afeta os fluxos de petróleo e gás entre a Rússia e a Europa. Já a Rússia é um dos três maiores produtores de petróleo do mundo, ficando atrás apenas dos Estados Unidos e da Arábia Saudita.<br>

                - *Pandemias* <br>
                A pandemia de COVID-19, iniciada em meados de 2020, impactou fortemente nos preços do petróleo, principalmente no início deste período. O preço estava entre 70,25 dólares no início do mês de janeiro deste ano e atingiu o menor valor de 2020 em 21 de abril com preço em 9,12 dólares.<br>
                
    """, unsafe_allow_html=True)

    # Garantindo que dados_s seja uma série temporal válida
    dados_s = dados.set_index('ds')['y']  # Agora 'dados_s' é uma série temporal com 'ds' como índice e 'y' como valores

    # Decomposição sazonal
    resultado = seasonal_decompose(dados_s, model='multiplicative', period=120)

    # Título e descrição adicional
    st.markdown("""
    ### Decomposição Sazonal do Preço do Petróleo
    A decomposição sazonal ajuda a entender melhor os padrões sazonais nos preços do petróleo ao longo do tempo. A decomposição divide os dados em componentes como tendência, sazonalidade e resíduos, o que pode fornecer insights sobre os fatores que afetam as flutuações nos preços.
    """, unsafe_allow_html=True)

    st.subheader("Componente de Tendência")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=resultado.trend.index, y=resultado.trend.dropna(), mode='lines', name='Trend'))
    st.plotly_chart(fig)

    st.subheader("Componente Sazonal")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=resultado.seasonal.index, y=resultado.seasonal.dropna(), mode='lines', name='Trend'))
    st.plotly_chart(fig)

    st.subheader("Resíduos")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=resultado.resid.index, y=resultado.resid.dropna(), mode='lines', name='Trend'))
    st.plotly_chart(fig)

# Seção: Paineis
if selected == "Paineis Interativos":
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

        #Intervalo de anos
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

        st.write("""
                 
        Aqui é o local em que apresentaremos previsões utilizando o modelo PROPHET. <br>
        Apresentamos toda a base de dados no primeiro gráfico, que é de 01/2018 até 10/2024.
        Iniciamos a previsão em 01/11/24 e conseguimos prever os valores (com certa margem de erro) até 03/26.

        É valido citar que, as previsões são realizadas apenas para os dias úteis.
        """, unsafe_allow_html=True)

        # Gráfico histórico de preços
        fig = px.line(dados, x="ds", y="y", title="Histórico do Preço do Petróleo")
        fig.update_layout(xaxis_title="Data de Referência", yaxis_title="Preço (em US$)", template="plotly_white")
        st.plotly_chart(fig)

        # Remover colunas desnecessárias (como 'ano' e 'media', se existirem)
        df_1 = dados[['ds','y']]

        # Funções de Avaliação
        def calculate_metrics(y_true, y_pred):
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            return mae, mse, mape

        # Função para mostras as métricas
        def print_metrics(metrics):
            mae, mse, mape = metrics
            st.write(f"**MAE (Mean Absolute Error)**: {mae:.2f}")
            st.write(f"**MSE (Mean Squared Error)**: {mse:.2f}")
            st.write(f"**MAPE (Mean Absolute Percentage Error)**: {mape:.2f}%")

        tamanho = int(len(df_1)* 0.8)
        treino = df_1[:tamanho] #o treino é "tudo até a data de corte, menos a data de corte"
        teste = df_1[tamanho:] #o teste ficou "tudo a partir da data de corte, contando com ela"

        modelo = Prophet()
        modelo.add_seasonality(name='mensal', period=30, fourier_order=8)
        modelo.fit(df_1)

        futuro = modelo.make_future_dataframe(periods = 362, freq='B')
        forecast = modelo.predict(futuro)

        forecast_teste = forecast[forecast['ds'] > df_1['ds'].max()]

        # Gráfico com cores distintas
        fig_colored = go.Figure()


        # Dados de previsão
        fig_colored.add_trace(go.Scatter(
            x=forecast_teste['ds'], y=forecast_teste['yhat'], 
            mode='lines', name='Previsão', 
            line=dict(color='orange')
        ))

        # Configuração final do gráfico
        fig_colored.update_layout(
            title="Previsão do Preço do Petróleo",
            xaxis_title="Data",
            yaxis_title="Preço (em US$)",
            template="plotly_white"
        )

        st.write("Abaixo é possível escolher até qual data quer que seja realizada a previsão.")
        # Intervalo de datas
        st.write("### Selecione uma data:")

        data_ref = st.date_input("Intervalo a partir da primeira previsão de dados a partir de 01/11/24")

        grafico = forecast_teste[
            (forecast_teste["ds"] >= pd.to_datetime('01/11/2024')) &
            (forecast_teste["ds"] <= pd.to_datetime(data_ref))
        ]

        grid = grafico[['ds','yhat']]
        grid['ds'] = pd.to_datetime(grid['ds'], format='%d/%m/%Y')
        grid.rename(columns={'ds': 'Data', 'yhat': 'Valor Predito'}, inplace=True)

        # Gráfico com cores distintas
        grafico_1 = go.Figure()

        # Dados históricos
        grafico_1.add_trace(go.Scatter(
            x=grafico['ds'], y=grafico['yhat'], 
            mode='lines', name='Histórico', 
            line=dict(color='green')
        ))

        st.plotly_chart(grafico_1)

        st.markdown("<center><br><br><h3>Métricas para avaliação do modelo </h3> </center> <br>", unsafe_allow_html=True)

        prophet_m = calculate_metrics(teste['y'], forecast_teste['yhat'].values)
        print_metrics(prophet_m)

        # Exemplo de dataframe que você deseja exportar (substitua por seu dataframe real)
        df_exportar = grid  # Aqui 'grid' é o dataframe que você quer exportar

        # Cria um buffer em memória para o arquivo Excel
        output = io.BytesIO()

        # Escreve o dataframe no arquivo Excel com openpyxl
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df_exportar.to_excel(writer, index=False, sheet_name="Dados")

        # Rewind o buffer para que possa ser lido
        output.seek(0)

        # Exibe o botão para download
        st.markdown("<center><h3>Exportar Tabela para Excel</h3></center>", unsafe_allow_html=True)

        st.download_button(
            label="Baixar Dados em Excel",
            data=output,
            file_name="dados_previsao_petroleo.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


    # Seção: Desenvolvimento
    if submenu == "Derivados":
        st.title("Derivados")
        st.markdown("""
        ### **Produtos Derivados de Petróleo no Brasil** <br>
        Baseado na “Série Histórica de Preços de Combustíveis e de GLP” disponível no site da ANP 
        Agência Nacional do Petróleo, Gás Natural e Biocombustíveis, que acompanha através de pesquisa semanal
        os preços praticados por revendedores de combustíveis automotivos e de gás liquefeito de petróleo envasilhado em botijões,
        buscamos os dados detalhados por cada revendedor em cada cidade de todos os estados, e aplicamos uma média diária de preço de venda por produto. <br><br>
                    
        Considerando a base de dados da ANP, disponível por semestres, entre Janeiro de 2020 até Junho de 2024, que é o último período disponível. <br><br>
        
        Os resultados estão representados nos gráficos abaixo.
        """, unsafe_allow_html=True)

        # Inicializar valores padrão no session_state
        if "data_inicio_ano" not in st.session_state:
            st.session_state.data_inicio_ano = date(2020, 1, 1)
        if "data_fim_ano" not in st.session_state:
            st.session_state.data_fim_ano = date(2024, 10, 31)

        intervalo_datas = st.date_input(
            "Intervalo de datas",
            value=(st.session_state.data_inicio_ano, st.session_state.data_fim_ano)
        )

        lista = ["Diesel","Diesel S10","Etanol","Gasolina","Gasolina Aditivada","GNV"]

        input_dv = st.multiselect(
            "Selecione os derivados:",  # Rótulo do campo
            options=lista,             # Opções disponíveis
            default=["Diesel","Diesel S10","Etanol","Gasolina","Gasolina Aditivada","GNV"]  # Valor(es) padrão
        )

        st.session_state.data_inicio_ano, st.session_state.data_fim_ano = intervalo_datas

        # Filtrar os dados com base no intervalo
        df_dv_filtrado = dados_dv[
            (dados_dv["ds"] >= pd.to_datetime(st.session_state.data_inicio_ano)) &
            (dados_dv["ds"] <= pd.to_datetime(st.session_state.data_fim_ano))
        ]

        # Gerar o gráfico de preço
        fig_preco_dv = px.line(df_dv_filtrado, x="ds", y=input_dv, title="Variação do Preço dos Derivados do Petróleo")
        st.plotly_chart(fig_preco_dv)

        # Ajustando as nomenclaturas dos eixos x e y
        fig_preco_dv.update_layout(
            xaxis_title="Data de Referência",
            yaxis_title="Preço (em US$)",
            template="plotly_white"
        )
        # Lista da Multiseleção
        lista = ["Media_Diesel", "Media_Diesel_S10", "Media_Etanol", "Media_Gasolina", "Media_Gasolina_Aditivada", "Media_GNV"]

        # Multiseleção
        input_dv_ano = st.multiselect(
            "Selecione os derivados:",  # Rótulo do campo
            options=lista,             # Opções disponíveis
            default=["Media_Diesel", "Media_Diesel_S10", "Media_Etanol", "Media_Gasolina", "Media_Gasolina_Aditivada", "Media_GNV"]  # Valor(es) padrão
        )


        # Garantir que a coluna 'ano' seja criada corretamente
        dados_dv_ano["ano"] = dados_dv_ano.index

        # Filtrar os dados com base no intervalo de anos
        df_dv_filtrado_ano = dados_dv_ano[
            (dados_dv_ano["ano"] >= st.session_state.data_inicio_ano.year) &
            (dados_dv_ano["ano"] <= st.session_state.data_fim_ano.year)
        ]

        # Gráfico de barras para os derivados
        fig_preco_dv_ano = px.bar(
            df_dv_filtrado_ano, 
            x="ano", 
            y=input_dv_ano, 
            title="Variação do Preço dos Derivados do Petróleo"
        )

        # Atualizando o layout do gráfico
        fig_preco_dv_ano.update_layout(
            xaxis_title="Ano",
            yaxis_title="Média de Preço (em US$)",
            template="plotly_white"
        )

        # Adicionando os valores sobre as barras
        fig_preco_dv_ano.update_traces(
            texttemplate='%{y}', 
            textposition='outside'
        )

        # Exibir o gráfico
        st.plotly_chart(fig_preco_dv_ano)
