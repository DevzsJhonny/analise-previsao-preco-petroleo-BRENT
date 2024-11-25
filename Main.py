# Para instalar o Streamlit, use: pip install streamlit

import streamlit as st

# Título da aplicação
st.title("Aplicação Simples com Streamlit")

# Entrada de texto
nome = st.text_input("Digite seu nome:")

# Botão para exibir saudação
if st.button("Enviar"):
    if nome:
        st.success(f"Olá, {nome}! Bem-vindo ao Streamlit! 🎉")
    else:
        st.error("Por favor, digite seu nome!")
