import streamlit as st
import pandas as pd

model = pd.read_pickle("model_happy.pkl")

st.markdown("# Descubra a felicidade")

cursos = st.selectbox("Quantos cursos acompanhou do Téo Me Why?", [
    "0", "1", "2", "3", "Mais que 3"
    ])

teo = st.selectbox("Como conheceu o Téo Me Why?", [
    "LinkedIn", "Twitch", "YouTube", "Instagram", "Amigos","Twitter / X", "Outra rede social"
    ])

col1, col2, col3 = st.columns(3)

with col1:
    video_game = st.radio("Curte Video Game?", ["Sim", "Não"])
    futebol = st.radio("Curte Futebol?", ["Sim", "Não"])
    idade = st.number_input("Sua idade", 18, 100)
    tempo = st.selectbox("Tempo que atua na área de dados", [
        "De 0 a 6 meses", "De 6 meses a 1 ano", "De 1 ano a 2 anos",
        "de 2 anos a 4 anos", "Mais de 4 anos", "Não atuo"
    ])

with col2:
    livros = st.radio("Curte Livros?", ["Sim", "Não"])
    tabuleiro = st.radio("Curte Jogos de Tabeleiro?", ["Sim", "Não"])
    estado = st.selectbox("Estado que mora atualmente", [
        "MG", "SC", "SP", "CE", "PE", "RJ", "AM", "PR", "BA", "PA", "MT",
        "RS", "DF", "RN", "ES", "PB", "GO", "MA"
    ])
    posicao = st.selectbox("Posição da cadeira (senioridade)", [
        "Iniciante", "Júnior", "Pleno", "Sênior" , "Coordenação",
        "Especialista", "Gerência", "Diretoria", "C-Level"
    ])

with col3:
    formula = st.radio("Curte Fórmula 1?", ["Sim", "Não"])
    mma = st.radio("Curte MMA?", ["Sim", "Não"])
    area = st.selectbox("Área de Formação", [
        "Exatas", "Biológicas", "Humanas"
    ])

data = {
    "Como conheceu o Téo Me Why?": teo,
    "Quantos cursos acompanhou do Téo Me Why?": cursos, 
    "Curte games?": video_game,
    "Curte futebol?": futebol, 
    "Curte livros?": livros, 
    "Curte jogos de tabuleiro?": tabuleiro,
    "Curte jogos de fórmula 1?": formula, 
    "Curte jogos de MMA?": mma, 
    "Idade": idade,
    "Estado que mora atualmente": estado, 
    "Área de Formação": area,
    "Tempo que atua na área de dados": tempo, 
    "Posição da cadeira (senioridade)": posicao
}

df = pd.DataFrame([data]).replace({"Sim": 1, "Não": 0})

dummy_vars = [
    "Como conheceu o Téo Me Why?",
    "Quantos cursos acompanhou do Téo Me Why?",
    "Estado que mora atualmente",
    "Área de Formação",
    "Tempo que atua na área de dados",
    "Posição da cadeira (senioridade)"
]

df = pd.get_dummies(df[dummy_vars]).astype(int)

template = [
    'Como conheceu o Téo Me Why?_Amigos',
    'Como conheceu o Téo Me Why?_Instagram',
    'Como conheceu o Téo Me Why?_LinkedIn',
    'Como conheceu o Téo Me Why?_Outra rede social',
    'Como conheceu o Téo Me Why?_Twitch',
    'Como conheceu o Téo Me Why?_Twitter / X',
    'Como conheceu o Téo Me Why?_YouTube',
    'Quantos cursos acompanhou do Téo Me Why?_0',
    'Quantos cursos acompanhou do Téo Me Why?_1',
    'Quantos cursos acompanhou do Téo Me Why?_2',
    'Quantos cursos acompanhou do Téo Me Why?_3',
    'Quantos cursos acompanhou do Téo Me Why?_Mais que 3',
    'Estado que mora atualmente_AM', 'Estado que mora atualmente_BA',
    'Estado que mora atualmente_CE', 'Estado que mora atualmente_DF',
    'Estado que mora atualmente_ES', 'Estado que mora atualmente_GO',
    'Estado que mora atualmente_MA', 'Estado que mora atualmente_MG',
    'Estado que mora atualmente_MT', 'Estado que mora atualmente_PA',
    'Estado que mora atualmente_PB', 'Estado que mora atualmente_PE',
    'Estado que mora atualmente_PR', 'Estado que mora atualmente_RJ',
    'Estado que mora atualmente_RN', 'Estado que mora atualmente_RS',
    'Estado que mora atualmente_SC', 'Estado que mora atualmente_SP',
    'Área de Formação_Biológicas', 'Área de Formação_Exatas',
    'Área de Formação_Humanas',
    'Tempo que atua na área de dados_De 0 a 6 meses',
    'Tempo que atua na área de dados_De 1 ano a 2 anos',
    'Tempo que atua na área de dados_De 6 meses a 1 ano',
    'Tempo que atua na área de dados_Mais de 4 anos',
    'Tempo que atua na área de dados_Não atuo',
    'Tempo que atua na área de dados_de 2 anos a 4 anos',
    'Posição da cadeira (senioridade)_C-Level',
    'Posição da cadeira (senioridade)_Coordenação',
    'Posição da cadeira (senioridade)_Diretoria',
    'Posição da cadeira (senioridade)_Especialista',
    'Posição da cadeira (senioridade)_Gerência',
    'Posição da cadeira (senioridade)_Iniciante',
    'Posição da cadeira (senioridade)_Júnior',
    'Posição da cadeira (senioridade)_Pleno',
    'Posição da cadeira (senioridade)_Sênior', 'Curte games?',
    'Curte futebol?', 'Curte livros?', 'Curte jogos de tabuleiro?',
    'Curte jogos de fórmula 1?', 'Curte jogos de MMA?', 'Idade'
    ]

df_template = pd.DataFrame(columns=template)

df = pd.concat([df_template, df]).fillna(0)

proba = model["model"].predict_proba(df[model["features"]])[:,1][0]

if proba > 0.7:
    st.success(f"Você é uma pessoa feliz! Probabilidade: {proba*100:.0f}%")
elif proba > 0.4:
    st.warning(f"Você é uma pessoa meio feliz! Probabilidade: {proba*100:.0f}%")
else:
    st.error(f"Você não é uma pessoa feliz! Probabilidade: {proba*100:.0f}%")