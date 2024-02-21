import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import random

# Carregar os dados
data = pd.read_csv('dados_carros.csv')

# Pré-processamento dos dados
todas_marca_combustivel_cambio = pd.concat([data['Marca'], data['Combustivel'], data['Cambio']]).unique()
le = LabelEncoder()
le.fit(todas_marca_combustivel_cambio)

data['Marca_Codigo'] = le.transform(data['Marca'])
data['Combustivel_Codigo'] = le.transform(data['Combustivel'])
data['Cambio_Codigo'] = le.transform(data['Cambio'])

# Dividir os dados em features e target
X = data[['Marca_Codigo', 'Combustivel_Codigo', 'Cambio_Codigo']]
y = data['Modelo']

# Construir o modelo de árvore de decisão
modelo = DecisionTreeClassifier()
modelo.fit(X, y)

# Interface Streamlit
st.title('Sistema Especialista de Recomendação de Carros')

marca = st.selectbox('Selecione a Marca:', options=data['Marca'].unique())
combustivel = st.selectbox('Selecione o Combustível:', options=data['Combustivel'].unique())
cambio = st.selectbox('Selecione o Câmbio:', options=data['Cambio'].unique())

if st.button('Recomendar'):
    marca_codigo = le.transform([marca])[0]
    combustivel_codigo = le.transform([combustivel])[0]
    cambio_codigo = le.transform([cambio])[0]
    dados_entrada = [[marca_codigo, combustivel_codigo, cambio_codigo]]
    previsoes = modelo.predict_proba(dados_entrada)[0]  
    carros_disponiveis = [modelo.classes_[i] for i, probabilidade in enumerate(previsoes) if probabilidade > 0] 
    if len(carros_disponiveis) < 1:
        st.write("Não há carros disponíveis com as características selecionadas.")
    else:
        carros_recomendados = random.choices(carros_disponiveis, k=5) 
        st.write('Modelos Recomendados:')
        for carro in carros_recomendados:
            preco_medio = data[data['Modelo'] == carro]['PrecoMedio'].mean()
            st.write(f"Modelo: {carro}, Preço Médio: R${preco_medio:.2f}")
#versao final