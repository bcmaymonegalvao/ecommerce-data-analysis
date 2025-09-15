import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

@st.cache_data(ttl=3600)
def load_data():
    base_url = "https://raw.githubusercontent.com/bcmaymonegalvao/ecommerce-data-analysis/main/"
    data_files = {
        "geolocation": "olist_geolocation_dataset.parquet",
        "customers": "olist_customers_dataset.parquet",
        "translation": "product_category_name_translation.parquet",
        "sellers": "olist_sellers_dataset.parquet",
        "products": "olist_products_dataset.parquet",
        "orders": "olist_orders_dataset.parquet",
        "reviews": "olist_order_reviews_dataset.parquet",
        "payments": "olist_order_payments_dataset.parquet",
        "order_items": "olist_order_items_dataset.parquet"
    }
    data = {}
    for key, filename in data_files.items():
        st.info(f"Carregando {filename}...")
        df = pd.read_parquet(base_url + filename)
        data[key] = df
    return data

def eda_section(data):
    st.header("Análise Exploratória dos Dados")

    tab_names = list(data.keys())
    tab = st.selectbox("Escolha a base para análise", tab_names)
    df = data[tab]
    st.subheader(f"Visualização de '{tab}'")
    st.write(df.head())

    st.subheader("Resumo estatístico")
    st.write(df.describe(include='all'))

    st.subheader("Gráfico de distribuições para colunas numéricas")
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    chosen_col = st.selectbox("Escolha uma coluna numérica", num_cols)
    fig = px.histogram(df, x=chosen_col)
    st.plotly_chart(fig)

def relational_analysis(data):
    st.header("Análise Relacional entre as Bases")

    st.subheader("Quantidade de pedidos por estado do cliente (geolocalização)")
    customers = data['customers']
    orders = data['orders']

    df_merged = orders.merge(customers, on='customer_id')
    state_counts = df_merged['customer_state'].value_counts().reset_index()
    state_counts.columns = ['Estado', 'Número de Pedidos']

    fig = px.bar(state_counts, x='Estado', y='Número de Pedidos', title='Pedidos por Estado')
    st.plotly_chart(fig)

    st.subheader("Top 10 categorias de produto vendidas")
    order_items = data['order_items']
    products = data['products']
    df_prod = order_items.merge(products[['product_id', 'product_category_name']], on='product_id')
    cat_counts = df_prod['product_category_name'].value_counts().reset_index().head(10)
    cat_counts.columns = ['Categoria', 'Número de Itens Vendidos']
    fig2 = px.bar(cat_counts, x='Categoria', y='Número de Itens Vendidos', title='Top 10 categorias')
    st.plotly_chart(fig2)

def prepare_ml_data(data):
    # Criar dataset para recomendação (exemplo simples):
    # Prever se um consumidor vai comprar uma categoria de produto
    customers = data['customers']
    orders = data['orders']
    order_items = data['order_items']
    products = data['products']

    # Mesclar para ter cliente, pedido e categoria do produto
    df = orders.merge(order_items, on='order_id')
    df = df.merge(products[['product_id', 'product_category_name']], on='product_id')
    df = df.merge(customers[['customer_id', 'customer_unique_id']], on='customer_id')

    # Codificar cliente e categoria (label encoding simples)
    df['category_code'] = df['product_category_name'].astype('category').cat.codes
    df['customer_code'] = df['customer_unique_id'].astype('category').cat.codes

    # Construir features e target:
    # Simplificação: Features = customer_code, Target = category_code
    # Modelagem de multi-class para recomendar categoria para usuário

    X = df[['customer_code']]
    y = df['category_code']

    return X, y, df

def train_and_evaluate_models(X, y):
    st.header("Treinamento e Avaliação dos Modelos de Recomendação")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)
    xgb_pred = xgb.predict(X_test)

    rf_acc = accuracy_score(y_test, rf_pred)
    xgb_acc = accuracy_score(y_test, xgb_pred)

    st.write(f"Random Forest Accuracy: {rf_acc:.4f}")
    st.write(f"XGBoost Accuracy: {xgb_acc:.4f}")

    best_model = rf if rf_acc > xgb_acc else xgb
    st.success(f"Melhor modelo escolhido: {'Random Forest' if best_model == rf else 'XGBoost'}")

    return best_model

def recommend_product(model, df, customer_id):
    st.header("Recomendar produtos")

    st.write(f"Recomendações para o cliente código: {customer_id}")

    customer_code = df.loc[df['customer_unique_id'] == customer_id, 'customer_code'].iloc[0]

    possible_categories = df['category_code'].unique()
    preds = []
    for cat in possible_categories:
        preds.append((cat, model.predict([[customer_code]])[0]))

    # Aqui simplificamos retornando categorias mais prováveis (pode ser melhorado)
    st.write("Produto(s) recomendado(s):")
    top_cats_codes = model.predict([[customer_code]])
    top_cats = df.loc[df['category_code'].isin(top_cats_codes), 'product_category_name'].unique()
    st.write(top_cats)

def main():
    st.set_page_config(
        page_title="E-commerce Data Analysis & Recommender",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.sidebar.title("Menu de Navegação")
    options = ["Carga e EDA", "Análise Relacional", "Recomendação de Produtos"]
    choice = st.sidebar.radio("Escolha uma seção", options)

    data = load_data()

    if choice == "Carga e EDA":
        eda_section(data)
    elif choice == "Análise Relacional":
        relational_analysis(data)
    elif choice == "Recomendação de Produtos":
        X, y, df_ml = prepare_ml_data(data)
        model = train_and_evaluate_models(X, y)

        st.sidebar.subheader("Faça uma Requisição")
        customers = df_ml['customer_unique_id'].unique()
        selected_customer = st.sidebar.selectbox("Selecione o cliente para recomendação", customers)

        recommend_product(model, df_ml, selected_customer)

if __name__ == "__main__":
    main()
