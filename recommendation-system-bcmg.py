import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import kagglehub
from kagglehub import KaggleDatasetAdapter

# --- Header e Footer personalizados ---
def add_header():
    st.markdown("""
    <style>
    .header {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        font-size: 2rem;
        padding: 0.5rem 2rem;
        z-index: 1000;
        box-shadow: 0 4px 15px rgba(31, 38, 135, 0.37);
        display: flex;
        align-items: center;
    }
    .header span {
        margin-left: 0.75rem;
    }
    .main-content {
        padding-top: 4rem;
        padding-bottom: 3rem; /* evitar sobreposição do footer */
    }
    </style>
    <div class="header">
        🛒🚀 E-Commerce Data Analysis & Recommender <span>🔍🤖</span>
    </div>
    """, unsafe_allow_html=True)

def add_footer():
    st.markdown("""
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: linear-gradient(90deg, #764ba2 0%, #6c5ce7 100%);
        color: white;
        text-align: center;
        font-size: 1rem;
        padding: 0.75rem 1rem;
        box-shadow: 0 -4px 15px rgba(31, 38, 135, 0.37);
        z-index: 1000;
    }
    .footer a {
        color: #ffe066;
        text-decoration: none;
        font-weight: bold;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    </style>
    <div class="footer">
        Desenvolvido com ❤️ por <a href="https://github.com/bcmaymonegalvao" target="_blank">Bruno Galvão</a> | 
        <span>📅 2025</span> | &copy; Todos os direitos reservados
    </div>
    """, unsafe_allow_html=True)

# --- Carregamento e análises (exemplo reduzido) ---
@st.cache_data(ttl=3600)
def load_data():
    # Dataset do Kaggle
    dataset = "olistbr/brazilian-ecommerce"

    # Mapeamento Kaggle → nossos nomes
    data_files = {
        "geolocation": "olist_geolocation_dataset.csv",
        "customers": "olist_customers_dataset.csv",
        "translation": "product_category_name_translation.csv",
        "sellers": "olist_sellers_dataset.csv",
        "products": "olist_products_dataset.csv",
        "orders": "olist_orders_dataset.csv",
        "reviews": "olist_order_reviews_dataset.csv",
        "payments": "olist_order_payments_dataset.csv",
        "order_items": "olist_order_items_dataset.csv"
    }

    data = {}
    for key, filename in data_files.items():
        st.info(f"Carregando {filename} do Kaggle...")
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            dataset,
            file_path=filename,
            pandas_kwargs={"low_memory": False}
        )
        data[key] = df

    return data

def eda_section(data):
    st.header("🔎 Análise Exploratória dos Dados")
    tab_names = list(data.keys())
    tab = st.selectbox("📂 Escolha a base para análise", tab_names)
    df = data[tab]
    st.dataframe(df.head())
    st.write(df.describe(include='all'))
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    if num_cols:
        chosen_col = st.selectbox("📊 Escolha coluna numérica para visualizar histograma", num_cols)
        fig = px.histogram(df, x=chosen_col)
        st.plotly_chart(fig)

def relational_analysis(data):
    st.header("🔗 Análise Relacional entre as Bases")
    try:
        orders = data['orders']
        customers = data['customers']
        df_merged = orders.merge(customers, on='customer_id')
        state_counts = df_merged['customer_state'].value_counts().reset_index()
        state_counts.columns = ['Estado', 'Pedidos']
        fig = px.bar(state_counts, x='Estado', y='Pedidos', title='Pedidos por Estado')
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Erro na análise relacional: {e}")

def prepare_ml_data(data):
    customers = data['customers']
    orders = data['orders']
    order_items = data['order_items']
    products = data['products']

    df = orders.merge(order_items, on='order_id').merge(products[['product_id','product_category_name']], on='product_id')
    df = df.merge(customers[['customer_id','customer_unique_id']], on='customer_id')

    df['category_code'] = df['product_category_name'].astype('category').cat.codes
    df['customer_code'] = df['customer_unique_id'].astype('category').cat.codes

    X = df[['customer_code']]
    y = df['category_code']

    return X, y, df


def train_and_evaluate_models(X, y):
    st.header("🧠 Treinamento e Avaliação dos Modelos")

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
    st.success(f"🏆 Melhor modelo escolhido: {'Random Forest' if best_model == rf else 'XGBoost'}")

    return best_model

def recommend_product(model, df, customer_id):
    st.header("🎯 Recomendações de Produto")
    st.write(f"Cliente selecionado: {customer_id}")

    customer_code = df.loc[df['customer_unique_id'] == customer_id, 'customer_code'].iloc[0]
    pred_cat_code = model.predict([[customer_code]])[0]

    cat_name = df.loc[df['category_code'] == pred_cat_code, 'product_category_name'].iloc[0]
    st.success(f"Produto mais recomendado para o cliente é da categoria: **{cat_name}**")

def main():
    st.set_page_config(page_title="E-commerce Data Analysis & Recommender", layout="wide")
    add_header()

    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    st.sidebar.title("Menu")
    options = ["Análise Exploratória", "Análise Relacional", "Recomendação"]
    choice = st.sidebar.radio("Escolha uma seção", options)

    data = load_data()

    if choice == "Análise Exploratória":
        eda_section(data)
    elif choice == "Análise Relacional":
        relational_analysis(data)
    elif choice == "Recomendação":
        X, y, df_ml = prepare_ml_data(data)
        model = train_and_evaluate_models(X, y)
        customers = df_ml['customer_unique_id'].unique()
        selected_customer = st.sidebar.selectbox("Selecione o cliente para recomendação", customers)
        recommend_product(model, df_ml, selected_customer)

    st.markdown('</div>', unsafe_allow_html=True)
    add_footer()

if __name__ == "__main__":
    main()
