import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import kagglehub
import time

# ============================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================

st.set_page_config(
    page_title="E-Commerce Data Analysis & Recommender",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# ESTILOS PERSONALIZADOS
# ============================================

def add_custom_css():
    st.markdown("""
    <style>
    /* Header fixo com gradiente */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    
    /* Cards de métricas */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Tabs customizadas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Footer */
    .footer {
        margin-top: 3rem;
        padding: 2rem;
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    
    .footer a {
        color: #ffe066;
        text-decoration: none;
        font-weight: bold;
    }
    
    .footer a:hover {
        text-decoration: underline;
    }
    
    /* Alertas customizados */
    .custom-info {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .custom-success {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .custom-warning {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

add_custom_css()

# ============================================
# FUNÇÕES DE CARREGAMENTO DE DADOS
# ============================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_data(max_retries=3):
    """Carrega dados do Kaggle com retry automático"""
    dataset = "olistbr/brazilian-ecommerce"
    
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
    
    for tentativa in range(max_retries):
        try:
            data = {}
            dataset_path = kagglehub.dataset_download(dataset)
            
            for key, filename in data_files.items():
                file_path = f"{dataset_path}/{filename}"
                df = pd.read_csv(file_path, low_memory=False)
                
                # Converte colunas object para string
                for col in df.select_dtypes(include=["object"]).columns:
                    df[col] = df[col].astype(str)
                
                data[key] = df
            
            return data, None
            
        except Exception as e:
            if tentativa < max_retries - 1:
                time.sleep(2 ** tentativa)
                continue
            return None, str(e)
    
    return None, "Falha após múltiplas tentativas"

def safe_show_df(df, caption=None, height=400):
    """Exibe DataFrame com tratamento para compatibilidade Arrow"""
    df_fixed = df.copy()
    for col in df_fixed.columns:
        if df_fixed[col].dtype == "object":
            df_fixed[col] = df_fixed[col].astype(str)
    st.dataframe(df_fixed, use_container_width=True, height=height)
    if caption:
        st.caption(caption)

# ============================================
# HEADER
# ============================================

st.markdown("""
<div class="main-header">
    <h1>🛒 E-Commerce Data Analysis & Recommender</h1>
    <p>Análise completa de dados de e-commerce brasileiro com sistema de recomendação inteligente</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/shopping-cart.png", width=80)
    st.title("🎯 Navegação")
    
    opcoes = [
        "📊 Visão Geral",
        "🔍 Análise Exploratória", 
        "🔗 Análise Relacional", 
        "🤖 Sistema de Recomendação",
        "📚 Documentação"
    ]
    
    escolha = st.radio("Escolha uma seção", opcoes, label_visibility="collapsed")
    
    st.markdown("---")
    
    st.markdown("""
    <div style='padding: 1rem; background-color: #f0f2f6; border-radius: 8px;'>
        <h4 style='margin: 0 0 0.5rem 0;'>💡 Dica</h4>
        <p style='margin: 0; font-size: 0.9rem;'>
        Use as abas para navegar entre diferentes análises e visualizações.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# CARREGAMENTO DE DADOS
# ============================================

with st.spinner("🔄 Carregando dados do Kaggle..."):
    data, error = load_data()

if error or data is None:
    st.error(f"❌ Erro ao carregar dados: {error}")
    st.info("""
    **Possíveis soluções:**
    - Verifique sua conexão com a internet
    - Certifique-se de que tem acesso ao Kaggle
    - Tente recarregar a página (F5)
    """)
    st.stop()

st.success(f"✅ Dados carregados com sucesso!")

# ============================================
# SEÇÃO: VISÃO GERAL
# ============================================

if escolha == "📊 Visão Geral":
    st.header("📊 Visão Geral do Dataset")
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total de Pedidos</div>
            <div class="metric-value">{len(data['orders']):,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Clientes Únicos</div>
            <div class="metric-value">{data['customers']['customer_unique_id'].nunique():,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Produtos</div>
            <div class="metric-value">{len(data['products']):,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Vendedores</div>
            <div class="metric-value">{len(data['sellers']):,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Informações dos datasets
    st.subheader("📁 Informações dos Datasets")
    
    dataset_info = []
    for nome, df in data.items():
        dataset_info.append({
            "Dataset": nome.title(),
            "Registros": f"{len(df):,}",
            "Colunas": len(df.columns),
            "Memória (MB)": f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}"
        })
    
    info_df = pd.DataFrame(dataset_info)
    st.dataframe(info_df, use_container_width=True, hide_index=True)
    
    # Distribuição temporal de pedidos
    st.subheader("📈 Distribuição Temporal de Pedidos")
    
    orders_temp = data['orders'].copy()
    orders_temp['order_purchase_timestamp'] = pd.to_datetime(orders_temp['order_purchase_timestamp'])
    orders_temp['year_month'] = orders_temp['order_purchase_timestamp'].dt.to_period('M').astype(str)
    
    orders_by_month = orders_temp.groupby('year_month').size().reset_index(name='count')
    
    fig = px.line(
        orders_by_month, 
        x='year_month', 
        y='count',
        title='Evolução de Pedidos ao Longo do Tempo',
        labels={'year_month': 'Mês', 'count': 'Número de Pedidos'}
    )
    fig.update_traces(line_color='#667eea', line_width=3)
    fig.update_layout(
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# SEÇÃO: ANÁLISE EXPLORATÓRIA
# ============================================

elif escolha == "🔍 Análise Exploratória":
    st.header("🔍 Análise Exploratória dos Dados")
    
    tab_names = list(data.keys())
    tab_selecionada = st.selectbox("📂 Escolha o dataset para análise", tab_names)
    df = data[tab_selecionada]
    
    st.subheader(f"Dataset: {tab_selecionada.title()}")
    
    # Criar tabs para diferentes visualizações
    tab1, tab2, tab3 = st.tabs(["📋 Dados", "📊 Estatísticas", "📈 Visualizações"])
    
    with tab1:
        st.markdown("##### Pré-visualização dos Dados")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            num_rows = st.slider("Número de linhas", 5, 100, 10)
        with col2:
            show_all = st.checkbox("Mostrar tudo", value=False)
        
        if show_all:
            safe_show_df(df, height=600)
        else:
            safe_show_df(df.head(num_rows), height=400)
        
        st.markdown("##### Informações do Dataset")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de Registros", f"{len(df):,}")
        with col2:
            st.metric("Total de Colunas", len(df.columns))
        with col3:
            st.metric("Valores Nulos", f"{df.isnull().sum().sum():,}")
    
    with tab2:
        st.markdown("##### Resumo Estatístico")
        safe_show_df(df.describe(include='all'), height=400)
        
        st.markdown("##### Análise de Valores Nulos")
        null_counts = df.isnull().sum()
        null_df = pd.DataFrame({
            'Coluna': null_counts.index,
            'Valores Nulos': null_counts.values,
            'Percentual (%)': (null_counts.values / len(df) * 100).round(2)
        })
        null_df = null_df[null_df['Valores Nulos'] > 0].sort_values('Valores Nulos', ascending=False)
        
        if len(null_df) > 0:
            fig = px.bar(
                null_df, 
                x='Coluna', 
                y='Valores Nulos',
                title='Distribuição de Valores Nulos',
                color='Percentual (%)',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ Nenhum valor nulo encontrado!")
    
    with tab3:
        st.markdown("##### Visualizações")
        
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if num_cols:
            st.markdown("**Análise de Colunas Numéricas**")
            col_num = st.selectbox("Escolha uma coluna numérica", num_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    df, 
                    x=col_num,
                    title=f'Distribuição de {col_num}',
                    nbins=50
                )
                fig.update_traces(marker_color='#667eea')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(
                    df, 
                    y=col_num,
                    title=f'Box Plot de {col_num}'
                )
                fig.update_traces(marker_color='#764ba2')
                st.plotly_chart(fig, use_container_width=True)
        
        if cat_cols:
            st.markdown("**Análise de Colunas Categóricas**")
            col_cat = st.selectbox("Escolha uma coluna categórica", cat_cols)
            
            top_n = st.slider("Top N categorias", 5, 20, 10)
            value_counts = df[col_cat].value_counts().head(top_n)
            
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f'Top {top_n} - {col_cat}',
                labels={'x': col_cat, 'y': 'Frequência'}
            )
            fig.update_traces(marker_color='#667eea')
            st.plotly_chart(fig, use_container_width=True)

# ============================================
# SEÇÃO: ANÁLISE RELACIONAL
# ============================================

elif escolha == "🔗 Análise Relacional":
    st.header("🔗 Análise Relacional entre Datasets")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ Análise Geográfica", 
        "📦 Análise de Produtos",
        "💳 Análise de Pagamentos",
        "⭐ Análise de Avaliações"
    ])
    
    with tab1:
        st.subheader("Distribuição de Pedidos por Estado")
        
        customers = data['customers']
        orders = data['orders']
        
        df_merged = orders.merge(customers, on='customer_id')
        state_counts = df_merged['customer_state'].value_counts().reset_index()
        state_counts.columns = ['Estado', 'Número de Pedidos']
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.choropleth(
                state_counts,
                locations='Estado',
                locationmode='USA-states',
                color='Número de Pedidos',
                scope='south america',
                title='Mapa de Calor - Pedidos por Estado',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("##### Top 10 Estados")
            safe_show_df(state_counts.head(10), height=400)
    
    with tab2:
        st.subheader("Análise de Categorias de Produtos")
        
        order_items = data['order_items']
        products = data['products']
        translation = data['translation']
        
        df_prod = order_items.merge(
            products[['product_id', 'product_category_name']], 
            on='product_id'
        )
        
        # Traduzir categorias
        df_prod = df_prod.merge(
            translation, 
            on='product_category_name', 
            how='left'
        )
        
        df_prod['category_display'] = df_prod['product_category_name_english'].fillna(
            df_prod['product_category_name']
        )
        
        cat_counts = df_prod['category_display'].value_counts().reset_index().head(15)
        cat_counts.columns = ['Categoria', 'Vendas']
        
        fig = px.bar(
            cat_counts, 
            x='Vendas', 
            y='Categoria',
            orientation='h',
            title='Top 15 Categorias Mais Vendidas',
            color='Vendas',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Análise de preços por categoria
        st.markdown("##### Análise de Preços por Categoria (Top 10)")
        
        price_analysis = df_prod.groupby('category_display')['price'].agg([
            ('Média', 'mean'),
            ('Mediana', 'median'),
            ('Mínimo', 'min'),
            ('Máximo', 'max')
        ]).reset_index()
        price_analysis = price_analysis.nlargest(10, 'Média')
        
        fig = px.box(
            df_prod[df_prod['category_display'].isin(price_analysis['category_display'])],
            x='category_display',
            y='price',
            title='Distribuição de Preços por Categoria',
            labels={'category_display': 'Categoria', 'price': 'Preço (R$)'}
        )
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Análise de Métodos de Pagamento")
        
        payments = data['payments']
        
        col1, col2 = st.columns(2)
        
        with col1:
            payment_type = payments['payment_type'].value_counts().reset_index()
            payment_type.columns = ['Método', 'Quantidade']
            
            fig = px.pie(
                payment_type,
                values='Quantidade',
                names='Método',
                title='Distribuição de Métodos de Pagamento',
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            payment_summary = payments.groupby('payment_type').agg({
                'payment_value': ['sum', 'mean', 'count']
            }).reset_index()
            payment_summary.columns = ['Método', 'Valor Total', 'Valor Médio', 'Transações']
            
            st.markdown("##### Resumo por Método")
            safe_show_df(payment_summary, height=300)
        
        # Análise de parcelas
        st.markdown("##### Análise de Parcelas")
        installments = payments[payments['payment_installments'] > 0]['payment_installments'].value_counts().head(10).reset_index()
        installments.columns = ['Parcelas', 'Frequência']
        
        fig = px.bar(
            installments,
            x='Parcelas',
            y='Frequência',
            title='Distribuição de Número de Parcelas',
            color='Frequência',
            color_continuous_scale='Greens'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Análise de Avaliações")
        
        reviews = data['reviews']
        
        col1, col2 = st.columns(2)
        
        with col1:
            score_dist = reviews['review_score'].value_counts().sort_index().reset_index()
            score_dist.columns = ['Nota', 'Quantidade']
            
            fig = px.bar(
                score_dist,
                x='Nota',
                y='Quantidade',
                title='Distribuição de Notas de Avaliação',
                color='Nota',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            avg_score = reviews['review_score'].mean()
            total_reviews = len(reviews)
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Nota Média</div>
                <div class="metric-value">{avg_score:.2f} ⭐</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total de Avaliações</div>
                <div class="metric-value">{total_reviews:,}</div>
            </div>
            """, unsafe_allow_html=True)

# ============================================
# SEÇÃO: SISTEMA DE RECOMENDAÇÃO
# ============================================

elif escolha == "🤖 Sistema de Recomendação":
    st.header("🤖 Sistema de Recomendação de Produtos")
    
    # Preparar dados para ML
    @st.cache_data
    def prepare_ml_data(data_dict):
        customers = data_dict['customers']
        orders = data_dict['orders']
        order_items = data_dict['order_items']
        products = data_dict['products']
        
        df = orders.merge(order_items, on='order_id').merge(
            products[['product_id', 'product_category_name']], 
            on='product_id'
        )
        df = df.merge(customers[['customer_id', 'customer_unique_id']], on='customer_id')
        
        df['category_code'] = df['product_category_name'].astype('category').cat.codes
        df['customer_code'] = df['customer_unique_id'].astype('category').cat.codes
        
        X = df[['customer_code']]
        y = df['category_code']
        
        return X, y, df
    
    with st.spinner("🔄 Preparando dados para treinamento..."):
        X, y, df_ml = prepare_ml_data(data)
    
    st.success("✅ Dados preparados!")
    
    # Configurações do modelo
    st.subheader("⚙️ Configurações do Modelo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Tamanho do conjunto de teste (%)", 10, 40, 20) / 100
        n_estimators_rf = st.slider("Random Forest - Nº de árvores", 50, 200, 100, step=50)
    
    with col2:
        random_state = st.number_input("Random State", 0, 100, 42)
        n_estimators_xgb = st.slider("XGBoost - Nº de estimadores", 50, 200, 100, step=50)
    
    if st.button("🚀 Treinar Modelos", type="primary", use_container_width=True):
        with st.spinner("🔄 Treinando modelos..."):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Random Forest
            progress_bar = st.progress(0)
            st.info("🌲 Treinando Random Forest...")
            
            rf = RandomForestClassifier(n_estimators=n_estimators_rf, random_state=random_state, n_jobs=-1)
            rf.fit(X_train, y_train)
            rf_pred = rf.predict(X_test)
            progress_bar.progress(50)
            
            # XGBoost
            st.info("🚀 Treinando XGBoost...")
            xgb = XGBClassifier(
                n_estimators=n_estimators_xgb, 
                use_label_encoder=False, 
                eval_metric='mlogloss',
                random_state=random_state
            )
            xgb.fit(X_train, y_train)
            xgb_pred = xgb.predict(X_test)
            progress_bar.progress(100)
            
            # Métricas
            rf_metrics = {
                'accuracy': accuracy_score(y_test, rf_pred),
                'precision': precision_score(y_test, rf_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, rf_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, rf_pred, average='weighted', zero_division=0)
            }
            
            xgb_metrics = {
                'accuracy': accuracy_score(y_test, xgb_pred),
                'precision': precision_score(y_test, xgb_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, xgb_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, xgb_pred, average='weighted', zero_division=0)
            }
            
            # Armazenar no session_state
            st.session_state['models_trained'] = True
            st.session_state['rf_model'] = rf
            st.session_state['xgb_model'] = xgb
            st.session_state['rf_metrics'] = rf_metrics
            st.session_state['xgb_metrics'] = xgb_metrics
            st.session_state['df_ml'] = df_ml
            st.session_state['y_test'] = y_test
            st.session_state['rf_pred'] = rf_pred
            st.session_state['xgb_pred'] = xgb_pred
        
        st.success("✅ Modelos treinados com sucesso!")
        st.rerun()
    
    # Exibir resultados se modelos foram treinados
    if st.session_state.get('models_trained', False):
        st.markdown("---")
        st.subheader("📊 Resultados dos Modelos")
        
        # Comparação de métricas
        metrics_df = pd.DataFrame({
            'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Random Forest': [
                st.session_state['rf_metrics']['accuracy'],
                st.session_state['rf_metrics']['precision'],
                st.session_state['rf_metrics']['recall'],
                st.session_state['rf_metrics']['f1']
            ],
            'XGBoost': [
                st.session_state['xgb_metrics']['accuracy'],
                st.session_state['xgb_metrics']['precision'],
                st.session_state['xgb_metrics']['recall'],
                st.session_state['xgb_metrics']['f1']
            ]
        })
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Random Forest',
                x=metrics_df['Métrica'],
                y=metrics_df['Random Forest'],
                marker_color='#667eea'
            ))
            fig.add_trace(go.Bar(
                name='XGBoost',
                x=metrics_df['Métrica'],
                y=metrics_df['XGBoost'],
                marker_color='#764ba2'
            ))
            fig.update_layout(
                title='Comparação de Métricas dos Modelos',
                barmode='group',
                yaxis_title='Score',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("##### Tabela de Métricas")
            st.dataframe(
                metrics_df.style.format({
                    'Random Forest': '{:.4f}',
                    'XGBoost': '{:.4f}'
                }).highlight_max(axis=1, color='lightgreen', subset=['Random Forest', 'XGBoost']),
                use_container_width=True
            )
            
            # Melhor modelo
            rf_mean = metrics_df['Random Forest'].mean()
            xgb_mean = metrics_df['XGBoost'].mean()
            
            if rf_mean > xgb_mean:
                st.markdown("""
                <div class="custom-success">
                    <strong>🏆 Melhor Modelo: Random Forest</strong><br>
                    Score médio: {:.4f}
                </div>
                """.format(rf_mean), unsafe_allow_html=True)
                best_model = st.session_state['rf_model']
                st.session_state['best_model_name'] = 'Random Forest'
            else:
                st.markdown("""
                <div class="custom-success">
                    <strong>🏆 Melhor Modelo: XGBoost</strong><br>
                    Score médio: {:.4f}
                </div>
                """.format(xgb_mean), unsafe_allow_html=True)
                best_model = st.session_state['xgb_model']
                st.session_state['best_model_name'] = 'XGBoost'
            
            st.session_state['best_model'] = best_model
        
        # Matriz de Confusão
        st.markdown("---")
        st.subheader("📉 Matrizes de Confusão")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Random Forest")
            cm_rf = confusion_matrix(st.session_state['y_test'], st.session_state['rf_pred'])
            
            # Limitar a 10x10 para visualização
            if cm_rf.shape[0] > 10:
                top_classes = pd.Series(st.session_state['y_test']).value_counts().head(10).index
                mask = np.isin(st.session_state['y_test'], top_classes) & np.isin(st.session_state['rf_pred'], top_classes)
                cm_rf_display = confusion_matrix(
                    st.session_state['y_test'][mask], 
                    st.session_state['rf_pred'][mask]
                )
            else:
                cm_rf_display = cm_rf
            
            fig = px.imshow(
                cm_rf_display,
                text_auto=True,
                aspect='auto',
                color_continuous_scale='Blues',
                title='Top 10 Classes'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("##### XGBoost")
            cm_xgb = confusion_matrix(st.session_state['y_test'], st.session_state['xgb_pred'])
            
            # Limitar a 10x10 para visualização
            if cm_xgb.shape[0] > 10:
                mask = np.isin(st.session_state['y_test'], top_classes) & np.isin(st.session_state['xgb_pred'], top_classes)
                cm_xgb_display = confusion_matrix(
                    st.session_state['y_test'][mask], 
                    st.session_state['xgb_pred'][mask]
                )
            else:
                cm_xgb_display = cm_xgb
            
            fig = px.imshow(
                cm_xgb_display,
                text_auto=True,
                aspect='auto',
                color_continuous_scale='Purples',
                title='Top 10 Classes'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Sistema de Recomendação
        st.markdown("---")
        st.subheader("🎯 Fazer Recomendação")
        
        customers_unique = st.session_state['df_ml']['customer_unique_id'].unique()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_customer = st.selectbox(
                "Selecione o ID do Cliente",
                customers_unique,
                help="Escolha um cliente para receber recomendações personalizadas"
            )
        
        with col2:
            num_recommendations = st.slider("Número de recomendações", 3, 10, 5)
        
        if st.button("🔮 Gerar Recomendações", type="primary", use_container_width=True):
            df_ml = st.session_state['df_ml']
            best_model = st.session_state['best_model']
            
            # Obter histórico do cliente
            customer_history = df_ml[df_ml['customer_unique_id'] == selected_customer]
            
            if len(customer_history) > 0:
                customer_code = customer_history['customer_code'].iloc[0]
                
                # Prever probabilidades para todas as categorias
                if hasattr(best_model, 'predict_proba'):
                    probas = best_model.predict_proba([[customer_code]])[0]
                    top_indices = np.argsort(probas)[::-1][:num_recommendations]
                    
                    # Mapear de volta para nomes de categorias
                    category_mapping = df_ml[['category_code', 'product_category_name']].drop_duplicates()
                    category_mapping = category_mapping.set_index('category_code')['product_category_name'].to_dict()
                    
                    recommendations = []
                    for idx in top_indices:
                        if idx in category_mapping:
                            cat_name = category_mapping[idx]
                            recommendations.append({
                                'Categoria': cat_name,
                                'Confiança (%)': f"{probas[idx] * 100:.2f}%",
                                'Score': probas[idx]
                            })
                    
                    if recommendations:
                        rec_df = pd.DataFrame(recommendations)
                        
                        st.markdown("##### 🎁 Produtos Recomendados")
                        
                        # Visualização em cards
                        for i, rec in enumerate(recommendations, 1):
                            confidence = rec['Score'] * 100
                            color = '#4caf50' if confidence > 50 else '#ff9800' if confidence > 30 else '#f44336'
                            
                            st.markdown(f"""
                            <div style='padding: 1rem; background: linear-gradient(135deg, {color}22 0%, {color}11 100%); 
                                        border-left: 4px solid {color}; border-radius: 8px; margin-bottom: 0.5rem;'>
                                <strong>#{i} - {rec['Categoria']}</strong><br>
                                <span style='font-size: 1.2rem; font-weight: bold; color: {color};'>
                                    {rec['Confiança (%)']}
                                </span>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Gráfico de barras
                        fig = px.bar(
                            rec_df,
                            x='Score',
                            y='Categoria',
                            orientation='h',
                            title='Visualização das Recomendações',
                            color='Score',
                            color_continuous_scale='Viridis'
                        )
                        fig.update_layout(showlegend=False, height=300)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Histórico do cliente
                        st.markdown("##### 📜 Histórico de Compras do Cliente")
                        
                        hist_categories = customer_history['product_category_name'].value_counts().reset_index()
                        hist_categories.columns = ['Categoria', 'Quantidade']
                        hist_categories = hist_categories.head(10)
                        
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            safe_show_df(hist_categories, height=300)
                        
                        with col2:
                            fig = px.pie(
                                hist_categories,
                                values='Quantidade',
                                names='Categoria',
                                title='Distribuição de Compras Anteriores',
                                hole=0.4
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("⚠️ Não foi possível gerar recomendações para este cliente.")
                else:
                    st.error("❌ Modelo não suporta predição de probabilidades.")
            else:
                st.warning("⚠️ Cliente não encontrado no histórico.")

# ============================================
# SEÇÃO: DOCUMENTAÇÃO
# ============================================

elif escolha == "📚 Documentação":
    st.header("📚 Documentação do Sistema")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📖 Visão Geral",
        "🗂️ Estrutura dos Dados",
        "🤖 Modelos de ML",
        "💡 Como Usar"
    ])
    
    with tab1:
        st.markdown("""
        ### 🎯 Sobre o Sistema
        
        Este sistema foi desenvolvido para análise completa de dados de e-commerce brasileiro, 
        utilizando o dataset público da **Olist** disponível no Kaggle.
        
        #### 🌟 Principais Funcionalidades
        
        1. **📊 Visão Geral**
           - Métricas gerais do dataset
           - Análise temporal de pedidos
           - Informações consolidadas
        
        2. **🔍 Análise Exploratória**
           - Visualização detalhada de cada dataset
           - Estatísticas descritivas
           - Análise de valores nulos
           - Gráficos interativos
        
        3. **🔗 Análise Relacional**
           - Análise geográfica de pedidos
           - Distribuição de produtos por categoria
           - Análise de métodos de pagamento
           - Análise de avaliações dos clientes
        
        4. **🤖 Sistema de Recomendação**
           - Treinamento de modelos de Machine Learning
           - Comparação entre Random Forest e XGBoost
           - Recomendações personalizadas por cliente
           - Análise de confiança das recomendações
        
        #### 🛠️ Tecnologias Utilizadas
        
        - **Streamlit**: Interface web interativa
        - **Pandas**: Manipulação de dados
        - **Plotly**: Visualizações interativas
        - **Scikit-learn**: Modelos de Machine Learning
        - **XGBoost**: Gradient Boosting otimizado
        - **KaggleHub**: Download de datasets
        """)
    
    with tab2:
        st.markdown("""
        ### 🗂️ Estrutura dos Datasets
        
        O sistema trabalha com 9 datasets principais:
        
        #### 📦 Orders (Pedidos)
        - **order_id**: ID único do pedido
        - **customer_id**: ID do cliente
        - **order_status**: Status do pedido
        - **order_purchase_timestamp**: Data/hora da compra
        - **order_delivered_timestamp**: Data/hora da entrega
        
        #### 👥 Customers (Clientes)
        - **customer_id**: ID único do cliente
        - **customer_unique_id**: ID permanente do cliente
        - **customer_zip_code**: CEP do cliente
        - **customer_city**: Cidade
        - **customer_state**: Estado
        
        #### 📦 Order Items (Itens do Pedido)
        - **order_id**: ID do pedido
        - **order_item_id**: Número do item
        - **product_id**: ID do produto
        - **seller_id**: ID do vendedor
        - **price**: Preço do item
        - **freight_value**: Valor do frete
        
        #### 🏷️ Products (Produtos)
        - **product_id**: ID único do produto
        - **product_category_name**: Nome da categoria
        - **product_weight_g**: Peso em gramas
        - **product_length_cm**: Comprimento
        - **product_height_cm**: Altura
        - **product_width_cm**: Largura
        
        #### 💳 Payments (Pagamentos)
        - **order_id**: ID do pedido
        - **payment_sequential**: Número sequencial
        - **payment_type**: Tipo de pagamento
        - **payment_installments**: Número de parcelas
        - **payment_value**: Valor do pagamento
        
        #### ⭐ Reviews (Avaliações)
        - **review_id**: ID da avaliação
        - **order_id**: ID do pedido
        - **review_score**: Nota (1-5)
        - **review_comment_title**: Título do comentário
        - **review_comment_message**: Comentário
        
        #### 🏪 Sellers (Vendedores)
        - **seller_id**: ID único do vendedor
        - **seller_zip_code**: CEP
        - **seller_city**: Cidade
        - **seller_state**: Estado
        
        #### 🗺️ Geolocation (Geolocalização)
        - **geolocation_zip_code**: CEP
        - **geolocation_lat**: Latitude
        - **geolocation_lng**: Longitude
        - **geolocation_city**: Cidade
        - **geolocation_state**: Estado
        
        #### 🌐 Translation (Tradução)
        - **product_category_name**: Nome em português
        - **product_category_name_english**: Nome em inglês
        """)
    
    with tab3:
        st.markdown("""
        ### 🤖 Modelos de Machine Learning
        
        O sistema utiliza dois algoritmos de classificação para o sistema de recomendação:
        
        #### 🌲 Random Forest Classifier
        
        **Características:**
        - Ensemble de múltiplas árvores de decisão
        - Reduz overfitting através de bootstrap aggregating
        - Robusto a outliers e dados não balanceados
        - Fornece importância das features
        
        **Parâmetros Configuráveis:**
        - `n_estimators`: Número de árvores (50-200)
        - `random_state`: Seed para reprodutibilidade
        
        **Vantagens:**
        - Alta acurácia
        - Funciona bem com dados categóricos
        - Menos propenso a overfitting
        
        #### 🚀 XGBoost Classifier
        
        **Características:**
        - Gradient Boosting otimizado
        - Treinamento mais rápido que Random Forest
        - Regularização integrada (L1 e L2)
        - Excelente performance em competições
        
        **Parâmetros Configuráveis:**
        - `n_estimators`: Número de estimadores (50-200)
        - `eval_metric`: Métrica de avaliação (mlogloss)
        - `random_state`: Seed para reprodutibilidade
        
        **Vantagens:**
        - Performance superior em muitos casos
        - Treinamento eficiente
        - Lida bem com dados desbalanceados
        
        #### 📊 Métricas de Avaliação
        
        O sistema avalia os modelos usando 4 métricas principais:
        
        1. **Accuracy (Acurácia)**
           - Proporção de predições corretas
           - Fórmula: (VP + VN) / Total
        
        2. **Precision (Precisão)**
           - Proporção de predições positivas corretas
           - Fórmula: VP / (VP + FP)
        
        3. **Recall (Revocação)**
           - Proporção de positivos reais identificados
           - Fórmula: VP / (VP + FN)
        
        4. **F1-Score**
           - Média harmônica entre Precision e Recall
           - Fórmula: 2 * (Precision * Recall) / (Precision + Recall)
        
        #### 🎯 Sistema de Recomendação
        
        O sistema funciona em 3 etapas:
        
        1. **Preparação dos Dados**
           - Codificação de clientes e categorias
           - Criação de features numéricas
        
        2. **Treinamento**
           - Divisão treino/teste configurável
           - Treinamento paralelo de ambos os modelos
        
        3. **Recomendação**
           - Predição de probabilidades por categoria
           - Ranking das top N categorias
           - Visualização de confiança
        """)
    
    with tab4:
        st.markdown("""
        ### 💡 Como Usar o Sistema
        
        #### 1️⃣ Navegação Básica
        
        Use o menu lateral para alternar entre as seções:
        - **📊 Visão Geral**: Métricas e informações gerais
        - **🔍 Análise Exploratória**: Explore cada dataset
        - **🔗 Análise Relacional**: Relações entre dados
        - **🤖 Sistema de Recomendação**: Treine modelos e faça recomendações
        - **📚 Documentação**: Esta página
        
        #### 2️⃣ Análise Exploratória
        
        1. Selecione um dataset no dropdown
        2. Use as tabs para alternar entre:
           - **Dados**: Visualização tabular
           - **Estatísticas**: Resumo estatístico
           - **Visualizações**: Gráficos interativos
        3. Configure o número de linhas ou colunas a visualizar
        
        #### 3️⃣ Análise Relacional
        
        Explore 4 áreas principais:
        - **Geográfica**: Mapa de pedidos por estado
        - **Produtos**: Top categorias e análise de preços
        - **Pagamentos**: Métodos e distribuição de parcelas
        - **Avaliações**: Notas e satisfação dos clientes
        
        #### 4️⃣ Sistema de Recomendação
        
        **Passo a Passo:**
        
        1. **Configure os Parâmetros**
           - Tamanho do conjunto de teste (10-40%)
           - Número de árvores para Random Forest (50-200)
           - Número de estimadores para XGBoost (50-200)
           - Random state para reprodutibilidade
        
        2. **Treine os Modelos**
           - Clique em "🚀 Treinar Modelos"
           - Aguarde o treinamento (pode levar alguns minutos)
           - Observe as métricas de performance
        
        3. **Analise os Resultados**
           - Compare as métricas entre os modelos
           - Visualize as matrizes de confusão
           - Identifique o melhor modelo
        
        4. **Gere Recomendações**
           - Selecione um ID de cliente
           - Escolha quantas recomendações deseja (3-10)
           - Clique em "🔮 Gerar Recomendações"
           - Analise as categorias sugeridas e o histórico do cliente
        
        #### 💡 Dicas de Uso
        
        - **Performance**: Use cache do navegador para acelerar carregamento
        - **Visualizações**: Passe o mouse sobre os gráficos para ver detalhes
        - **Exportação**: Use Ctrl+P ou Cmd+P para exportar visualizações
        - **Comparação**: Treine múltiplas vezes com diferentes parâmetros
        - **Interpretação**: Foque em múltiplas métricas, não apenas accuracy
        
        #### ⚠️ Limitações
        
        - Dataset limitado a dados históricos da Olist
        - Recomendações baseadas apenas em categorias
        - Modelos simples sem fine-tuning avançado
        - Dados de cache expiram após 1 hora
        
        #### 🐛 Solução de Problemas
        
        **Erro ao carregar dados:**
        - Verifique sua conexão com a internet
        - Aguarde alguns minutos e tente novamente
        - Limpe o cache do navegador
        
        **Modelos não treinam:**
        - Reduza o número de estimadores
        - Aumente o tamanho do conjunto de teste
        - Verifique se há memória suficiente
        
        **Recomendações inconsistentes:**
        - Cliente pode ter histórico limitado
        - Tente outro cliente com mais compras
        - Verifique as métricas de confiança
        """)

# ============================================
# FOOTER
# ============================================

st.markdown("""
<div class="footer">
    <p style='margin: 0; font-size: 1.1rem;'>
        Desenvolvido com ❤️ por <a href="https://github.com/bcmaymonegalvao" target="_blank">Bruno Galvão</a>
    </p>
    <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>
        📅 2025 | Powered by Streamlit, Plotly & Scikit-learn
    </p>
    <p style='margin: 0.25rem 0 0 0; font-size: 0.9rem; opacity: 0.8;'>
        Dataset: Olist Brazilian E-Commerce | © Todos os direitos reservados
    </p>
</div>
""", unsafe_allow_html=True)
