import os
import streamlit as st
import pandas as pd
import plotly.express as px

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Reclamações Porto Alegre",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Painel de Reclamações de Consumo — Porto Alegre/RS")
st.markdown("Dados: Consumidor.gov.br | Jan/2023 – Jul/2025")

# =========================
# CARREGAR DADOS
# =========================
PASTA = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def carregar_dados():
    df    = pd.read_csv(f"{PASTA}/dataset_tratado.zip", encoding="utf-8-sig")
    df_ia = pd.read_csv(f"{PASTA}/amostra_com_ia.csv",  encoding="utf-8-sig")
    return df, df_ia

df, df_ia = carregar_dados()

if "resolvida" not in df.columns:
    df["resolvida"] = df["situacao"].apply(
        lambda x: 1 if "Finalizada avaliada" in str(x) else 0
    )

df["data_abertura"] = pd.to_datetime(df["data_abertura"], errors="coerce")
df["ano_mes"] = df["data_abertura"].dt.to_period("M").astype(str)

# =========================
# MÉTRICAS GERAIS
# =========================
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

col1.metric("📋 Total de Reclamações", f"{len(df):,}".replace(",", "."))
col2.metric("✅ Taxa de Resolução", f"{df['resolvida'].mean():.1%}")

if "tempo_resposta" in df.columns:
    col3.metric("⏱️ Tempo Médio de Resposta", f"{df['tempo_resposta'].dropna().mean():.1f} dias")

if "nota_do_consumidor" in df.columns:
    col4.metric("⭐ Nota Média", f"{df['nota_do_consumidor'].dropna().mean():.2f}")

st.markdown("---")

# =========================
# VOLUME + TAXA RESOLUÇÃO
# =========================
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("📅 Volume de Reclamações por Mês")
    vol = df.groupby("ano_mes").size().reset_index(name="total").sort_values("ano_mes")
    fig1 = px.line(vol, x="ano_mes", y="total", markers=True,
                   labels={"ano_mes": "Mês", "total": "Reclamações"},
                   color_discrete_sequence=["#1f77b4"])
    fig1.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig1, width='stretch')

with col_b:
    st.subheader("✅ Taxa de Resolução por Mês")
    taxa = df.groupby("ano_mes")["resolvida"].mean().reset_index()
    taxa["taxa_pct"] = taxa["resolvida"] * 100
    taxa = taxa.sort_values("ano_mes")
    fig2 = px.bar(taxa, x="ano_mes", y="taxa_pct",
                  labels={"ano_mes": "Mês", "taxa_pct": "Taxa (%)"},
                  color_discrete_sequence=["#2ca02c"])
    fig2.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig2, width='stretch')

# =========================
# RANKING EMPRESAS + ASSUNTOS
# =========================
col_c, col_d = st.columns(2)

with col_c:
    st.subheader("🏢 Top 20 Empresas com Mais Reclamações")
    top_emp = df["nome_fantasia"].value_counts().head(20).reset_index()
    top_emp.columns = ["empresa", "total"]
    fig3 = px.bar(top_emp.sort_values("total"), x="total", y="empresa",
                  orientation="h",
                  labels={"total": "Reclamações", "empresa": ""},
                  color_discrete_sequence=["#d62728"])
    st.plotly_chart(fig3, width='stretch')

with col_d:
    st.subheader("⚠️ Principais Assuntos")
    top_ass = df["assunto"].value_counts().head(20).reset_index()
    top_ass.columns = ["assunto", "total"]
    fig4 = px.bar(top_ass.sort_values("total"), x="total", y="assunto",
                  orientation="h",
                  labels={"total": "Reclamações", "assunto": ""},
                  color_discrete_sequence=["#ff7f0e"])
    st.plotly_chart(fig4, width='stretch')

# =========================
# TEMPO RESPOSTA + SEGMENTO
# =========================
col_e, col_f = st.columns(2)

with col_e:
    st.subheader("⏱️ Tempo Médio de Resposta por Segmento")
    tempo_seg = (df.groupby("segmento_de_mercado")["tempo_resposta"]
                 .mean().reset_index().dropna()
                 .sort_values("tempo_resposta", ascending=False).head(10))
    fig5 = px.bar(tempo_seg.sort_values("tempo_resposta"),
                  x="tempo_resposta", y="segmento_de_mercado", orientation="h",
                  labels={"tempo_resposta": "Dias", "segmento_de_mercado": ""},
                  color_discrete_sequence=["#9467bd"])
    st.plotly_chart(fig5, width='stretch')

with col_f:
    st.subheader("🏭 Reclamações por Segmento")
    seg = df["segmento_de_mercado"].value_counts().head(8).reset_index()
    seg.columns = ["segmento", "total"]
    fig6 = px.pie(seg, values="total", names="segmento", hole=0.4)
    st.plotly_chart(fig6, width='stretch')

# =========================
# RESULTADO ML
# =========================
st.markdown("---")
st.subheader("🤖 Resultado do Modelo de Machine Learning")

col_g, col_h = st.columns(2)

with col_g:
    st.markdown("**Acurácia:** `62.94%` | **Algoritmo:** Random Forest")
    df_imp = pd.DataFrame({
        "Feature": ["segmento_cod", "assunto_cod", "area_cod", "sexo_cod", "uf_cod"],
        "Importância": [0.408, 0.359, 0.184, 0.050, 0.000]
    }).sort_values("Importância")
    fig7 = px.bar(df_imp, x="Importância", y="Feature", orientation="h",
                  labels={"Importância": "Peso no modelo", "Feature": ""},
                  color_discrete_sequence=["#17becf"])
    st.plotly_chart(fig7, width='stretch')

with col_h:
    st.markdown("**Distribuição do Target**")
    dist = df["resolvida"].value_counts().reset_index()
    dist.columns = ["classe", "total"]
    dist["classe"] = dist["classe"].map({1: "Resolvida", 0: "Não Resolvida"})
    fig8 = px.pie(dist, values="total", names="classe", hole=0.4,
                  color_discrete_sequence=["#2ca02c", "#d62728"])
    st.plotly_chart(fig8, width='stretch')

# =========================
# ANÁLISE IA
# =========================
st.markdown("---")
st.subheader("🧠 Análise da IA — Amostra de Reclamações")

col_i, col_j = st.columns(2)

with col_i:
    if "categoria_ia" in df_ia.columns:
        st.markdown("**Categorias identificadas pela IA**")
        cat = df_ia["categoria_ia"].value_counts().reset_index()
        cat.columns = ["categoria", "total"]
        fig9 = px.pie(cat, values="total", names="categoria", hole=0.4)
        st.plotly_chart(fig9, width='stretch')

with col_j:
    if "sentimento_ia" in df_ia.columns:
        st.markdown("**Sentimento identificado pela IA**")
        sent = df_ia["sentimento_ia"].value_counts().reset_index()
        sent.columns = ["sentimento", "total"]
        fig10 = px.bar(sent, x="sentimento", y="total",
                       labels={"total": "Quantidade", "sentimento": ""},
                       color_discrete_sequence=["#d62728", "#ff7f0e", "#bcbd22"])
        st.plotly_chart(fig10, width='stretch')

# =========================
# TABELA AMOSTRA IA
# =========================
st.markdown("---")
st.subheader("📄 Amostra Analisada pela IA")
cols = [c for c in ["assunto", "problema", "resumo_ia", "categoria_ia", "sentimento_ia", "urgencia_ia"] if c in df_ia.columns]
st.dataframe(df_ia[cols], width='stretch')
