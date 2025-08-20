import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from pandas.plotting import scatter_matrix

# -------------------------
# Configuración de la página
# -------------------------
st.set_page_config(
    page_title="ML EAFIT - Comparador de Modelos",
    page_icon="🌟",
    layout="wide"
)

# CSS simple para embellecer
st.markdown(
    """
    <style>
    .title {font-size:40px; font-weight:700;}
    .big-font {font-size:20px !important;}
    .stApp {
        background: linear-gradient(180deg, #0f172a 0%, #071029 100%);
        color: #e6eef8;
    }
    .card {background: rgba(255,255,255,0.03); padding: 12px; border-radius: 12px}
    </style>
    """,
    unsafe_allow_html=True
)

# =============================
# Generar dataset simulado
# =============================
@st.cache_data
def generar_datos(n_samples=400, n_features=6, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features - 2),
        n_redundant=0,
        n_classes=3,
        random_state=random_state
    )
    columnas = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=columnas)
    df["target"] = y
    return df

# =============================
# Barra lateral: controles globales
# =============================
st.sidebar.header("Controles")
n_samples = st.sidebar.slider("Número de muestras", 300, 1200, 400, step=50)
n_features = st.sidebar.slider("Número de features", 6, 12, 6)
random_state = st.sidebar.number_input("Random state", value=42, step=1)

# =============================
# Opción de carga de dataset
# =============================
subido = st.file_uploader("📂 Sube tu archivo CSV (opcional)", type=["csv"])

if subido is not None:
    df = pd.read_csv(subido)
    st.success("✅ Dataset cargado correctamente desde archivo.")
else:
    with st.spinner("Generando dataset simulado... 🌱"):
        df = generar_datos(n_samples=n_samples, n_features=n_features, random_state=int(random_state))

# Top header
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("<div class='title'>🌟 Comparador de Modelos Supervisados</div>", unsafe_allow_html=True)
    st.markdown("**KNN, Árbol de Decisión y Naive Bayes** — demo interactiva con EDA integrado.")
with col2:
    st.image("https://static.streamlit.io/examples/dice.jpg", width=120)

# Descargar dataset
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Descargar dataset (CSV)", data=csv, file_name="dataset_simulado.csv", mime='text/csv')

# =============================
# Tabs: EDA / Dataset / Model
# =============================
tabs = st.tabs(["EDA 📊", "Dataset 📋", "Modelado 🤖"])

# -----------------------------
# TAB: Dataset
# -----------------------------
with tabs[1]:
    st.subheader("📋 Vista previa del dataset")
    st.dataframe(df.head(200))
    with st.expander("🧾 Estadísticas descriptivas"):
        st.write(df.describe().T)

# -----------------------------
# TAB: EDA
# -----------------------------
with tabs[0]:
    st.subheader("🔍 Análisis exploratorio de datos (EDA)")

    # Resumen rápido
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Muestras", df.shape[0])
    with col_b:
        st.metric("Features", df.shape[1] - 1)
    with col_c:
        st.metric("Clases", df['target'].nunique())

    # Distribución de clases - Pie chart
    if "target" in df.columns:
        st.markdown("**Distribución de la variable objetivo**")
        fig1, ax1 = plt.subplots(figsize=(4, 4))
        counts = df['target'].value_counts().sort_index()
        ax1.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
        ax1.set_aspect('equal')
        st.pyplot(fig1)

    # Histograma de features (grid)
    st.markdown("**Histogramas por feature**")
    features = [c for c in df.columns if c != 'target']
    n_plots = len(features)
    cols = min(3, n_plots)
    rows = int(np.ceil(n_plots / cols))
    fig2, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = axes.ravel()
    for i, f in enumerate(features):
        axes[i].hist(df[f], bins=20, alpha=0.9)
        axes[i].set_title(f)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    st.pyplot(fig2)

    # Correlation heatmap
    if len(features) > 1:
        st.markdown("**Mapa de correlaciones**")
        corr = df[features].corr()
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        cax = ax3.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        ax3.set_xticks(range(len(features)))
        ax3.set_xticklabels(features, rotation=45, ha='right')
        ax3.set_yticks(range(len(features)))
        ax3.set_yticklabels(features)
        fig3.colorbar(cax, ax=ax3, fraction=0.046, pad=0.04)
        st.pyplot(fig3)

    # Scatter matrix (solo primeras 4 features para no saturar)
    if len(features) > 1:
        st.markdown("**Scatter matrix (primeras 4 features)**")
        to_plot = features[:4]
        fig4 = scatter_matrix(df[to_plot], figsize=(10, 10), diagonal='kde')
        st.pyplot(plt.gcf())

# -----------------------------
# TAB: Modelado
# -----------------------------
with tabs[2]:
    st.subheader("🤖 Entrenamiento y comparación de modelos")

    if "target" not in df.columns:
        st.error("❌ El dataset debe contener una columna llamada 'target' para realizar el modelado.")
    else:
        # Separación de datos
        X = df.drop("target", axis=1)
        y = df["target"]

        test_size = st.slider("Proporción de prueba", 0.1, 0.5, 0.3, step=0.05, key='test_size')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        st.write(f"👉 Tamaño entrenamiento: {X_train.shape[0]} | Tamaño prueba: {X_test.shape[0]}")

        # Selección de modelo
        modelo = st.selectbox("Elige el modelo", ["KNN", "Árbol de Decisión", "Naive Bayes"]) 

        # Parámetros del modelo en la barra lateral (mejor UX)
        st.sidebar.markdown("---")
        st.sidebar.subheader("Parámetros del modelo")
        if modelo == "KNN":
            k = st.sidebar.slider("Número de vecinos (k)", 1, 25, 5)
            clf = KNeighborsClassifier(n_neighbors=k)
        elif modelo == "Árbol de Decisión":
            max_depth = st.sidebar.slider("Profundidad máxima", 1, 12, 3)
            clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        else:
            clf = GaussianNB()

        # Entrenar y predecir
        with st.spinner("Entrenando el modelo..."):
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

        # Métricas
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Presentar métricas en columnas
        m1, m2, m3 = st.columns(3)
        m1.metric("Exactitud (accuracy)", f"{acc:.3f}")
        macro_f1 = report['macro avg']['f1-score']
        macro_recall = report['macro avg']['recall']
        m2.metric("F1 (macro)", f"{macro_f1:.3f}")
        m3.metric("Recall (macro)", f"{macro_recall:.3f}")

        # Matriz de confusión y reporte
        colm1, colm2 = st.columns([1, 1])
        with colm1:
            st.subheader("📌 Matriz de Confusión")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
            ConfusionMatrixDisplay(cm).plot(ax=ax_cm, cmap="Blues", colorbar=False)
            st.pyplot(fig_cm)

        with colm2:
            st.subheader("🧾 Reporte de clasificación")
            st.dataframe(pd.DataFrame(report).transpose())

        # Visualización de dispersión (2 primeras features)
        st.subheader("🌐 Visualización de clases (2 primeras características)")
        fig_sc, ax_sc = plt.subplots(figsize=(6, 4))
        scatter = ax_sc.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_pred, cmap="viridis", alpha=0.85, edgecolors="k")
        legend1 = ax_sc.legend(*scatter.legend_elements(), title="Clases")
        ax_sc.add_artist(legend1)
        ax_sc.set_xlabel(X.columns[0])
        ax_sc.set_ylabel(X.columns[1])
        st.pyplot(fig_sc)

        # Visualización árbol si aplica
        if modelo == "Árbol de Decisión":
            st.subheader("🌳 Visualización del Árbol de Decisión")
            fig_tree, ax_tree = plt.subplots(figsize=(14, 7))
            plot_tree(clf, filled=True, feature_names=X.columns, class_names=[str(c) for c in np.unique(y)])
            st.pyplot(fig_tree)

        # Mostrar algunas observaciones incorrectas para analizar errores
        st.subheader("🔎 Ejemplos de predicciones erradas (hasta 10)")
        errors = X_test[y_test != y_pred].copy()
        if not errors.empty:
            errors['y_true'] = y_test[y_test != y_pred].values
            errors['y_pred'] = y_pred[y_test != y_pred]
            st.dataframe(errors.head(10))
        else:
            st.write("✅ No se encontraron errores en el set de prueba.")

        # Pie de página
        st.markdown("---")
        st.markdown("Proyecto desplegado desde: **CesarMontoyag1/ML-EAFIT-IA**")

# FIN
