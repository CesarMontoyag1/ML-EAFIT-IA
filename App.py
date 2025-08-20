import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns # Añadimos seaborn para visualizaciones más atractivas

# ====================================================================
# Configuración de la página y estilo
# ====================================================================
st.set_page_config(
    page_title="Comparación de Modelos Supervisados",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================================================================
# Generar dataset simulado (con cache para optimizar)
# ====================================================================
@st.cache_data
def generar_datos():
    """Genera un dataset simulado de clasificación con 3 clases."""
    X, y = make_classification(
        n_samples=400,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        random_state=42
    )
    columnas = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=columnas)
    df["target"] = y
    return df

# ====================================================================
# Título y descripción de la App
# ====================================================================
st.title("🌟 Comparación de Modelos Supervisados en Python")
st.markdown(
    """
    Este demo te permite explorar **KNN, Árbol de Decisión y Naive Bayes** aplicados a un dataset simulado. Usa la barra lateral para ajustar los 
    parámetros y ver cómo afectan los resultados.
    """
)

# Generar dataset
df = generar_datos()

# ====================================================================
# Análisis de Datos Exploratorio (EDA)
# ====================================================================
st.header("🔎 Análisis Exploratorio de Datos (EDA)")
with st.expander("Haz clic para ver las visualizaciones del dataset"):
    st.subheader("📊 Vista previa y estadísticas del dataset")
    st.write(df.head())
    st.write(df.describe())

    st.subheader("Distribution de Clases")
    fig, ax = plt.subplots()
    df['target'].value_counts().plot.pie(
        autopct='%1.1f%%',
        startangle=90,
        colors=sns.color_palette("viridis", 3)
    )
    ax.set_title("Distribución de la Variable Objetivo")
    ax.set_ylabel("") # Oculta la etiqueta del eje y
    st.pyplot(fig)

    st.subheader("Distribución de Características")
    fig = plt.figure(figsize=(15, 10))
    for i, col in enumerate(df.drop("target", axis=1).columns):
        plt.subplot(2, 3, i + 1)
        sns.histplot(data=df, x=col, hue="target", multiple="stack", palette="viridis")
        plt.title(f"Distribución de {col}")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Relación entre Características")
    # Usamos st.columns para un diseño más limpio
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(
            data=df,
            x="feature_0",
            y="feature_1",
            hue="target",
            palette="viridis",
            ax=ax
        )
        ax.set_title("Feature 0 vs Feature 1")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(
            data=df,
            x="feature_2",
            y="feature_3",
            hue="target",
            palette="viridis",
            ax=ax
        )
        ax.set_title("Feature 2 vs Feature 3")
        st.pyplot(fig)

# ====================================================================
# Configuración del modelo y barra lateral
# ====================================================================
st.sidebar.header("⚙️ Configuración del Modelo")

# Dividir datos
X = df.drop("target", axis=1)
y = df["target"]

test_size = st.sidebar.slider("Proporción de prueba", 0.1, 0.5, 0.3, step=0.05)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
st.sidebar.write(f"👉 Tamaño de entrenamiento: {X_train.shape[0]}")
st.sidebar.write(f"👉 Tamaño de prueba: {X_test.shape[0]}")

# Selección de modelo
modelo = st.sidebar.selectbox("Elige el modelo", ["KNN", "Árbol de Decisión", "Naive Bayes"])

# Parámetros del modelo
if modelo == "KNN":
    k = st.sidebar.slider("Número de vecinos (k)", 1, 15, 5)
    clf = KNeighborsClassifier(n_neighbors=k)

elif modelo == "Árbol de Decisión":
    max_depth = st.sidebar.slider("Profundidad máxima", 1, 10, 3)
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

elif modelo == "Naive Bayes":
    clf = GaussianNB()

# Entrenar modelo
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# ====================================================================
# Resultados y visualizaciones del modelo
# ====================================================================
st.header("📈 Resultados del Modelo")

# Mostrar exactitud
acc = accuracy_score(y_test, y_pred)
st.success(f"**Exactitud (Accuracy):** {acc:.3f}")

# Matriz de confusión
st.subheader("📌 Matriz de Confusión")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2])
disp.plot(cmap=plt.cm.Blues, ax=ax)
st.pyplot(fig)

# Visualización especial para árbol
if modelo == "Árbol de Decisión":
    st.subheader("🌳 Visualización del Árbol de Decisión")
    fig, ax = plt.subplots(figsize=(15, 8))
    plot_tree(
        clf,
        filled=True,
        feature_names=X.columns,
        class_names=[str(c) for c in np.unique(y)],
        ax=ax
    )
    plt.tight_layout()
    st.pyplot(fig)
