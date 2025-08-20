import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# =============================
# Configuración general
# =============================
st.set_page_config(page_title="Árbol de Decisión - Clasificación de Cultivos", layout="wide")

# =============================
# Generar dataset simulado (si no se sube CSV)
# =============================
@st.cache_data
def generar_datos(n_samples=400, n_features=6, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        random_state=random_state
    )
    columnas = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=columnas)
    df["target"] = y
    return df

# =============================
# Sidebar: parámetros
# =============================
st.sidebar.header("⚙️ Parámetros del modelo")

# Parámetros del dataset simulado
n_samples = st.sidebar.slider("Número de muestras (simulado)", 200, 1000, 400, step=50)
n_features = st.sidebar.slider("Número de características (simulado)", 6, 12, 6)
test_size = st.sidebar.slider("Proporción de prueba", 0.1, 0.5, 0.3, step=0.05)

# Parámetros del árbol de decisión
criterion = st.sidebar.selectbox("Criterio de división", ["gini", "entropy", "log_loss"])
max_depth = st.sidebar.slider("Profundidad máxima", 1, 15, 3)
min_samples_split = st.sidebar.slider("Mínimo de muestras para dividir", 2, 20, 2)
min_samples_leaf = st.sidebar.slider("Mínimo de muestras en hoja", 1, 20, 1)

# =============================
# Cargar dataset o generar
# =============================
st.title("🌽🌾🥔 Clasificación de Cultivos con Árbol de Decisión")

uploaded_file = st.file_uploader("Sube un archivo CSV (ej: con columna 'cultivo')", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset cargado exitosamente")
else:
    df = generar_datos(n_samples=n_samples, n_features=n_features)
    st.info("ℹ️ Usando dataset simulado")

st.subheader("📊 Vista previa del dataset")
st.write(df.head())

# =============================
# Selección de la variable objetivo
# =============================
columnas = df.columns.tolist()
target_col = st.selectbox("Selecciona la columna objetivo (ej: 'cultivo')", columnas, index=len(columnas)-1)

if target_col not in df.columns:
    st.error("❌ No se encontró la columna objetivo seleccionada.")
else:
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # =============================
    # División de datos
    # =============================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    st.write(f"👉 Tamaño entrenamiento: {X_train.shape[0]} | Tamaño prueba: {X_test.shape[0]}")

    # =============================
    # Entrenar modelo
    # =============================
    clf = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # =============================
    # Resultados
    # =============================
    acc = accuracy_score(y_test, y_pred)
    st.subheader("📈 Resultados del Modelo")
    st.write(f"**Exactitud (Accuracy):** {acc:.3f}")

    st.text("📌 Reporte de Clasificación:")
    st.text(classification_report(y_test, y_pred))

    # =============================
    # Matriz de confusión
    # =============================
    st.subheader("📌 Matriz de Confusión")
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm, display_labels=np.unique(y)).plot(ax=ax, cmap="Blues", colorbar=False)
    st.pyplot(fig)

    # =============================
    # Visualización de dispersión (siempre que haya al menos 2 features numéricas)
    # =============================
    if X.shape[1] >= 2:
        st.subheader("🌐 Visualización de Clases (2 primeras características)")
        fig, ax = plt.subplots()
        scatter = ax.scatter(
            X_test.iloc[:, 0], X_test.iloc[:, 1],
            c=pd.Categorical(y_pred).codes, cmap="viridis", alpha=0.7, edgecolors="k"
        )
        legend1 = ax.legend(*scatter.legend_elements(), title="Clases")
        ax.add_artist(legend1)
        plt.xlabel(X.columns[0])
        plt.ylabel(X.columns[1])
        st.pyplot(fig)

    # =============================
    # Visualización del Árbol
    # =============================
    st.subheader("🌳 Visualización del Árbol de Decisión")
    fig, ax = plt.subplots(figsize=(14, 6))
    plot_tree(
        clf,
        filled=True,
        feature_names=X.columns,
        class_names=[str(c) for c in np.unique(y)],
        rounded=True
    )
    st.pyplot(fig)
