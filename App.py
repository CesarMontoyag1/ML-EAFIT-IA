import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# =============================
# Generar dataset simulado
# =============================
@st.cache_data
def generar_datos(n_samples=400, n_features=6):
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        random_state=42
    )
    columnas = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=columnas)
    df["target"] = y
    return df

# =============================
# App Streamlit
# =============================
st.title("🌽 Clasificación de Cultivos con Árbol de Decisión")

uploaded_file = st.file_uploader("📂 Sube un archivo CSV (ej: con columna 'cultivo')", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset cargado exitosamente")
else:
    df = generar_datos()
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

    # 🔧 Convertir variables categóricas a numéricas
    X = pd.get_dummies(X)

    # =============================
    # División de datos
    # =============================
    test_size = st.slider("Proporción de prueba", 0.1, 0.5, 0.3, step=0.05)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    st.write(f"👉 Tamaño entrenamiento: {X_train.shape[0]} | Tamaño prueba: {X_test.shape[0]}")

    # =============================
    # Parámetros del árbol
    # =============================
    criterion = st.selectbox("Criterio de división", ["gini", "entropy", "log_loss"])
    max_depth = st.slider("Profundidad máxima", 1, 20, 5)
    min_samples_split = st.slider("Mínimo muestras para dividir", 2, 10, 2)
    min_samples_leaf = st.slider("Mínimo muestras por hoja", 1, 10, 1)

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

    # Matriz de confusión
    st.subheader("📌 Matriz de Confusión")
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm, display_labels=np.unique(y)).plot(ax=ax, cmap="Blues", colorbar=False)
    st.pyplot(fig)

    # =============================
    # Visualización del Árbol
    # =============================
    st.subheader("🌳 Visualización del Árbol de Decisión")
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_tree(clf, filled=True, feature_names=X.columns, class_names=[str(c) for c in np.unique(y)])
    st.pyplot(fig)
