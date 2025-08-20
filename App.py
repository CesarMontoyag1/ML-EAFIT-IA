import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# =============================
# Generar dataset simulado
# =============================
@st.cache_data
def generar_datos():
    X, y = make_classification(
        n_samples=300,        # mínimo 300 muestras
        n_features=6,         # mínimo 6 columnas
        n_informative=4,
        n_redundant=0,
        n_classes=3,          # 3 clases para más variedad
        random_state=42
    )
    columnas = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=columnas)
    df["target"] = y
    return df

# =============================
# App Streamlit
# =============================
st.title("Modelos Supervisados: KNN y Árbol de Decisión 🌱")

# Generar dataset
df = generar_datos()
st.subheader("📊 Vista previa del dataset")
st.write(df.head())

# Dividir datos
X = df.drop("target", axis=1)
y = df["target"]

test_size = st.slider("Proporción de prueba", 0.1, 0.5, 0.3, step=0.05)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

st.write(f"👉 Tamaño entrenamiento: {X_train.shape[0]} | Tamaño prueba: {X_test.shape[0]}")

# Selección de modelo
modelo = st.selectbox("Elige el modelo", ["KNN", "Árbol de Decisión"])

if modelo == "KNN":
    k = st.slider("Número de vecinos (k)", 1, 15, 5)
    clf = KNeighborsClassifier(n_neighbors=k)
elif modelo == "Árbol de Decisión":
    max_depth = st.slider("Profundidad máxima", 1, 10, 3)
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)

# Entrenar modelo
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Métricas
acc = accuracy_score(y_test, y_pred)
st.subheader("📈 Resultados")
st.write(f"**Exactitud (Accuracy):** {acc:.3f}")

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
ConfusionMatrixDisplay(cm).plot(ax=ax, cmap="Blues", colorbar=False)
st.pyplot(fig)

# Visualización especial para árbol
if modelo == "Árbol de Decisión":
    st.subheader("🌳 Visualización del Árbol")
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_tree(clf, filled=True, feature_names=X.columns, class_names=[str(c) for c in np.unique(y)])
    st.pyplot(fig)
