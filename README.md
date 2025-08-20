# ML-EAFIT-IA

# Modelos de Aprendizaje Supervisado y No Supervisado en Python

Este repositorio tiene como objetivo explicar y mostrar ejemplos básicos de los **modelos de Machine Learning supervisados y no supervisados** en Python.  
A continuación, se presenta una descripción general de cada enfoque.

---

## 📌 Aprendizaje Supervisado

El **aprendizaje supervisado** es un enfoque en el que el modelo se entrena utilizando un conjunto de datos que incluye tanto las **entradas (features)** como las **salidas esperadas (etiquetas o labels)**.  
El objetivo principal es que el modelo aprenda a **predecir la salida correcta** para nuevos datos a partir de la experiencia previa.

### 🔹 Características principales
- Requiere datos etiquetados.
- Se enfoca en aprender la relación entre variables de entrada y salida.
- Se utiliza principalmente para:
  - **Clasificación**: asignar categorías (ejemplo: detectar spam/no spam).
  - **Regresión**: predecir valores numéricos (ejemplo: precio de una casa).

### 🔹 Ejemplos de algoritmos supervisados
- Regresión Lineal y Logística  
- Árboles de Decisión  
- Random Forest  
- Máquinas de Soporte Vectorial (SVM)  
- Redes Neuronales Artificiales  

---

## 📌 Aprendizaje No Supervisado

El **aprendizaje no supervisado** trabaja con conjuntos de datos que **no tienen etiquetas**.  
El modelo intenta **encontrar patrones, estructuras ocultas o relaciones** dentro de los datos sin conocimiento previo de las salidas.

### 🔹 Características principales
- No requiere datos etiquetados.
- Se utiliza para explorar y analizar datos.
- Comúnmente aplicado en:
  - **Clustering (agrupamiento)**: agrupar elementos similares (ejemplo: segmentación de clientes).
  - **Reducción de dimensionalidad**: simplificar datos manteniendo la mayor parte de la información (ejemplo: PCA).

### 🔹 Ejemplos de algoritmos no supervisados
- K-Means  
- DBSCAN  
- PCA (Análisis de Componentes Principales)  
- t-SNE  

---

## 📊 Diferencias entre Supervisado y No Supervisado

| Aspecto              | Aprendizaje Supervisado ✅ | Aprendizaje No Supervisado ❌ |
|-----------------------|----------------------------|-------------------------------|
| Datos necesarios      | Etiquetados               | No etiquetados                |
| Objetivo              | Predecir resultados       | Encontrar patrones            |
| Ejemplo común         | Clasificación de correos  | Segmentación de clientes      |
| Tipo de salida        | Categorías o valores      | Grupos o representaciones     |

---

## 🚀 Python y Machine Learning

En Python, estos modelos suelen implementarse con librerías como:
- **scikit-learn** → Modelos de ML
- **pandas** → Manejo de datos
- **numpy** → Cálculos numéricos
- **matplotlib / seaborn** → Visualización

---

## ✨ Conclusión

- El **aprendizaje supervisado** se centra en **predecir resultados conocidos** a partir de datos de entrenamiento con etiquetas.  
- El **aprendizaje no supervisado** busca **patrones ocultos** cuando no se cuenta con etiquetas.  

Ambos enfoques son fundamentales en Machine Learning y se aplican en diferentes escenarios según la naturaleza de los datos y el problema a resolver.
