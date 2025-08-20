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

    # 🔧 Convertir variables categóricas en numéricas
    X = pd.get_dummies(X)

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
