"""
sprint3.py
Sprint 3 - Machine Learning
Genera 3 modelos:
A) Clasificación: producto TOP (top20% ventas)
B) Regresión: predecir cantidad total vendida por producto
C) Clasificación por venta: venta "grande" o "chica"

Requisitos:
- Archivos en ../db/: productos.xlsx, detalle_venta.xlsx, ventas.xlsx, clientes.xlsx
- Guardará gráficos en ./graficos/
- Python packages: pandas, numpy, matplotlib, seaborn, scikit-learn, joblib (opcional)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# --------------------------
# RUTAS Y Utilidades
# --------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # carpeta sprint3
DATA_DIR = os.path.join(BASE_DIR, "..", "db")          # carpeta db
GRAFICOS_DIR = os.path.join(BASE_DIR, "graficos")      # carpeta dentro de sprint3
MODELS_DIR = os.path.join(BASE_DIR, "models")          # carpeta dentro de sprint3

os.makedirs(GRAFICOS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def cargar_excel(nombre):
    path = os.path.join(DATA_DIR, nombre)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encuentra {path}")
    return pd.read_excel(path)

def guardar_figura(fig, nombre_archivo):
    ruta = os.path.join(GRAFICOS_DIR, nombre_archivo)
    fig.tight_layout()
    fig.savefig(ruta, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Guardado: {ruta}")

def guardar_modelo(modelo, nombre_archivo):
    ruta = os.path.join(MODELS_DIR, nombre_archivo)
    joblib.dump(modelo, ruta)
    print(f"Modelo guardado: {ruta}")

# --------------------------
# CARGA DE DATOS
# --------------------------
productos = cargar_excel("productos.xlsx")           # id_producto, nombre_producto, categoria, precio_unitario
detalle = cargar_excel("detalle_ventas.xlsx")        # id_venta, id_producto, nombre_producto, cantidad, precio_unitario, importe
ventas = cargar_excel("ventas.xlsx")                 # id_venta, fecha, id_cliente, nombre_cliente, email, medio_pago
clientes = cargar_excel("clientes.xlsx")            # id_cliente, nombre_cliente, email, ciudad, fecha_alta

# --------------------------
# PREPARACIONES COMUNES
# --------------------------
# Normalizar nombres de columnas por si hay espacios/Mayúsculas
productos.columns = [c.strip() for c in productos.columns]
detalle.columns = [c.strip() for c in detalle.columns]
ventas.columns = [c.strip() for c in ventas.columns]
clientes.columns = [c.strip() for c in clientes.columns]

# Asegurarnos tipos
detalle['cantidad'] = pd.to_numeric(detalle['cantidad'], errors='coerce').fillna(0)
detalle['precio_unitario'] = pd.to_numeric(detalle['precio_unitario'], errors='coerce').fillna(0)
detalle['importe'] = pd.to_numeric(detalle.get('importe', detalle['cantidad'] * detalle['precio_unitario']), errors='coerce').fillna(0)

# Creamos ventas agregadas por producto
ventas_producto = detalle.groupby('id_producto').agg(
    cantidad_total=('cantidad', 'sum'),
    importe_total=('importe', 'sum'),
    ventas_registros=('id_venta', 'nunique')
).reset_index()

# Unimos con productos para features
productos_ml = productos.merge(ventas_producto, how='left', on='id_producto').fillna({
    'cantidad_total': 0,
    'importe_total': 0,
    'ventas_registros': 0
})

# Añadir columna precio_unitario si falta en productos
if 'precio_unitario' not in productos_ml.columns:
    productos_ml['precio_unitario'] = productos_ml.get('precio', 0)

# Feature: ventas promedio por venta (si hay registros)
productos_ml['cantidad_promedio_por_venta'] = productos_ml.apply(
    lambda r: r['cantidad_total'] / r['ventas_registros'] if r['ventas_registros'] > 0 else 0, axis=1
)

# --------------------------
# MODELO A: Clasificación - Producto TOP (Top 20% por cantidad_total)
# --------------------------
def modelo_A_clasificacion_top(productos_df):
    df = productos_df.copy()

    # Target: top 20% en cantidad_total
    umbral = df['cantidad_total'].quantile(0.80)
    df['es_top'] = (df['cantidad_total'] >= umbral).astype(int)
    print(f"Umbral top20% (cantidad_total): {umbral}")

    # Features: precio_unitario, categoria (one-hot), cantidad_promedio_por_venta, importe_total
    X = df[['precio_unitario', 'cantidad_promedio_por_venta', 'importe_total', 'ventas_registros']].copy()
    # One-hot categoria
    if 'categoria' in df.columns:
        cats = pd.get_dummies(df['categoria'], prefix='cat')
        X = pd.concat([X, cats], axis=1)
    y = df['es_top']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Modelo: RandomForestClassifier para robustez
    modelo = RandomForestClassifier(n_estimators=200, random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print("Modelo A - Accuracy:", acc)
    print("Modelo A - Confusion matrix:\n", cm)

    # Guardar modelo
    guardar_modelo(modelo, "modelo_A_clasificador_top20.joblib")

    # Gráficos: matriz de confusión
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title("A: Matriz de confusión - Clasificador Top20%")
    guardar_figura(fig, "A_matriz_confusion_top20.png")

    # Feature importances
    importances = pd.Series(modelo.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
    fig2, ax2 = plt.subplots(figsize=(8,4))
    importances.plot.bar(ax=ax2)
    ax2.set_title("A: Importancia de features (Top)")
    guardar_figura(fig2, "A_importancias_features.png")

    # Guardar tabla resumen de resultados
    resumen = pd.DataFrame({
        'accuracy': [acc],
        'umbral_top20': [umbral],
        'n_productos': [len(df)]
    })
    resumen.to_csv(os.path.join(GRAFICOS_DIR, "A_resumen.csv"), index=False)

    return {'modelo': modelo, 'accuracy': acc, 'cm': cm, 'umbral': umbral}

# --------------------------
# MODELO B: Regresión - Predecir cantidad_total vendida por producto
# --------------------------
def modelo_B_regresion(productos_df):
    df = productos_df.copy()

    # Target: cantidad_total
    y = df['cantidad_total']
    # Features: precio_unitario, categoria (one-hot), ventas_registros, importe_total, cantidad_promedio_por_venta
    X = df[['precio_unitario', 'ventas_registros', 'importe_total', 'cantidad_promedio_por_venta']].copy()
    if 'categoria' in df.columns:
        cats = pd.get_dummies(df['categoria'], prefix='cat')
        X = pd.concat([X, cats], axis=1)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Pipeline: StandardScaler + RandomForestRegressor (compara con LinearRegression)
    scaler = StandardScaler()
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    print("Modelo B - RandomForest MSE:", mse_rf)

    # Linear baseline
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    print("Modelo B - LinearRegression MSE:", mse_lr)

    # Guardar el mejor (según MSE)
    mejor_modelo = rf if mse_rf <= mse_lr else lr
    nombre_modelo = "modelo_B_regresor_rf.joblib" if mejor_modelo is rf else "modelo_B_regresor_lr.joblib"
    guardar_modelo(mejor_modelo, nombre_modelo)

    # Gráfico: real vs predicho (para rf)
    fig, ax = plt.subplots(figsize=(6,5))
    ax.scatter(y_test, y_pred_rf, alpha=0.6)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # linea ideal
    ax.set_xlabel("Real (cantidad_total)")
    ax.set_ylabel("Predicho")
    ax.set_title("B: Real vs Predicho (RandomForest)")
    guardar_figura(fig, "B_real_vs_predicho_rf.png")

    # Guardar métricas
    resumen = pd.DataFrame({
        'mse_rf': [mse_rf],
        'mse_lr': [mse_lr],
        'n_productos': [len(df)]
    })
    resumen.to_csv(os.path.join(GRAFICOS_DIR, "B_resumen.csv"), index=False)

    return {'mejor_modelo': mejor_modelo, 'mse_rf': mse_rf, 'mse_lr': mse_lr}

# --------------------------
# MODELO C: Clasificación por venta - venta "grande" o "chica"
# --------------------------
def modelo_C_clasificacion_ventas(detalle_df, ventas_df, productos_df):
    # Trabajamos a nivel de registro detalle (cada linea de detalle_venta)
    df = detalle_df.copy()

    # Merge para obtener categoria del producto (si existe)
    productos_small = productos_df[['id_producto', 'categoria']].drop_duplicates()
    df = df.merge(productos_small, how='left', on='id_producto')

    # Definimos umbral de "venta grande" por importe:
    umbral_importe = df['importe'].mean() * 1.2
    df['venta_grande'] = (df['importe'] > umbral_importe).astype(int)
    print(f"Umbral venta_grande (importe): {umbral_importe}")

    # Features: cantidad, precio_unitario, importe, categoria (one-hot), medio_pago (desde ventas)
    df = df.merge(ventas_df[['id_venta', 'medio_pago']], how='left', on='id_venta')
    X = df[['cantidad', 'precio_unitario', 'importe']].copy()
    # categorización
    if 'categoria' in df.columns:
        cats = pd.get_dummies(df['categoria'], prefix='cat')
        X = pd.concat([X, cats], axis=1)
    if 'medio_pago' in df.columns:
        pagos = pd.get_dummies(df['medio_pago'].fillna('unknown'), prefix='pago')
        X = pd.concat([X, pagos], axis=1)

    y = df['venta_grande']

    # Evitar filas con inf/nan
    X = X.fillna(0)

    # Train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Modelo: LogisticRegression + RandomForest (probamos ambos)
    log = LogisticRegression(max_iter=1000)
    log.fit(X_train, y_train)
    pred_log = log.predict(X_test)
    acc_log = accuracy_score(y_test, pred_log)

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, pred_rf)

    # Elegir mejor
    modelo_final = rf if acc_rf >= acc_log else log
    nombre_modelo = "modelo_C_rf.joblib" if modelo_final is rf else "modelo_C_log.joblib"
    guardar_modelo(modelo_final, nombre_modelo)

    cm = confusion_matrix(y_test, modelo_final.predict(X_test))
    print("Modelo C - acc_log:", acc_log, "acc_rf:", acc_rf)
    print("Modelo C - confusion matrix:\n", cm)

    # Gráfico matriz confusión
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title("C: Matriz confusión - Venta grande vs chica")
    guardar_figura(fig, "C_matriz_confusion.png")

    # Guardar resumen
    resumen = pd.DataFrame({
        'umbral_importe': [umbral_importe],
        'acc_log': [acc_log],
        'acc_rf': [acc_rf],
        'n_registros': [len(df)]
    })
    resumen.to_csv(os.path.join(GRAFICOS_DIR, "C_resumen.csv"), index=False)

    return {'modelo': modelo_final, 'acc_log': acc_log, 'acc_rf': acc_rf, 'cm': cm, 'umbral': umbral_importe}

# --------------------------
# EJECUTAR LOS 3 MODELOS
# --------------------------
if __name__ == "__main__":
    print("Iniciando Sprint 3 - ML")
    resultado_A = modelo_A_clasificacion_top(productos_ml)
    resultado_B = modelo_B_regresion(productos_ml)
    resultado_C = modelo_C_clasificacion_ventas(detalle, ventas, productos)
    print("Fin Sprint 3. Resumen:")
    print("A:", resultado_A)
    print("B:", resultado_B)
    print("C:", resultado_C)

