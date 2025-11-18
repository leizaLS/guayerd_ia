import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# 1. CONFIGURACIÓN Y CARGA DE DATOS
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "db")
GRAFICOS_DIR = os.path.join(BASE_DIR, "graficos")
os.makedirs(GRAFICOS_DIR, exist_ok=True)

def cargar_excel(nombre):
    return pd.read_excel(os.path.join(DATA_DIR, nombre))

clientes = cargar_excel("clientes.xlsx")
productos = cargar_excel("productos.xlsx")
ventas = cargar_excel("ventas.xlsx")
detalle = cargar_excel("detalle_ventas.xlsx")

# ==============================
# 2. UNION DE TABLAS
# ==============================
detalle_ventas = detalle.merge(ventas, on="id_venta", how="left")
detalle_ventas = detalle_ventas.merge(productos, on="id_producto", how="left")
detalle_ventas = detalle_ventas.merge(clientes, on="id_cliente", how="left")

# Asegurarse de tener la columna precio_unitario
if "precio_unitario" not in detalle_ventas.columns:
    posibles = [col for col in detalle_ventas.columns if "precio_unitario" in col and col != "precio_unitario"]
    if posibles:
        detalle_ventas["precio_unitario"] = detalle_ventas[posibles[0]]
    else:
        raise KeyError("No se encontró ninguna columna de precio unitario.")

# ==============================
# FUNCIONES AUXILIARES
# ==============================
def guardar_tabla_como_imagen(df, filename, fmt="{:.2f}"):
    # Formatear valores numéricos con 2 decimales como texto
    df_formatted = df.copy()
    for col in df_formatted.select_dtypes(include=['float', 'int']).columns:
        df_formatted[col] = df_formatted[col].map(lambda x: fmt.format(x))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(
        cellText=df_formatted.values,
        rowLabels=df_formatted.index,
        colLabels=df_formatted.columns,
        loc='center',
        cellLoc='center',
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    plt.savefig(os.path.join(GRAFICOS_DIR, filename), bbox_inches="tight", dpi=300)
    plt.close()

# ==============================
# 3. ESTADÍSTICAS DETALLE VENTAS
# ==============================
stats = detalle_ventas[["cantidad", "precio_unitario", "importe"]].describe()
guardar_tabla_como_imagen(stats, "estadisticas_detalle_ventas.png")

# ==============================
# 4. ESTADÍSTICAS PRECIO UNITARIO
# ==============================
stats_precio = productos["precio_unitario"].describe()
df_precio = pd.DataFrame(stats_precio).transpose()
guardar_tabla_como_imagen(df_precio, "estadisticas_precio_unitario.png")

# ==============================
# 5. OUTLIERS IMPORTE
# ==============================
def outliers_iqr(serie):
    q1, q3 = serie.quantile([0.25, 0.75])
    iqr = q3 - q1
    return serie[(serie < q1 - 1.5 * iqr) | (serie > q3 + 1.5 * iqr)]

outliers = outliers_iqr(detalle_ventas["importe"])
outliers_df = detalle_ventas.loc[outliers.index, ["id_venta", "id_producto", "importe"]]
guardar_tabla_como_imagen(outliers_df, "outliers_importe.png")

# ==============================
# 6. CORRELACIONES
# ==============================
corr = detalle_ventas[["cantidad", "precio_unitario", "importe"]].corr()
guardar_tabla_como_imagen(corr, "tabla_correlacion.png")

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.savefig(os.path.join(GRAFICOS_DIR, "correlacion.png"), bbox_inches="tight", dpi=300)
plt.close()

# ==============================
# 7. GRÁFICOS BÁSICOS
# ==============================
sns.histplot(detalle_ventas["precio_unitario"], kde=True)
plt.title("Distribución de Precios Unitarios")
plt.savefig(os.path.join(GRAFICOS_DIR, "distribucion_precios.png"))
plt.close()

sns.boxplot(x=detalle_ventas["importe"])
plt.title("Boxplot de Importes")
plt.savefig(os.path.join(GRAFICOS_DIR, "boxplot_importes.png"))
plt.close()

# ==============================
# 8. ESTADÍSTICAS VENTAS
# ==============================
ventas = cargar_excel("ventas.xlsx")

# Convertir 'fecha' a solo fecha (sin hora)
if 'fecha' in ventas.columns:
    ventas['fecha'] = pd.to_datetime(ventas['fecha']).dt.date

# Seleccionar estadísticas deseadas
ventas_desc = ventas.describe(include='all').transpose()[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
guardar_tabla_como_imagen(ventas_desc, "estadisticas_ventas.png")

# ==============================
# 9. IMPORTE VENDIDO POR CIUDAD
# ==============================
if "ciudad" in detalle_ventas.columns:
    ventas_por_ciudad = detalle_ventas.groupby("ciudad")["importe"].sum().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=ventas_por_ciudad, x="ciudad", y="importe")
    plt.xticks(rotation=30)
    plt.title("Total de Importe Vendido por Ciudad")
    plt.savefig(os.path.join(GRAFICOS_DIR, "importe_por_ciudad.png"), bbox_inches="tight", dpi=300)
    plt.close()

# ==============================
# 10. GRÁFICO DE TORTA: DISTRIBUCIÓN DE MEDIOS DE PAGO
# ==============================

if 'medio_pago' in ventas.columns:
    # Contar cada medio de pago
    medios_pago = ventas['medio_pago'].value_counts()

    # Crear gráfico de torta
    plt.figure(figsize=(8, 8))
    plt.pie(medios_pago, labels=medios_pago.index, autopct='%1.1f%%', startangle=90)
    plt.title("Distribución de Medios de Pago")
    plt.axis('equal')  # Para mantener el aspecto de círculo

    # Guardar imagen
    pie_chart_path = os.path.join(GRAFICOS_DIR, "distribucion_medios_pago.png")
    plt.savefig(pie_chart_path, bbox_inches="tight", dpi=300)
    plt.close()

# ==============================
# 11. TOP 10 PRODUCTOS MÁS VENDIDOS (por cantidad de ventas)
# ==============================

# Asegurarse de que tenemos las columnas necesarias
if "id_producto" in detalle_ventas.columns:

    # Contar las veces que aparece cada id_producto en las ventas
    top_ids = detalle_ventas["id_producto"].value_counts().nlargest(10).reset_index()
    top_ids.columns = ["id_producto", "veces_vendido"]

    # Unir con productos.xlsx para obtener el nombre del producto
    top_productos = top_ids.merge(productos[["id_producto", "nombre_producto"]], on="id_producto", how="left")

    # Reordenar columnas
    top_productos = top_productos[["nombre_producto", "veces_vendido"]]

    # Guardar como tabla
    guardar_tabla_como_imagen(top_productos, "top_10_productos_vendidos_veces.png")

    # Gráfico de barras
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_productos, x="veces_vendido", y="nombre_producto", orient="h")
    plt.title("Top 10 Productos por Cantidad de Ventas")
    plt.xlabel("Cantidad de Ventas")
    plt.ylabel("Producto")
    plt.savefig(os.path.join(GRAFICOS_DIR, "top_10_productos_vendidos_veces.png"), bbox_inches="tight", dpi=300)
    plt.close()



