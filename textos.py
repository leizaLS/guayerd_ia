document = {
    "1": """1. Tema, problema y solución.
*Este proyecto simula la gestión de una Tienda a partir de datos sintéticos.
*El objetivo es disponer de un escenario consistente para practicar análisis, visualización y modelado. 
""",
    
    "2": """2. Dataset de referencia: fuente y definición
*Fuente: datos generados con fines educativos.
*Definición: base que representa una Tienda, con catálogo de productos, registro de clientes y 
operaciones de venta.
""",

    "3": """3. Estructura por tabla (columnas, tipo y escala de medición)

*Productos (productos.csv)
- id_producto (int, Nominal)
- nombre_producto (str, Nominal)
- categoria (str, Nominal)
- precio_unitario (int, Razón)

*Clientes (clientes.csv)
- id_cliente (int, Nominal)
- nombre_cliente (str, Nominal)
- email (str, Nominal)
- ciudad (str, Nominal)
- fecha_alta (date, Intervalo)

*Ventas (ventas.csv)
- id_venta (int, Nominal)
- fecha (date, Intervalo)
- id_cliente (int, Nominal)
- nombre_cliente (str, Nominal)
- email (str, Nominal)
- medio_pago (str, Nominal)

*Detalle_Ventas (detalle_ventas.csv)
- id_venta (int, Nominal)
- id_producto (int, Nominal)
- nombre_producto (str, Nominal)
- cantidad (int, Razón)
- precio_unitario (int, Razón)
- importe (int, Razón)
""",

    "4": """4. Escalas de medición
*Se usan 3 escalas:
- Nominal
- Intervalo
- Razón
""",

    "5": """5. Sugerencias y mejoras aplicadas con Copilot

- Separar la documentación en plantillas reutilizables.
- Proveer búsqueda por palabra clave.
- Agregar opción de exportar sección en ".txt" o ".md".
- Incluir tests mínimos para el router de opciones.
""",
}
