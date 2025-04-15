import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Crear carpeta de gráficos si no existe
os.makedirs("Graficos", exist_ok=True)
print("[INFO] Carpeta 'Graficos' creada o ya existente.")

# 1. Cargar archivos Excel con datos de cada año
try:
    df_2019 = pd.read_excel("datos/totales-finales-de-ingresos-ano-2019-municipal.xls", skiprows=4, decimal=',')
    df_2020 = pd.read_excel("datos/totales-finales-de-ingresos-ano-2020-municipal.xls", skiprows=4, decimal=',')
    df_2021 = pd.read_excel("datos/totales-finales-de-ingresos-ano-2021municipal.xls", skiprows=4, decimal=',')
    # df_2019 = pd.read_excel("datos/totales-finales-de-ingresos-ano-2019-municipal.xls", skiprows=4, engine="openpyxl")
    # df_2020 = pd.read_excel("datos/totales-finales-de-ingresos-ano-2020-municipal.xls", skiprows=4, engine="openpyxl")
    # df_2021 = pd.read_excel("datos/totales-finales-de-ingresos-ano-2021municipal.xls", skiprows=4, engine="openpyxl")
    print("[INFO] Archivos Excel cargados correctamente.")
except FileNotFoundError as e:
    print(f"[ERROR] Archivo no encontrado: {e}")
    exit(1)

# 2. Limpieza de datos
def procesar_archivo(df, año):
    print(f"[INFO] Procesando archivo del año {año}...")
    df = df.dropna(how='all')  # Eliminar filas totalmente vacías
    df.columns = ["mes", "etiqueta", "presupuesto_inicial", "presupuesto_vigente",
                  "recaudado", "recaudado_anterior", "porcentaje", "recaudado_real",
                  "recaudado_real_anterior", "porcentaje_real", "porcentaje_cambio"]
    df = df[df["etiqueta"] == "TOTALES FINALES"]
    df['Año'] = año
    print(f"[INFO] Archivo {año} procesado con {len(df)} filas.")
    return df

df_2019 = procesar_archivo(df_2019, 2019)
df_2020 = procesar_archivo(df_2020, 2020)
df_2021 = procesar_archivo(df_2021, 2021)

# Unir todos los años
df_total = pd.concat([df_2019, df_2020, df_2021])
print(f"[INFO] Dataset total unificado con forma: {df_total.shape}")

# 3. Análisis por mes
print("[INFO] Analizando ingresos por mes...")
# Asegurarse de que 'recaudado_real' sea numérico antes de la agregación
df_total['recaudado_real'] = pd.to_numeric(df_total['recaudado_real'], errors='coerce')
ingresos_por_mes = df_total.groupby("mes")["recaudado_real"].sum().sort_values(ascending=False)
print("\n[INFO] Meses con mayores ingresos acumulados:\n", ingresos_por_mes)

variabilidad_por_año = df_total.groupby("Año")["recaudado_real"].std()
print("\n[INFO] Año con mayor variabilidad mensual:\n", variabilidad_por_año)

# 4. Visualización
print("[INFO] Generando visualizaciones...")

# a. Gráfico de líneas
plt.figure(figsize=(12,6))
sns.lineplot(data=df_total, x="mes", y="recaudado_real", hue="Año")
plt.title("Ingresos Recaudados Reales Mensuales por Año")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Graficos/lineas_por_año.png")
print("[INFO] Gráfico de líneas por año guardado.")

# b. Gráfico de barras agrupadas
plt.figure(figsize=(12,6))
sns.barplot(data=df_total, x="mes", y="recaudado_real", hue="Año")
plt.title("Comparación de Ingresos Reales por Mes y Año")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Graficos/barras_agrupadas.png")
print("[INFO] Gráfico de barras agrupadas guardado.")

# c. Mapa de calor
print("[INFO] Generando mapa de calor...")
pivot_df = df_total.pivot_table(index='mes', columns='Año', values='recaudado_real', aggfunc='sum')
plt.figure(figsize=(10,6))
sns.heatmap(pivot_df, annot=True, fmt=".0f", cmap='YlGnBu')
plt.title("Mapa de calor de ingresos reales por mes y año")
plt.tight_layout()
plt.savefig("Graficos/mapa_calor.png")
print("[INFO] Mapa de calor guardado.")

# 5. Clustering con KMeans
print("[INFO] Iniciando clustering con KMeans...")

df_kmeans = df_total.groupby(["Año", "mes"]).agg({"recaudado_real": "sum"}).reset_index()

# Convertir meses a números
mes_a_num = {
    'ENERO': 1, 'FEBRERO': 2, 'MARZO': 3, 'ABRIL': 4, 'MAYO': 5, 'JUNIO': 6,
    'JULIO': 7, 'AGOSTO': 8, 'SEPTIEMBRE': 9, 'OCTUBRE': 10, 'NOVIEMBRE': 11, 'DICIEMBRE': 12
}

# Limpiar la columna 'mes': convertir a mayúsculas y eliminar espacios
df_kmeans["mes_clean"] = df_kmeans["mes"].str.upper().str.strip()
df_kmeans["mes_num"] = df_kmeans["mes_clean"].map(mes_a_num)

print("Valores únicos en la columna 'mes' original:")
print(df_kmeans['mes'].unique())
print("Valores únicos en la columna 'mes_clean' después de limpieza:")
print(df_kmeans['mes_clean'].unique())
print("\nNúmero de valores nulos en 'mes_num':", df_kmeans['mes_num'].isnull().sum())
print("\nFilas con valores nulos en 'mes_num':")
print(df_kmeans[df_kmeans['mes_num'].isnull()])

# Asegurarse de que 'recaudado_real' en df_kmeans sea numérico para KMeans
df_kmeans['recaudado_real'] = pd.to_numeric(df_kmeans['recaudado_real'], errors='coerce')
df_kmeans_no_na = df_kmeans.dropna(subset=['recaudado_real', 'mes_num']) # Incluir 'mes_num' para eliminar filas sin mes numérico válido

print("\ndf_kmeans:")
print(df_kmeans)
print("\ndf_kmeans_no_na después de eliminar NaN:")
print(df_kmeans_no_na)
print("\nDescripción estadística de 'recaudado_real' en df_kmeans_no_na:")
print(df_kmeans_no_na['recaudado_real'].describe())

if df_kmeans_no_na.empty:
    print("[ERROR] No hay datos válidos para realizar el clustering.")
else:
    # Aplicar KMeans solo si hay datos válidos
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_kmeans_no_na["cluster"] = kmeans.fit_predict(df_kmeans_no_na[["recaudado_real"]])
    print("\nValores únicos en la columna 'cluster':", df_kmeans_no_na['cluster'].unique())
    print("[INFO] Clustering completado.")

    # Visualizar clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_kmeans_no_na, x="mes_num", y="recaudado_real", hue="cluster", palette="Set2", style="Año")
    plt.title("Clusters de Ingresos Recaudados Reales")
    plt.xlabel("Mes (número)")
    plt.ylabel("Ingresos Reales (CLP)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Graficos/clusters_ingresos.png")
    print("[INFO] Gráfico de clustering guardado.")

print("[WARN] No se pudo realizar el clustering debido a la falta de datos numéricos válidos.")
print("\n[INFO] Script finalizado con éxito.")