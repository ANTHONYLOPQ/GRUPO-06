import json

notebook = {
 "cells": [],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

def add_md(text):
    notebook["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [text]})

def add_code(text):
    notebook["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [text]})

add_md("# Análisis de Datos Sísmicos\n\nEste notebook contiene el análisis del dataset `seismic_data.csv`, abarcando limpieza, análisis exploratorio, reducción de dimensionalidad (PCA y Kernel PCA) y clustering (KMeans y DBSCAN).")

add_md("## 1. Analizar los datos")
add_code("""import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Cargar los datos
df = pd.read_csv('seismic_data.csv')

# Mostrar las primeras filas
display(df.head())

# Información general del dataset
print("\\n--- Información del Dataset ---")
df.info()

# Resumen estadístico
print("\\n--- Resumen Estadístico ---")
display(df.describe())

# Revisar valores nulos
print("\\n--- Valores Nulos por Columna ---")
display(df.isnull().sum())
""")

add_md("## 2. Observar Correlaciones")
add_code("""import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(16, 12))

# Seleccionar solo columnas numéricas para la matriz de correlación
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()

# Dibujar el mapa de calor
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Matriz de Correlación - Datos Sísmicos', fontsize=16)
plt.tight_layout()
plt.show()

# Mostrar las correlaciones más fuertes con la Probabilidad de Colapso
target_col = 'Predicted Collapse Probability (%)'
if target_col in correlation_matrix.columns:
    target_corr = correlation_matrix[target_col].sort_values(ascending=False)
    print(f"\\nCorrelación con '{target_col}':")
    display(target_corr.to_frame())
""")

add_md("## 3. Observar Insights y Gráficos")
add_code("""# Distribución de la Probabilidad de Colapso
plt.figure(figsize=(8, 5))
sns.histplot(df['Predicted Collapse Probability (%)'], bins=30, kde=True, color='teal')
plt.title('Distribución de la Probabilidad de Colapso', fontsize=14)
plt.xlabel('Probabilidad de Colapso (%)')
plt.ylabel('Frecuencia')
plt.show()

# Boxplot de Probabilidad de Colapso por Tipo de Suelo
plt.figure(figsize=(10, 6))
sns.boxplot(x='Soil Type', y='Predicted Collapse Probability (%)', data=df, palette='Set2')
plt.title('Probabilidad de Colapso según el Tipo de Suelo', fontsize=14)
plt.xlabel('Tipo de Suelo')
plt.ylabel('Probabilidad de Colapso (%)')
plt.show()

# Scatter plot: Magnitud Histórica vs Desplazamiento del Techo
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Historical Earthquake Magnitude (Mw)', y='Predicted Max Roof Displacement (m)', hue='Seismic Zone', data=df, palette='magma', alpha=0.7)
plt.title('Magnitud de Terremoto vs Desplazamiento Máximo del Techo', fontsize=14)
plt.xlabel('Magnitud Histórica (Mw)')
plt.ylabel('Desplazamiento Máximo del Techo (m)')
plt.legend(title='Zona Sísmica')
plt.show()
""")

add_md("## 4. Desarrollo de PCA y Kernel PCA")
add_code("""from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA

# Preprocesamiento: Usaremos solo las características numéricas y estandarizaremos
X_num = numeric_df.fillna(numeric_df.mean()) # Imputar nulos con la media si existen
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_num)

# --- PCA Tradicional ---
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
scatter1 = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Predicted Damage Index (0–1 Scale)'], cmap='viridis', alpha=0.6)
plt.colorbar(scatter1, label='Índice de Daño')
plt.title(f'PCA Tradicional\\nVarianza Explicada: {pca.explained_variance_ratio_.sum():.2%}')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')

# --- Kernel PCA (usando RBF) ---
# Usamos gamma=0.05 y fit_inverse_transform=False para optimizar
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.05, fit_inverse_transform=False)
X_kpca = kpca.fit_transform(X_scaled)

plt.subplot(1, 2, 2)
scatter2 = plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=df['Predicted Damage Index (0–1 Scale)'], cmap='viridis', alpha=0.6)
plt.colorbar(scatter2, label='Índice de Daño')
plt.title('Kernel PCA (Kernel RBF)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')

plt.tight_layout()
plt.show()
""")

add_md("## 5. KMeans y DBSCAN")
add_code("""from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# --- KMeans ---
# Determinamos 3 clusters (por ejemplo: bajo, medio, alto riesgo)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# --- DBSCAN ---
# Parámetros eps y min_samples ajustables
dbscan = DBSCAN(eps=3.5, min_samples=10)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Graficar resultados usando los componentes de PCA para visualización 2D
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='Set1', alpha=0.6)
plt.title('Clustering con KMeans (k=3)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')

plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap='Set2', alpha=0.6)
plt.title('Clustering con DBSCAN (-1 = Ruido)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')

plt.tight_layout()
plt.show()

# Evaluar calidad del clustering
print("--- Evaluación de Clusters ---")
if len(set(kmeans_labels)) > 1:
    print(f"Silhouette Score (KMeans): {silhouette_score(X_scaled, kmeans_labels):.4f}")
    
valid_dbscan_mask = dbscan_labels != -1
if len(set(dbscan_labels[valid_dbscan_mask])) > 1:
    print(f"Silhouette Score (DBSCAN - sin ruido): {silhouette_score(X_scaled[valid_dbscan_mask], dbscan_labels[valid_dbscan_mask]):.4f}")
else:
    print("DBSCAN agrupó todo como ruido o en un solo cluster (ajustar eps). ")
""")

with open('Analisis_Sismico.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Notebook generado exitosamente.")
