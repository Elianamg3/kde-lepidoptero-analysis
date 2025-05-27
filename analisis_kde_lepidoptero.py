#!/usr/bin/env python3
# Análisis de Densidad de Kernel (KDE) para determinar instares larvarios
# de una especie de lepidóptero a partir de anchos de cápsulas cefálicas
# Autor: [Eliana Galindez]
# Fecha: 27 de mayo de 2025

import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import gdown
from google.colab import drive
import os

# Paso 1: Montar Google Drive
print("Montando Google Drive...")
drive.mount('/content/drive')

# Paso 2: Descargar el archivo CSV desde Google Drive
print("
Descargando el archivo CSV...")
file_id = '1kSEqzLNi3xjzID7pn1OjLase0uiohwXs'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'capsulas_cefalicas.csv'
gdown.download(url, output, quiet=False)

# Verificar que el archivo se descargó
if not os.path.exists(output):
    raise FileNotFoundError(f"No se pudo descargar el archivo '{output}'. Verifica el enlace o permisos.")

# Paso 3: Leer y verificar el CSV
print("
Leyendo y verificando el archivo CSV...")
data = pd.read_csv(output)
print("Primeras 5 filas del archivo:")
print(data.head())
print("
Columnas en el archivo:", data.columns.tolist())
if 'Ancho_cap' not in data.columns:
    raise KeyError(f"La columna 'Ancho_cap' no existe. Columnas disponibles: {data.columns.tolist()}")
print("
Tipo de datos de 'Ancho_cap':", data['Ancho_cap'].dtype)
print("Valores faltantes en 'Ancho_cap':", data['Ancho_cap'].isna().sum())
data = data[pd.to_numeric(data['Ancho_cap'], errors='coerce').notnull()]
data = data.dropna(subset=['Ancho_cap'])
data['Ancho_cap'] = data['Ancho_cap'].astype(float)
print("Resumen estadístico de 'Ancho_cap':")
print(data['Ancho_cap'].describe())

# Paso 4: Análisis de Densidad de Kernel (KDE)
print("
Realizando análisis KDE con ancho de banda = 0.06...")
hcw = data['Ancho_cap'].values.reshape(-1, 1)
kde = KernelDensity(kernel='gaussian', bandwidth=0.06).fit(hcw)
x_grid = np.linspace(min(hcw), max(hcw), 1000).reshape(-1, 1)
log_dens = kde.score_samples(x_grid)
dens = np.exp(log_dens)
plt.figure(figsize=(10, 6))
plt.hist(hcw, bins=30, density=True, alpha=0.5, color='gray', label='Histograma')
plt.plot(x_grid, dens, color='blue', lw=2, label='Densidad de Kernel (bw=0.06)')
peaks, properties = find_peaks(dens, height=0, prominence=0.1 * max(dens))
peak_values = x_grid[peaks].flatten()
peak_heights = dens[peaks]
if len(peak_values) > 6:
    sorted_indices = np.argsort(peak_heights)[::-1][:6]
    peak_values = peak_values[sorted_indices]
    peak_heights = peak_heights[sorted_indices]
    peaks = peaks[sorted_indices]
sorted_indices = np.argsort(peak_values)
peak_values = peak_values[sorted_indices]
peak_heights = peak_heights[sorted_indices]
peaks = peaks[sorted_indices]
plt.plot(peak_values, peak_heights, 'ro', label='Picos (Instar)')
for i, peak in enumerate(peak_values):
    plt.text(peak, peak_heights[i], f'{peak:.3f}', ha='center', va='bottom', color='red')
plt.xlabel('Ancho de Cápsula Cefálica (mm)')
plt.ylabel('Densidad')
plt.title('Estimación de Densidad de Kernel (bw=0.06)')
plt.legend()
output_dir = '/content/drive/MyDrive/Analisis_Lepidoptero'
os.makedirs(output_dir, exist_ok=True)
kde_path = os.path.join(output_dir, 'kde_plot.png')
plt.savefig(kde_path)
print(f"Gráfico KDE guardado en '{kde_path}'")
plt.show()
print(f"Número de instares detectados: {len(peak_values)}")
print(f"Medias de los instares (mm): {peak_values}")

# Paso 5: Validación con la Regla de Dyar
if len(peak_values) > 1:
    print("
Validando con la regla de Dyar...")
    instar = np.arange(1, len(peak_values) + 1)
    dyar_model = np.polyfit(instar, np.log(peak_values), 1)
    r_squared = 1 - np.sum((np.log(peak_values) - (dyar_model[1] + dyar_model[0] * instar))**2) /                 np.sum((np.log(peak_values) - np.mean(np.log(peak_values)))**2)
    print(f"Regla de Dyar: Pendiente={dyar_model[0]:.3f}, R²={r_squared:.3f}")
    plt.figure(figsize=(10, 6))
    plt.scatter(instar, np.log(peak_values), color='blue', label='Datos')
    plt.plot(instar, dyar_model[1] + dyar_model[0] * instar, color='red', label='Ajuste Lineal')
    plt.xlabel('Instar')
    plt.ylabel('log(Ancho)')
    plt.title(f'Regla de Dyar (R²={r_squared:.3f})')
    plt.legend()
    dyar_path = os.path.join(output_dir, 'dyar_plot.png')
    plt.savefig(dyar_path)
    print(f"Gráfico Dyar guardado en '{dyar_path}'")
    plt.show()

# Paso 6: Guardar resultados en CSV
print("
Guardando resultados...")
results = pd.DataFrame({
    'instar': range(1, len(peak_values) + 1),
    'media_ancho_mm': peak_values
})
results_path = os.path.join(output_dir, 'instar_results.csv')
results.to_csv(results_path, index=False)
print(f"Resultados guardados en '{results_path}'")
