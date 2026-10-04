## 1. RESUMEN
Se busca realizar un análisis exploratorio, diagnóstico físico e integración normativa de la data. El objetivo es establecer las bases para el desarrollo de un modelo predictivo basado en Inteligencia Artificial que permita estimar la respuesta dinámica y el daño en edificaciones, alineando los resultados con los criterios de desempeño del Reglamento Nacional de Edificaciones (RNE) de Perú y específicamente las normas NTE E.030 (Diseño Sismorresistente).
## 2. CARACTERIZACIÓN DE LA BASE DE DATOS
El dataset contiene 1,000 casos que simulan la interacción suelo-estructura y parámetros geométricos de distintas edificaciones ante eventos sísmicos. Las 26 variables se estructuran técnicamente en Variables de Entrada (Inputs) y Variables de Respuesta Predicha (Outputs):
### 2.1 Variables de Entrada (Features)
**A. Amenaza y Peligro Sísmico:**
* **Seismic Zone:** Categoría de amenaza (Low, Moderate, High, Very High).
* **PGA (m/s²):** Aceleración máxima del suelo (Peak Ground Acceleration).
* **PGV (m/s) y PGD (m):** Velocidad y desplazamiento máximo del suelo.
* **Spectral Acceleration (g):** Aceleración espectral (Sa).
* **Historical Earthquake Magnitude (Mw) y Fault Distance (km):** Magnitud e hipocentro/distancia a falla.
* **Seismic Wave Frequency (Hz):** Frecuencia predominante del registro sísmico (f sismo).
  
**B. Condiciones del Sitio y Suelo:**
* **Soil Type:** Clasificación del suelo (Rock, Sand, Clay, Soft Soil).
* **Site Amplification Factor:** Factor de amplificación local de onda.
* **Foundation Type:** Tipo de cimentación (Shallow, Raft, Deep, Pile).
  
**C. Propiedades Físicas y Estructurales del Edificio:**
* **Building Height (m) y Number of Stories:** Altura total y número de niveles.
* **Structural Material:** Material predominante (Concrete, Steel, Masonry, Composite).
* **Lateral Load Resisting System:** Sistema resistente a cargas laterales (Shear Wall, Moment Frame, Braced Frame).
* **Natural Frequency (Hz):** Frecuencia fundamental del edificio (f_edificio = 1 / T).
* **Damping Ratio (%):** Ratio de amortiguamiento crítico (ξ).
* **Mass of Structure (kg):** Masa total de la estructura (m).
* **Axial Stiffness (kN/m) y Bending Stiffness (kN·m²):** Rigideces equivalentes (EA y EI).
### 2.2 Variables de Salida Predichas (Outputs)
* **Predicted Max Inter-Story Drift Ratio (%):** Distorsión / deriva máxima de entrepiso (γ).
* **Predicted Max Roof Displacement (m):** Desplazamiento de entrepisos (D).
* **Predicted Base Shear Force (kN):** Fuerza cortante en la base (V).
* **Predicted Structural Acceleration (m/s²):** Aceleración máxima de respuesta en la estructura.
* **Predicted Damage Index (0–1 Scale):** Índice de daño estructural normalizado.
* **Predicted Collapse Probability (%):** Probabilidad estimada de colapso.
## 3. CORRELACIÓN E IMPLEMENTACIÓN CON LA NORMATIVA PERUANA (NTE E.030 Y E.050)
Para aplicar este dataset al contexto de la ingeniería estructural en el Perú, se establece el siguiente esquema de homologación y validación normativa:
### 3.1 Mapeo de Parámetros de Entrada
| Parámetro del Dataset | Homólogo en Norma Peruana (NTE E.030 / E.050) | Equivalencia o Criterio Técnico |
| :--- | :--- | :--- |
| **Seismic Zone / PGA** | Factor de Zona (Z) (E.030 - Art. 10) | Z1 = 0.10g, Z2 = 0.25g, Z3 = 0.35g, Z4 = 0.45g.|
| **Soil Type** | Perfil de Suelo (S) (E.030 - E.050) | Rock → S0/S1, Sand/Clay → S2, Soft Soil → S3/S4. |
| **Lateral Load Resisting System** | Coeficiente de Reducción (R0) (E.030) | Moment Frame → R0=8, Shear Wall → R0=6, Braced Frame → R0=6.|
| **Natural Frequency (Hz)** | Periodo Fundamental (T) (E.030) | Relación directa: T = 1 / Natural Frequency (Hz). |
| **Structural Material** | Material Estructural (NTE E.060, E.070, E.090) | Determina el límite reglamentario de deriva de entrepiso. |

### 3.2 Criterio de Verificación Normativa: Distorsión de Entrepiso
El parámetro principal para auditar la seguridad estructural en el RNE es la Distorsión Máxima de Entrepiso.
Drift Predicho (%) = (Predicted Max Story Drift) / 100
* **Concreto Armado (Concrete):** Límite máximo admisible = 0.007 (0.7%)
* **Acero (Steel):** Límite máximo admisible = 0.010 (1.0%)
* **Albañilería (Masonry):** Límite máximo admisible = 0.005 (0.5%)
  
## 4. CONCLUSIONES 
1. **Aplicabilidad del Dataset:** El dataset constituye una base de entrenamiento sintética adecuada para implementar arquitecturas de regresión / aprendizaje automático y estrategias de Prompt Engineering orientadas a la predicción de daño.
2. **Cumplimiento Normativo:** La variable Predicted Max Inter-Story Drift Ratio (%) permite vincular directamente los resultados de la IA con parametros de la NTE E.030, proporcionando un criterio objetivo de aceptabilidad estructural.
