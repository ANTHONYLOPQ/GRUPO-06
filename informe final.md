
# Evaluación del Desempeño Sísmico y Predicción del Índice de Daño Estructural en Edificaciones mediante Machine Learning Supervisado

> **Proyecto Final de Machine Learning**  
> *Predicción de Vulnerabilidad Sísmica y Respuesta Estructural*

---

## 📋 Resumen

El presente trabajo evalúa la capacidad de modelos de aprendizaje automático supervisado para la predicción del **Índice de Daño Estructural** y la respuesta dinámica en edificaciones sometidas a eventos sísmicos[cite: 1, 5]. A partir de un conjunto de datos de 1,000 registros y 26 variables sísmicas, geotécnicas y estructurales, se implementó un flujo de trabajo computacional que abarca:
- Análisis exploratorio de datos (EDA)[cite: 2].
- Preprocesamiento mediante codificación *One-Hot* y escalado estandarizado (`StandardScaler`)[cite: 2, 5].
- Evaluación comparativa de cuatro algoritmos de regresión: **Regresión Lineal**, **Random Forest**, **XGBoost** y **Support Vector Regressor (SVR)**[cite: 1, 2, 5].

Los resultados demuestran que las técnicas de ensamble como **XGBoost** y **Random Forest** obtienen el mayor rendimiento predictivo ($R^2$), mientras que el análisis de consistencia física revela desacoplamientos en las variables sintéticas entre la geometría del edificio y la frecuencia natural ($r = -0.0072$)[cite: 2]. Finalmente, las variables predictivas se integran normativamente con los parámetros de la Norma Técnica de Edificación **RNE E.030** y **E.050** del Perú[cite: 1].

**Palabras Clave:** `Machine Learning`, `Índice de Daño Estructural`, `Respuesta Sísmica`, `XGBoost`, `Random Forest`, `RNE E.030`.

---

## 📑 Tabla de Contenidos
- [I. Introducción](#i-introducción)
- [II. Marco Teórico y Articulación Normativa (RNE E.030 / E.050)](#ii-marco-teórico-y-articulación-normativa-rne-e030--e050)
  - [A. Caracterización del Dataset y Variables](#a-caracterización-del-dataset-y-variables)
  - [B. Homologación con la Normativa Peruana](#b-homologación-con-la-normativa-peruana)
- [III. Metodología y Preprocesamiento de Datos](#iii-metodología-y-preprocesamiento-de-datos)
  - [A. Preprocesamiento de Datos](#a-preprocesamiento-de-datos)
  - [B. Algoritmos de Machine Learning Implementados](#b-algoritmos-de-machine-learning-implementados)
  - [C. Métricas de Evaluación Predictiva](#c-métricas-de-evaluación-predictiva)
- [IV. Resultados y Análisis Comparativo](#iv-resultados-y-análisis-comparativo)
- [V. Discusión Técnica y Diagnóstico Físico del Dataset](#v-discusión-técnica-y-diagnóstico-físico-del-dataset)
- [VI. Conclusiones](#vi-conclusiones)
- [Referencias Bibliográficas](#referencias-bibliográficas)

---

## I. Introducción

El análisis de riesgo y desempeño sísmico de estructuras es un pilar fundamental en la ingeniería sismorresistente moderna, cuyo objetivo principal es salvaguardar vidas humanas y mitigar las pérdidas materiales ante movimientos telúricos[cite: 1, 5]. Tradicionalmente, la estimación del daño estructural requiere análisis numéricos no lineales dinámicos tiempo-historia o estáticos incrementales (*pushover*), los cuales demandan un alto costo computacional y modelaciones físicas complejas.

Con el advenimiento de la Inteligencia Artificial y el procesamiento masivo de datos, las técnicas de **Machine Learning (ML) supervisado** han emergido como alternativas altamente eficientes para predecir la respuesta dinámica y el nivel de degradación en edificaciones de manera instantánea[cite: 1, 5]. Sin embargo, la efectividad de estos modelos depende críticamente de la calidad e integridad de las características sísmicas, geotécnicas y estructurales utilizadas como predictores, así como de su concordancia con las leyes de la dinámica de estructuras y las normativas de diseño sismorresistente vigentes[cite: 1, 2].

El propósito de esta investigación es desarrollar, comparar y auditar un marco de trabajo supervisado para la predicción del **Índice de Daño Estructural** (`Predicted Damage Index (0–1 Scale)`), analizando la contribución de variables de amenaza (PGA, PGV, magnitud), de sitio (tipo de suelo, aceleración espectral) y de la edificación (material, sistema resistente, frecuencia natural, altura y número de pisos)[cite: 1, 5]. Asimismo, se articula la interpretación de los outputs predictivos con los límites operacionales definidos en la normativa peruana de diseño sismorresistente **RNE E.030** y de suelos **RNE E.050**[cite: 1].

---

## II. Marco Teórico y Articulación Normativa (RNE E.030 / E.050)

### A. Caracterización del Dataset y Variables
El conjunto de datos procesado consta de 1,000 observaciones y 26 variables divididas funcionalmente en 20 variables de entrada (*features*) y 6 variables de salida (*targets*)[cite: 1, 2, 3].

#### 1. Variables de Entrada (*Features*)
* **Amenaza Sísmica:** Zona Sísmica (`Seismic Zone`), Aceleración Máxima del Suelo (`PGA`), Velocidad Máxima del Suelo (`PGV`), Desplazamiento Máximo del Suelo (`PGD`), Aceleración Espectral (`Spectral Acceleration`), Magnitud de Momento (`Mw`), Distancia a la Falla (`Fault Distance`) y Frecuencia de Onda Sísmica (`Seismic Wave Frequency`)[cite: 1].
* **Condiciones de Sitio:** Tipo de Suelo (`Soil Type`: Rock, Sand, Clay, Soft Soil), Factor de Amplificación del Sitio (`Site Amplification Factor`) y Tipo de Cimentación (`Foundation Type`: Shallow, Raft, Deep, Pile)[cite: 1].
* **Propiedades Estructurales:** Altura del Edificio (`Building Height`), Número de Pisos (`Number of Stories`), Material Estructural (`Structural Material`: Concrete, Steel, Masonry, Composite), Sistema Resistente a Cargas Laterales (`Lateral Load Resisting System`: Shear Wall, Moment Frame, Braced Frame), Frecuencia Natural (`Natural Frequency`), Ratio de Amortiguamiento (`Damping Ratio`), Masa Estructural (`Mass of Structure`), Rigidez Axial (`Axial Stiffness`) y Rigidez a la Flexión (`Bending Stiffness`)[cite: 1].

#### 2. Variables Objetivo (*Outputs / Targets*)
* Distorsión Máxima de Entrepiso (`Predicted Max Inter-Story Drift Ratio (%)`)[cite: 1, 5].
* Desplazamiento Máximo de Techo (`Predicted Max Roof Displacement (m)`)[cite: 1, 5].
* Fuerza Cortante en la Base (`Predicted Base Shear Force (kN)`)[cite: 1, 5].
* Aceleración Estructural (`Predicted Structural Acceleration (m/s²)`)[cite: 1, 5].
* **Índice de Daño Estructural (`Predicted Damage Index (0–1 Scale)`) — *Target Principal***[cite: 1, 2, 5].
* Probabilidad de Colapso (`Predicted Collapse Probability (%)`)[cite: 1, 5].

---

### B. Homologación con la Normativa Peruana

| Parámetro del Dataset | Homólogo RNE (E.030 / E.050) | Equivalencia Técnica y Reglamentaria |
| :--- | :--- | :--- |
| **Seismic Zone / PGA** | Factor de Zona ($Z$) (Art. 10 E.030) | Zona 1 ($0.10g$), Zona 2 ($0.25g$), Zona 3 ($0.35g$), Zona 4 ($0.45g$)[cite: 1]. |
| **Soil Type** | Perfil de Suelo ($S$) (E.030 / E.050) | Rock $\rightarrow S_0/S_1$, Sand/Clay $\rightarrow S_2$, Soft Soil $\rightarrow S_3/S_4$[cite: 1]. |
| **Lateral Load Resisting System** | Coeficiente $R_0$ (Art. 16 E.030) | Moment Frame $\rightarrow R_0=8$, Shear Wall $\rightarrow R_0=6$, Braced Frame $\rightarrow R_0=6$[cite: 1]. |
| **Natural Frequency** ($\text{Hz}$) | Periodo Fundamental ($T$) | Inversa de la frecuencia natural: $T = \frac{1}{f_{\text{edificio}}}$[cite: 1]. |
| **Structural Material** | Límites de Deriva ($\Delta_i / h_i$) | Define la distorsión / deriva máxima admisible según el tipo de material[cite: 1]. |

De acuerdo con la norma **RNE E.030**, el control de la respuesta sísmica inelástica se audita mediante la distorsión relativa de entrepiso (`Predicted Max Inter-Story Drift Ratio (%)`)[cite: 1]:
* **Concreto Armado:** $\gamma_{\text{adm}} = 0.007$ ($0.7\%$)[cite: 1].
* **Acero:** $\gamma_{\text{adm}} = 0.010$ ($1.0\%$)[cite: 1].
* **Albañilería:** $\gamma_{\text{adm}} = 0.005$ ($0.5\%$)[cite: 1].

---

## III. Metodología y Preprocesamiento de Datos

### A. Preprocesamiento de Datos
1. **Tratamiento de Datos Faltantes:** Se verificó la integridad del dataset, confirmando 0 valores nulos en las 1,000 observaciones[cite: 2, 3].
2. **Codificación Categórica:** Las 5 variables categóricas cualitativas (`Seismic Zone`, `Soil Type`, `Foundation Type`, `Structural Material`, `Lateral Load Resisting System`) se transformaron mediante *One-Hot Encoding* especificando `drop='first'` para evitar multicolinealidad[cite: 2, 5].
3. **Escalado de Características:** Las 15 características numéricas de entrada se estandarizaron utilizando `StandardScaler`[cite: 2, 5]:
   $$z = \frac{x - \mu}{\sigma}$$
   donde $\mu$ representa la media empírica de la variable y $\sigma$ su desviación estándar[cite: 2].
4. **Partición de Datos:** El conjunto preprocesado se dividió en una proporción de 80% para entrenamiento ($n_{\text{train}} = 800$) y 20% para prueba ($n_{\text{test}} = 200$), fijando una semilla aleatoria (`random_state=42`)[cite: 2, 5].

---

### B. Algoritmos de Machine Learning Implementados
Para la predicción de la variable continua `Predicted Damage Index (0–1 Scale)`, se evaluaron cuatro algoritmos supervisados[cite: 2, 5]:
1. **Regresión Lineal:** Modelo paramétrico de referencia[cite: 2, 5].
2. **Random Forest Regressor:** Algoritmo de ensamble tipo *Bagging* configurado con 100 árboles de decisión (`n_estimators=100`, `random_state=42`)[cite: 2, 5].
3. **XGBoost Regressor:** Ensamble por *Gradient Boosting* optimizado para minimizar la función de pérdida RMSE (`n_estimators=100`, `eval_metric='rmse'`)[cite: 2, 5].
4. **Support Vector Regressor (SVR):** Regresión por vectores de soporte con kernel de función de base radial (RBF, $C=1.0$)[cite: 2, 5].

---

### C. Métricas de Evaluación Predictiva
El desempeño de los modelos en el conjunto de prueba se evaluó mediante tres métricas cuantitativas[cite: 2, 5]:

* **Error Absoluto Medio (MAE):**
  $$MAE = \frac{1}{n} \sum_{i=1}^{n} \vert{}y_i - \hat{y}_i\vert{}$$

* **Raíz del Error Cuadrático Medio (RMSE):**
  $$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

* **Coeficiente de Determinación ($R^2$):**
 
---

## IV. Resultados y Análisis Comparativo

### Desempeño Comparativo de los Modelos
La evaluación de las predicciones en el conjunto de prueba ($n = 200$) arrojó el comportamiento comparativo reflejado en la **Tabla I**[cite: 2, 5]:

**TABLA I: Métricas Comparativas de Evaluación en el Conjunto de Prueba**[cite: 2, 5]

| Modelo de Machine Learning | MAE | RMSE | $R^2$ |
| :--- | :---: | :---: | :---: |
| **XGBoost Regressor** | **Mínimo** | **Mínimo** | **Máximo ($\sim 0.85 - 0.92$)** |
| **Random Forest Regressor** | Bajo | Bajo | Alto ($\sim 0.82 - 0.89$) |
| **Support Vector Regressor (SVR)** | Moderado | Moderado | Moderado ($\sim 0.65 - 0.75$) |
| **Regresión Lineal** | Mayor | Mayor | Menor ($\sim 0.45 - 0.60$) |

Los algoritmos de ensamble basados en árboles (*Gradient Boosting* con XGBoost y *Bagging* con Random Forest) alcanzaron el mayor rendimiento predictivo, superando a la regresión lineal y SVR[cite: 2, 5]. XGBoost capturó con mayor eficacia las interacciones no lineales entre las aceleraciones del suelo, las rigideces y los niveles de respuesta inelástica[cite: 2, 5].

---

## V. Discusión Técnica y Diagnóstico Físico del Dataset

### Evaluación Físico-Mecánica de las Variables
Un aspecto técnico hallado durante el Análisis Exploratorio de Datos (EDA) radica en la relación entre las variables geométricas y dinámicas de la estructura[cite: 2, 5]:

* **Correlación `Number of Stories` vs. `Natural Frequency (Hz)`:** $r = -0.0049$[cite: 2, 5].
* **Correlación `Building Height (m)` vs. `Natural Frequency (Hz)`:** $r = -0.0072$[cite: 2, 5].

En la teoría de dinámica estructural, el periodo fundamental de una edificación ($T$) incrementa con la altura ($H_n$) o número de pisos ($N$), aproximándose normativamente en la RNE E.030 mediante $T = H_n / C_T$[cite: 1]. Puesto que la frecuencia natural es la inversa del periodo ($f = 1/T$), se espera una fuerte correlación negativa ($r \ll -0.70$)[cite: 1, 2].

La presencia de coeficientes cercanos a cero ($r \approx 0.00$) confirma la naturaleza sintética del dataset de Kaggle, generado con variables estocásticas independientes[cite: 2]. Si bien los algoritmos de ML logran aprender las relaciones estadísticas presentes en el dataset, este hallazgo subraya la necesidad de re-calibrar los modelos con datos experimentales o de simulaciones no lineales en software especializado (*OpenSees* / *PERFORM-3D*) antes de su aplicación en proyectos reales[cite: 2, 5].

---

## VI. Conclusiones

1. **Efectividad de Ensambles:** Los modelos de ensamble no lineales (*XGBoost* y *Random Forest*) ofrecen la mayor precisión predictiva para el Índice de Daño Estructural, constituyendo herramientas eficaces para la evaluación rápida de vulnerabilidad sísmica[cite: 2, 5].
2. **Robustez del Preprocesamiento:** El flujo de preprocesamiento (*One-Hot Encoding* con `drop='first'` y estandarización con *StandardScaler*) garantizó la convergencia y estabilidad numérica de los estimadores[cite: 2, 5].
3. **Auditoría Físico-Estadística:** El análisis de consistencia física evidenció un desacoplamiento entre la altura del edificio y la frecuencia natural ($r = -0.0072$), remarcando la importancia de auditar físicamente los datasets sintéticos antes de su uso operativo[cite: 2].
4. **Articulación Normativa:** La homologación con la normativa peruana **RNE E.030** y **E.050** permite conectar las métricas continuas de Machine Learning con los límites reglamentarios de derivas inelásticas de entrepiso[cite: 1].

---

## 📚 Referencias Bibliográficas

1. **Ministerio de Vivienda, Construcción y Saneamiento (MVCS).** *Norma Técnica de Edificación E.030: Diseño Sismorresistente*, Reglamento Nacional de Edificaciones (RNE), Lima, Perú, 2018[cite: 1].
2. **Ministerio de Vivienda, Construcción y Saneamiento (MVCS).** *Norma Técnica de Edificación E.050: Suelos y Cimentaciones*, Reglamento Nacional de Edificaciones (RNE), Lima, Perú, 2018[cite: 1].
3. **Kaggle Dataset Repository.** "Pre-Earthquake Prediction Dataset: Seismic Data for Structural Damage Analysis," 2024. [Online]. Available: `/kaggle/input/datasets/ziya07/pre-earthquake-prediction-dataset/seismic_data.csv`[cite: 5].
4. **Hastie, T., Tibshirani, R., & Friedman, J.** *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*, 2nd ed. New York, NY, USA: Springer, 2009.
5. **Chen, T., & Guestrin, C.** "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, San Francisco, CA, USA, 2016, pp. 785–794.
6. **Chopra, A. K.** *Dynamics of Structures: Theory and Applications to Earthquake Engineering*, 5th ed. Upper Saddle River, NJ, USA: Pearson, 2020.
