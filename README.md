# Evaluación del Desempeño Sísmico y Predicción del Índice de Daño Estructural en Edificaciones mediante Machine Learning Supervisado
Trabajo escalonado Final - Grupo 06
## Integrantes:
-ANTHONY LOPEZ QUISPE

-MAX HOLGUINO ESPIRILLA
## Objetivo General:
Predecir de manera automática el nivel de daño físico y las variables de respuesta sísmica de una edificación por medio de sus características dinámico-mecánicas.
## Problema de ingeniería estructural
En la ingeniería estructural y sismorresistente, evaluar la respuesta sísmica de una edificación como son las derivas máximas de entrepiso junto con los valores de cortantes e índice de daños globales ante la ocurrencia de un sismo, requiere de un análisis dinámico cuyo método demanda costosos procedimientos relacionados a altos tiempos de cómputo y modelamientos avanzados.
En etapas de predimensionamiento o diseño preliminar y evaluación de vulnerabilidad sísmica, no es viable ejecutar métodos numéricos complejos para el análisis de estructuras. Por ello, es necesario contar con modelos predictivos que estimen con alta precisión la respuesta estructural y el nivel de daño a partir de parámetros sísmicos del suelo y de las propiedades dinámico-mecánicas del edificio.
## Descripción del Dataset Seleccionado
  El trabajo utilizará un archivo que consta de 1000 registros numéricos de 26 variables los cuales se agrupan en:
    	-Acción sísmica y propiedades del terreno:
        Aceleración espectral (Sa), PGA, PGV, PGD,Frecuencia de Onda (HZ), Magnitud del Sismo, Factor de amplificación de Sitio, Zona sísmica y tipo de suelo.  
      -Propiedades de la Edificación:
        Altura, Número de pisos, Frecuencia de Onda, Amortiguamiento, Masas, Rigidez Axial, Rigidez a flexión, Tipo de Material, Tipo de Cimentación y sistema estructural.  
     -Variables objetivo:
    	  Predicción de Indice de daño (escala 0-1, variable continua principal para regresión).  
    	  Predicción de deriva de entrepiso.  
      	Predicción de cortante en la base.  
## Marco VDS (Predictibilidad, Computabilidad y Estabilidad)
  -Predictibilidad. Se medirá el desempeño del modelo para predecir el índice de daño u la deriva máxima utilizando métricas de regresión.  
  -Computabilidad: Todo el flujo de trabajo será implementado a través de Python.  
  -Estabilidad: Se aplicará validación cruzada.
 
## Fuente
    Pre-Earthquake Prediction Dataset
