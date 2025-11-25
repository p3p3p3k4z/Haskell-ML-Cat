# Histogramas de Color y Clasificación KNN

## 1. Introducción: ¿Qué ve la computadora?

Para un ser humano, identificar un gato en una foto es trivial. Para una computadora, una imagen digital es solo una inmensa matriz de números.

Un algoritmo de aprendizaje automático "clásico" como K-Nearest Neighbors (KNN) no puede procesar eficientemente una imagen cruda de alta resolución debido a la **Maldición de la Dimensionalidad**.

**El Desafío:** Convertir una imagen compleja (ej. $1000 \times 1000$ píxeles) en una representación numérica compacta que capture su "esencia" cromática.

---

## 2. Ingeniería de Características: El Histograma de Color 3D

La solución implementada en este proyecto es la **Extracción de Características** mediante un histograma de color RGB.

### A. El Espacio de Color (El "Cubo")
Cada píxel en una imagen digital se compone de tres canales: Rojo (R), Verde (G) y Azul (B). Cada canal tiene un valor de intensidad de 0 a 255.

* **Total de colores posibles:** $256^3 = 16,777,216$.

Manejar 16 millones de dimensiones es inviable. Por ello, aplicamos una técnica de **Cuantización** (reducción de colores).

### B. La Cuantización (El "Divisor")
Dividimos el cubo de colores RGB en bloques más grandes (llamados *bins* o cubetas).

* **Configuración del Proyecto:** 8 divisiones por canal.
* **Tamaño de cada Bin:** $256 / 8 = 32$.
    * Cualquier valor de rojo entre 0 y 31 cae en el bin 0.
    * Cualquier valor entre 32 y 63 cae en el bin 1.
    * ...y así sucesivamente.



### C. Construcción del Vector
El algoritmo recorre cada píxel de la imagen y "vota" en el bin correspondiente.

1.  **Entrada:** Una imagen de $N \times M$ píxeles.
2.  **Proceso:**
    * Leer píxel $(R, G, B)$.
    * Calcular índices: $i = R/32, j = G/32, k = B/32$.
    * Incrementar el contador en la posición $(i, j, k)$ del histograma 3D.
3.  **Aplanamiento:** El cubo de $8 \times 8 \times 8$ se convierte en una lista lineal de $512$ números.
4.  **Normalización:** Dividimos cada contador por el número total de píxeles.
    * *Resultado:* Un vector donde la suma de todos los elementos es 1.0 (representando frecuencias relativas).

**Resultado Final:** Un vector de características de **512 dimensiones** ($8 \times 8 \times 8$).

---

## 3. El Algoritmo KNN (K-Nearest Neighbors)

Una vez que todas las imágenes (entrenamiento y prueba) se han convertido en vectores de 512 dimensiones, entra en juego el algoritmo de clasificación implementado en Haskell.

### A. Fundamentos
KNN es un algoritmo de **Aprendizaje Supervisado Perezoso (Lazy Learning)**.
* **Supervisado:** Aprende de ejemplos etiquetados (el archivo `training_data.csv`).
* **Perezoso:** No construye un modelo matemático durante el "entrenamiento". Simplemente memoriza todos los vectores de entrenamiento. La computación ocurre únicamente al momento de clasificar una nueva imagen.

### B. La Métrica de Similitud: Distancia Euclidiana
Para determinar si dos imágenes son "parecidas", calculamos la distancia geométrica entre sus vectores en el espacio de 512 dimensiones.

La fórmula implementada en `Classifier.hs` es:

$$d(p, q) = \sqrt{\sum_{i=1}^{512} (p_i - q_i)^2}$$

* $p$: Vector de la imagen de la webcam (el "query").
* $q$: Un vector de la base de datos de entrenamiento.

Si $d(p,q)$ es cercano a 0, significa que las dos imágenes tienen una distribución de colores casi idéntica.



[Image of KNN classification diagram]


### C. Clasificación por Voto Mayoritario
1.  Se calculan las distancias entre la imagen de entrada y **todas** las imágenes de entrenamiento (aprox. 20,000).
2.  Se ordenan de menor a mayor distancia.
3.  Se seleccionan los **$K$** vecinos más cercanos (en nuestro caso, $K=5$).
4.  Se cuentan las etiquetas de esos 5 vecinos.
    * *Ejemplo:* [Persa, Persa, Siamés, Persa, No-Gato].
5.  La etiqueta con más votos ("Persa" con 3 votos) es la predicción final.

---

## 4. Interpretación de Resultados

### Caso de Éxito
* **Input:** Foto de un gato naranja.
* **Histograma:** Picos altos en los bins correspondientes a naranja y marrón.
* **Vecinos:** El algoritmo encuentra fotos de gatos "Abisinios" (que suelen ser de color cobrizo/naranja) con histogramas similares.
* **Resultado:** "Abyssinian".

### Limitaciones (Por qué falla a veces)
Este modelo sufre de **Ceguera Semántica**.
* Solo ve colores, no formas.
* Una foto de una **naranja (fruta)** tendrá un histograma casi idéntico al de un **gato naranja**.
* El KNN verá una distancia muy corta entre ambos y podría clasificar la fruta como un gato.

### Solución Implementada
Para mitigar esto, se añadió la clase explícita **`NotCat`** con imágenes variadas (frutas, autos, paisajes). Si el vecino más cercano a la fruta es una imagen de la carpeta `NotCat`, el sistema responderá correctamente "NO ES UN GATO".

---

## 5. Resumen Técnico

| Componente | Tecnología | Función |
| :--- | :--- | :--- |
| **Extracción** | Python (OpenCV) / Haskell (JuicyPixels) | Convierte Imagen $\to$ Vector (512 floats) |
| **Modelo** | K-Nearest Neighbors (KNN) | Clasificación basada en similitud |
| **Métrica** | Distancia Euclidiana | Medida de similitud ($L^2$ norm) |
| **Tipo de Aprendizaje** | Supervisado / Lazy | Memorización de instancias |
| **Espacio de Características** | RGB Histogram (8 bins/channel) | Representación de datos |