### Optimización de Inferencia: Procesamiento por Lotes (Batch Processing) en Haskell

#### 1\. Planteamiento del Problema

La implementación original del clasificador basado en **Hasktorch** operaba bajo un esquema de ejecución secuencial unitaria. Para realizar la detección de objetos en múltiples zonas de la imagen (mediante la técnica de Ventana Deslizante), el sistema debía invocar el binario de Haskell repetidamente, una vez por cada región de interés.

Esto introducía un cuello de botella crítico conocido como **Overhead de Inicialización**. Dado que la carga del modelo ResNet-18 (aprox. 100 MB de tensores) en memoria es una operación costosa en términos de E/S, repetirla $N$ veces por fotograma hacía inviable la ejecución en tiempo real, generando una latencia acumulativa inaceptable.

#### 2\. Solución Implementada

Se rediseñó la arquitectura de comunicación entre la capa de presentación (Python) y el motor de inferencia (Haskell) para implementar **Procesamiento por Lotes (Batch Processing)**.

Bajo este nuevo enfoque, el script de Python segmenta la imagen en 5 regiones estratégicas y envía la lista completa de rutas de archivo al ejecutable de Haskell en una única llamada al sistema. El binario de Haskell fue refactorizado para recibir esta lista variable de argumentos y procesarla en una sola sesión de ejecución.

#### 3\. Implementación Técnica en Haskell

La optimización se logró aprovechando las capacidades del paradigma funcional para el manejo de listas y efectos secundarios controlados (Mónada IO).

Se modificó el punto de entrada (`Main.hs`) para utilizar la función de orden superior **`mapM_`** (Map Monadic). El flujo lógico es el siguiente:

1.  **Carga Estática del Modelo:** Se carga el grafo computacional de la red neuronal en memoria una única vez al inicio de la ejecución.
2.  **Aplicación Parcial:** Se crea una función de inferencia que ya contiene el modelo cargado en su cierre (closure).
3.  **Iteración Eficiente:** La función `mapM_` aplica esta función de inferencia sobre la lista de imágenes de entrada de manera secuencial.

**Pseudocódigo de la lógica implementada:**

```haskell
-- Se carga el modelo en memoria (Operación costosa: O(1))
model <- loadScript "resnet_model.pt"

-- Se aplica la inferencia a toda la lista sin recargar el modelo
-- mapM_ :: (a -> IO b) -> [a] -> IO ()
mapM_ (runInference model) listaDeImagenes
```

#### 4\. Resultados e Impacto

Esta modificación transformó la complejidad temporal de la operación de carga del modelo. Pasamos de una complejidad lineal $O(N)$ (donde $N$ es el número de regiones) a una complejidad constante $O(1)$ por ciclo de análisis.

El resultado es un sistema capaz de clasificar múltiples objetos simultáneamente dentro de un mismo fotograma de video, manteniendo la estabilidad y el rendimiento necesarios para una aplicación de visión por computadora en tiempo real, validando así la eficiencia de Haskell en tareas de computación intensiva.