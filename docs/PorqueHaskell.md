## 💡 Introducción a Haskell: Programación Funcional Pura

**Haskell** es un lenguaje de programación funcional estandarizado, conocido por su **pureza** y su robusto sistema de tipos. Es ideal para tu materia porque fuerza la aplicación estricta de los principios funcionales.

| Concepto | Explicación | Relevancia en FunCat |
| :--- | :--- | :--- |
| **Funciones Puras** | Una función siempre devuelve el mismo resultado para las mismas entradas y **no tiene efectos secundarios** (no modifica variables fuera de su alcance ni realiza E/S). | El algoritmo **KNN completo** (distancia, voto, clasificación) será un conjunto de funciones puras. |
| **Inmutabilidad** | Una vez que se crea una estructura de datos, no se puede cambiar. En lugar de modificarla, se crea una nueva versión con los cambios. | El **conjunto de entrenamiento** (`[LabeledPoint]`) es inmutable. El "entrenamiento" es simplemente cargar esta estructura en memoria. |
| **Tipos de Datos Algebraicos (ADTs)** | Se usan para construir estructuras de datos complejas. Incluyen **tipos producto** (registros) y **tipos suma** (enumeraciones). | Usamos tipos producto para `LabeledPoint` y tipos suma para `Label` (`Persian | Egyptian | NotCat`). |
| **Evaluación Perezosa (Lazy Evaluation)** | Haskell no evalúa una expresión hasta que su resultado es realmente necesario. | Permite trabajar con estructuras de datos potencialmente infinitas y escribir código muy modular y optimizado. |
| **Mónadas (IO Monad)** | Es un mecanismo para aislar y gestionar los **efectos secundarios** (como la lectura de archivos o la impresión en pantalla) dentro de un contexto controlado, manteniendo la pureza del resto del código. | Necesitarás la **Mónada IO** solo para leer el archivo CSV y recibir/enviar datos a través de la CLI. |

-----

## 🗺️ Resumen de la Construcción del Proyecto FunCat

Tu proyecto se construirá en tres etapas principales, manteniendo la lógica del clasificador **estrictamente en Haskell** para cumplir con los requisitos académicos.

| Etapa | Componente Principal | Tarea de Haskell | Output Final |
| :--- | :--- | :--- | :--- |
| **1. Preparación de Datos** | **Python Script** | *Ninguna* | Archivo `training_data.csv` (etiquetas y vectores de características). |
| **2. Núcleo Funcional** | **Haskell (`Classifier.hs`)** | Implementar las funciones **puras** `euclideanDistance`, `majorityVote` y `kNearestNeighbors`. | Un **ejecutable binario** (`./fun_cat_classifier`). |
| **3. Interfaz de Usuario** | **Python (FastAPI/Flask)** | Recibir el vector de consulta, ejecutar `kNearestNeighbors`, e imprimir la etiqueta resultante a la consola. | Una **API REST** funcional que da una predicción al usuario. |

### Detalle del Flujo de Predicción (La Parte Clave)

Cuando un usuario suba una imagen:

1.  **Python** extrae el vector de características de la nueva imagen (ej. `[0.12, 0.45, 0.99]`).
2.  **Python** llama al ejecutable de Haskell a través de la línea de comandos:
    ```bash
    $ ./fun_cat_classifier 0.12,0.45,0.99
    ```
3.  **Haskell (dentro de `main`):**
      * Lee el conjunto de entrenamiento (solo una vez).
      * Parsear el argumento de la CLI a un `FeatureVector`.
      * Llama a la función **pura**: `kNearestNeighbors K trainingData queryVector`.
      * El resultado es una etiqueta (ej. `Persian`).
      * Haskell imprime: `Persian`
4.  **Python** captura `Persian` y lo muestra en la interfaz web.

Esto asegura que todo el **algoritmo matemático y la lógica de decisión** residan en el código puro de Haskell.
