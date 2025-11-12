# 🐱 Proyecto: Clasificador Funcional de Gatos

## 🎯 Objetivo del Proyecto

Implementar un clasificador de imágenes robusto utilizando la **Programación Funcional Pura** (Haskell) para el núcleo del algoritmo, y una interfaz web moderna (Python) para la interacción con el usuario. El objetivo final es clasificar una imagen de entrada como una raza específica de gato (e.g., Persa, Egipcio, Común) o como "No-Gato".

### Requisitos
- tener instalado el GHCup

  curl --proto '=https' --tlsv1.2 -sSf https://get-ghcup.haskell.org | sh

- crear entorno virtual
  pip install opencv-python numpy


### Test
    cabal update
    TEST_VECTOR=$(head -n 1 training_data.csv | cut -d ',' -f 2-)
    cabal run fun_cat_classifier "$TEST_VECTOR"

#### Crear binario
    cabal install --installdir=. fun_cat_classifier


## 💡 Fundamentos Teóricos

### 1. Principios de Programación Funcional (Haskell)

El proyecto se centra en demostrar la capacidad de Haskell para construir algoritmos de ML complejos de forma **declarativa y verificable**.

* **Núcleo de Funciones Puras:** Las funciones de **Distancia Euclidiana** y el **Algoritmo KNN** (incluida la votación multi-clase) serán estrictamente puras.
* **Modelado con ADTs:** Se utilizarán **Tipos de Datos Algebraicos (ADTs)** para definir claramente el dominio:
    * `data Label = Persian | Egyptian | Common | NotCat deriving (Show, Eq)`
    * `data LabeledPoint = LabeledPoint { ... }`
* **Composición Funcional:** La lógica de clasificación se construirá mediante la composición de funciones (`.`) y el uso intensivo de **Funciones de Orden Superior** (`map`, `filter`, `foldl`, `sortBy`).

### 2. Algoritmo KNN Multi-Clase

* **Base:** El algoritmo **K-Nearest Neighbors** se mantiene, clasificando una nueva imagen basándose en la mayoría de sus $K$ vecinos más cercanos en el espacio de características.
* **Votación Multi-Clase:** La función `majorityVote` se actualizará para manejar cuatro o más etiquetas, contando las ocurrencias de cada raza y de "NotCat" para determinar la clase ganadora.

2. Algoritmo KNN Multi-Clase

    Base: El algoritmo K-Nearest Neighbors (KNN) clasifica una nueva instancia (imagen) basándose en las clases de sus K vecinos más cercanos en el espacio de características.

    Teoría Clave: La Distancia Euclidiana es la métrica utilizada para cuantificar la "similitud" entre el vector de características de la imagen a clasificar (q) y cada vector de características de las imágenes de entrenamiento (p).

    La fórmula, para dos puntos p y q en un espacio de n dimensiones (características), se define como:
    d(p,q)=i=1∑n​(pi​−qi​)2​

    Esta métrica se implementará como una función pura en Haskell.

## ⚙️ Arquitectura Híbrida y Comunicación

El proyecto se divide en dos entornos que interactúan a través de la Línea de Comandos (CLI) para mantener el núcleo Haskell lo más puro posible.

| Componente | Lenguaje | Módulos Clave | Tarea Principal |
| :--- | :--- | :--- | :--- |
| **I/O, Extracción** | **Python** | `data_extractor.py` | Carga, Pre-procesamiento y Extracción de **Vectores de Características** (e.g., Histogramas de Color). |
| **API Web** | **Python (Flask/FastAPI)** | `api.py` | Recibe la imagen, ejecuta la extracción de características y orquesta la llamada al ejecutable de Haskell. |
| **Núcleo ML** | **Haskell** | `Classifier.hs`, `Main.hs` | **Clasificación Pura:** Recibe el vector de características de Python a través de la CLI, ejecuta el KNN y devuelve la predicción. |

### Diagrama de Flujo de Predicción



1.  **Usuario** sube la imagen a la interfaz **Python/FastAPI**.
2.  **Python** usa librerías de ML para extraer el vector de características de la imagen.
3.  **Python** llama al programa compilado de Haskell (**`./fun_cat_classifier`**) a través de un *subproceso*, pasando el vector de características como argumento de **CLI**.
4.  El ejecutable **Haskell** lee el argumento, ejecuta la clasificación **pura** (KNN) y escribe la etiqueta predicha (`Persian`, `NotCat`, etc.) en la salida estándar (`stdout`).
5.  **Python** captura la salida (`stdout`) de Haskell y la retorna al usuario como respuesta de la API.

## 🛠️ Plan de Construcción (Etapas)

### Fase 1: Recolección y Preparación de Datos (Python)

* **Datos:** Recolectar un conjunto de imágenes etiquetadas con al menos tres clases de gatos y una clase `NotCat`.
* **Extracción:** Implementar `data_extractor.py` para leer las imágenes, extraer los **vectores de características** (normalizados) y generar el archivo **`training_data.csv`** con la etiqueta y sus correspondientes valores.

### Fase 2: Implementación del Núcleo Funcional (Haskell)

1.  **Tipos:** Definir las estructuras de datos, incluyendo el `data Label` multi-clase.
2.  **Lógica:** Implementar las funciones **puras**:
    * `euclideanDistance :: FeatureVector -> FeatureVector -> Double`
    * `majorityVote :: [Label] -> Label` (para manejar la votación multi-clase).
    * `kNearestNeighbors :: Int -> [LabeledPoint] -> FeatureVector -> Label`
3.  **CLI I/O:** Implementar el `main` de Haskell para que:
    * Cargue el `training_data.csv` (una vez) al inicio.
    * Parseé el vector de características de consulta desde los argumentos de la línea de comandos.
    * Imprima el resultado de la función `kNearestNeighbors` en `stdout`.

### Fase 3: Interfaz Web (Python FastAPI/Flask)

1.  **API REST:** Crear el *endpoint* `/predict` para manejar la solicitud `POST` y la carga del archivo de imagen.
2.  **Orquestación:** Dentro del *endpoint* `/predict`:
    * Llamar a la función de extracción de características.
    * Utilizar `subprocess` para ejecutar el binario de Haskell con las características como argumento.
    * Capturar la salida (`stdout`) de Haskell (que será la etiqueta predicha).
    * Devolver la predicción al usuario.

Este enfoque garantiza que se cumplen todos los requisitos: Haskell para el núcleo funcional, clasificación multi-clase, y una interfaz moderna con Python.
