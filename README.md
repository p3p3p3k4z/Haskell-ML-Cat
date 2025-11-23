# 🐱 Haskell ML Cat: KNN vs Deep Learning

Este proyecto es un **sistema de benchmarking híbrido** que compara dos paradigmas de Inteligencia Artificial aplicados a la visión por computadora:

1.  **Modelo Clásico (KNN):** Implementado desde cero en **Haskell Puro** utilizando algoritmos matemáticos clásicos.
2.  **Deep Learning (CNN):** Implementado con **Hasktorch** (bindings de Haskell para PyTorch), utilizando una red neuronal ResNet-18 pre-entrenada.

Todo el sistema está orquestado por una interfaz gráfica moderna en **Python (Pygame)** que gestiona la webcam, la visualización y la comunicación entre procesos.

-----

## Arquitectura del Sistema

El proyecto utiliza una arquitectura de **Software Híbrido** basada en comunicación por procesos (IPC) y sistema de archivos.

[Image of hybrid software architecture diagram]

1.  **Frontend (Python/Pygame):**
      * Controla la Webcam y la Interfaz Gráfica (GUI).
      * Captura fotogramas y los guarda temporalmente.
      * Orquesta la ejecución de los binarios de Haskell.
2.  **Backend 1: KNN Classifier (Haskell Puro):**
      * Recibe un vector de características extraído por Python (Histograma de Color).
      * Calcula distancias euclidianas contra una base de datos (`training_data.csv`).
      * Determina si la imagen es un gato y su raza mediante votación mayoritaria.
3.  **Backend 2: Hasktorch Detector (Haskell + C++):**
      * Recibe la ruta de la imagen.
      * Carga un modelo `ResNet-18` serializado (`.pt`).
      * Utiliza tensores y operaciones convolucionales (vía `libtorch` C++) para clasificar el objeto.

-----

## 💡 Fundamentos Teóricos

### 1\. K-Nearest Neighbors (KNN)

El modelo clásico implementado es un algoritmo de **Aprendizaje Supervisado**.

  * **Aprendizaje Supervisado:** El modelo aprende de un conjunto de datos previamente etiquetado (el archivo `training_data.csv`). Cada histograma de color (entrada) está asociado explícitamente a una raza de gato (salida esperada).
  * **Aprendizaje Perezoso (Lazy Learning):** A diferencia de las redes neuronales que "entrenan" y generalizan reglas, el KNN simplemente **memoriza** los datos de entrenamiento. No construye un modelo interno hasta que se le hace una pregunta.
      * *Consecuencia:* El "entrenamiento" es instantáneo (solo cargar datos), pero la predicción es costosa computacionalmente, ya que debe comparar la nueva imagen contra *todas* las imágenes guardadas cada vez.

### 2\. Haskell y el Paradigma Funcional

El uso de Haskell fuerza la aplicación estricta de principios funcionales, garantizando un código robusto y matemáticamente verificable.

| Concepto | Explicación Teórica | Aplicación en el Proyecto |
| :--- | :--- | :--- |
| **Funciones Puras** | Una función siempre devuelve el mismo resultado para las mismas entradas y **no tiene efectos secundarios** (no modifica variables globales ni realiza E/S oculta). | El algoritmo **KNN completo** (cálculo de distancia euclidiana, votación, clasificación) reside en `Classifier.hs` como un conjunto de funciones puras. |
| **Inmutabilidad** | Una vez creada una estructura de datos, no puede modificarse. Para "cambiarla", se crea una copia nueva con los cambios. | El **conjunto de entrenamiento** (`[LabeledPoint]`) es inmutable. Esto elimina errores de concurrencia: el modelo nunca puede ser corrompido por una escritura accidental. |
| **Tipos de Datos Algebraicos (ADTs)** | Estructuras de datos compuestas que expresan la forma exacta de los datos. Incluyen Tipos Producto (AND) y Tipos Suma (OR). | Usamos `data Label = Persian | Egyptian | ...` para modelar las clases de forma estricta, impidiendo estados inválidos. |
| **Evaluación Perezosa (Lazy Evaluation)** | Haskell no ejecuta una operación hasta que su resultado es estrictamente necesario. | Permite definir estructuras infinitas o muy grandes. En `Main.hs`, la lista de 1000 etiquetas de ImageNet no se carga en RAM hasta que se necesita imprimir un resultado específico. |
| **Mónada IO** | Un mecanismo para aislar los efectos secundarios (impuros) del resto del código puro. | En `Main.hs`, toda interacción con el sistema de archivos o la consola está confinada dentro del bloque `main :: IO ()`, manteniendo el resto del sistema puro. |

-----

## 🛠️ Guía de Instalación y Compilación

### Requisitos

  * **Linux/macOS** (Recomendado por dependencias de C++).
  * **GHC & Cabal:** (Vía GHCup).
  * **Python 3.x**
  * **Librerías de Sistema:** `zenity` (para diálogos de archivo).

### Paso 1: Compilar el Entorno Haskell (Hasktorch)

Este es el paso más crítico. Debemos compilar el "cerebro" de Deep Learning.
Nota: revisar docs/Hasktorch.md y clonar el repo de Hasktorch

```bash
cd load-torchscript
# Descargar modelo y configurar entorno
./setup-cabal.sh
python3 gen_resnet.py
# Compilar el ejecutable (esto puede tardar)
cabal build load-torchscript
# Copiar el binario a la carpeta local para acceso rápido
cabal install load-torchscript --install-method=copy --installdir=.
```

### Paso 2: Compilar el Clasificador KNN

Es necesario compilar el código fuente de Haskell puro para generar el binario `fun_cat_classifier`.
Nota: Revisar docs/KNN.md

```bash
# Desde la raíz del proyecto
cabal build 
# Instalar el binario en la raíz
cabal install --installdir=. fun_cat_classifier
```

> **Nota:** Asegúrate de tener el archivo `training_data.csv` generado. Si no, ejecuta `python3 data_extractor.py`.

### Paso 3: Configurar Python

Instalar las dependencias para la interfaz gráfica y visión.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

*(Contenido de requirements.txt: `pygame`, `opencv-python`, `numpy`, `torch`, `torchvision`)*

-----

## Ejecución

El proyecto cuenta con un **Hub Central** que unifica todo.

```bash
python3 app.py
```

### Uso de la Interfaz:

1.  **Selecciona el Modelo:**
      * **KNN Clásico:** Verás cómo el modelo intenta clasificar razas de gatos basándose en colores.
      * **Deep Learning:** Verás el poder de ResNet detectando cualquier objeto (no solo gatos).
2.  **Selecciona la Fuente:**
      * **Archivo:** Abre un explorador para probar con imágenes descargadas.
      * **Webcam:** Abre la cámara en tiempo real.
3.  **En modo Webcam:**
      * Apunta a un objeto.
      * Presiona **[ESPACIO]** para congelar y analizar.
      * Observa la comparativa de tiempo y resultado.

-----

## Retos de Desarrollo (Bitácora)

1.  **Infierno de Enlaces (Linking Hell):** Lograr que Haskell compilara contra las librerías dinámicas de C++ (`libtorch`) requirió una configuración precisa de `cabal.project` y `LD_LIBRARY_PATH`.
2.  **Interoperabilidad de Procesos:** Python fallaba al llamar al binario de Haskell porque el entorno de la shell no se heredaba. Se solucionó implementando llamadas con `subprocess.run(..., shell=True)`, replicando el comportamiento de una terminal real.
3.  **Sincronización:** Se implementó un sistema de archivos temporales y semáforos implícitos para evitar condiciones de carrera entre la captura de la webcam (Python) y la lectura de la imagen (Haskell).

-----

## 📂 Estructura del Proyecto

```text
Haskell-ML-Cat/
├── app.py                  # Hub Principal (Interfaz Pygame)
├── fun_cat_classifier      # Ejecutable Binario KNN (Haskell Puro)
├── training_data.csv       # Base de datos de conocimiento KNN
│
├── load-torchscript/       # Módulo de Deep Learning
│   ├── load-torchscript    # Ejecutable Binario CNN (Haskell+C++)
│   ├── resnet_model.pt     # Modelo neuronal serializado
│   └── Main.hs             # Código fuente Haskell (Inferencia)
│
├── src/                    # Código fuente KNN
│   ├── Classifier.hs    # Lógica pura (Distancia Euclidiana)
│   └── Main.hs   # Punto de entrada KNN
│
└── Cat_Breed/              # Dataset de imágenes 
```

- Si deseas puedes descargarla aqui: [Gatos](https://drive.google.com/drive/folders/1MeLRuxliR174CcY40eb60t0jsVF0afZ7?usp=sharing) 