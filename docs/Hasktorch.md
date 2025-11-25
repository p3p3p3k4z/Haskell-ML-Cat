# Detector de Objetos en Tiempo Real: Arquitectura Híbrida Haskell + Python

## 1\. Resumen del Proyecto

Este proyecto implementa un sistema de visión por computadora capaz de clasificar objetos del mundo real utilizando una cámara web. La solución propone una **arquitectura híbrida por procesos** que aprovecha las fortalezas de dos paradigmas de programación:

1.  **El Núcleo Funcional (Haskell):** Un ejecutable binario (`load-torchscript`) encargado exclusivamente de la lógica de inferencia. Utiliza **Hasktorch** para cargar un modelo de Red Neuronal Convolucional (ResNet-18) pre-entrenado.
2.  **La Interfaz de Hardware (Python):** Un script (`webcam.py`) que gestiona la captura de video mediante **OpenCV**, la interacción con el usuario y la orquestación del proceso de clasificación.

El objetivo principal es demostrar la viabilidad de **Haskell** en tareas de Inteligencia Artificial, resolviendo los retos de interoperabilidad con librerías de C++ (`libtorch`) y hardware.

-----

## 2\. Arquitectura del Sistema

El sistema no utiliza comunicación por red (sockets) ni librerías compartidas (FFI directo entre Python y Haskell), sino una comunicación basada en **procesos y sistema de archivos**.

[Image of hybrid software architecture diagram]

### Flujo de Datos:

1.  **Captura:** Python obtiene un fotograma de la webcam.
2.  **Persistencia:** Python guarda el fotograma en disco (`_frame_to_detect.jpg`) para evitar condiciones de carrera en memoria.
3.  **Invocación:** Python lanza el binario de Haskell como un subproceso, configurando el entorno (`shell=True`) para garantizar el acceso a las librerías dinámicas (`LD_LIBRARY_PATH`).
4.  **Inferencia Pura:** Haskell lee la imagen, la transforma en un Tensor, ejecuta el modelo y escribe el resultado en la salida estándar (`stdout`).
5.  **Presentación:** Python captura el texto, lo procesa y superpone la etiqueta detectada en el video en tiempo real.

-----

## 3\. Análisis de Programación Funcional (El Código Haskell)

El archivo `Main.hs` es un ejemplo práctico de cómo los principios funcionales gestionan la complejidad de la IA.

### A. La Mónada IO: Aislamiento de Efectos

En Haskell, las funciones deben ser puras (sin efectos secundarios). Sin embargo, la IA requiere leer archivos y modelos.

  * **Implementación:** La función `main :: IO ()` actúa como una "receta" de ejecución.
  * **Notación `do`:** Permite secuenciar acciones impuras.
  * **Bind (`<-`):** Extrae valores puros de acciones impuras. Por ejemplo, `model <- loadScript ...` carga el modelo desde el disco y entrega un objeto `model` inmutable al resto del programa.

### B. Composición de Funciones (El Pipeline de Datos)

El pre-procesamiento de la imagen (convertir una foto en números que la IA entienda) se realiza mediante una tubería de transformación declarativa:

```haskell
-- Código fuente de Main.hs
let img'' = toType Float $ hwc2chw $ normalize $ divScalar (255.0 :: Float) $ toType Float $ fromDynImage $ I.ImageRGB8 img'
```

[Image of data transformation pipeline]

El operador `$` pasa el resultado de la derecha a la función de la izquierda. El flujo es:

1.  `img'` (Imagen cruda) $\rightarrow$ `fromDynImage`
2.  $\rightarrow$ `toType Float` (Conversión de tipos)
3.  $\rightarrow$ `divScalar` (Normalización aritmética)
4.  $\rightarrow$ `hwc2chw` (Transposición de dimensiones del Tensor)
5.  $\rightarrow$ `img''` (Tensor final listo para inferencia).

Esto garantiza que no haya estados mutables intermedios; solo transformaciones de datos.

### C. Seguridad de Tipos y Pattern Matching

Haskell obliga a manejar los errores en tiempo de compilación. La carga de la imagen utiliza el tipo `Either`:

```haskell
case mimg of
  Left err -> print err       -- Manejo explícito del error
  Right (img_, _) -> do ...   -- Continuación solo si hay éxito
```

A diferencia de un `try-catch` imperativo, aquí es imposible "olvidar" manejar el caso de fallo, ya que el programa no compilaría.

### D. Inmutabilidad y Evaluación Perezosa

El programa define una lista de 1000 etiquetas (`labels`) para las clases de ImageNet.

  * **Inmutabilidad:** La lista es una definición constante.
  * **Evaluación Perezosa (Lazy Evaluation):** Haskell no carga los 1000 strings en memoria al iniciar. Solo cuando la inferencia termina y obtenemos un índice (ej. `386`), el runtime evalúa y asigna memoria para *ese* string específico necesario para imprimirlo.

-----

## 4\. Guía de Instalación y Ejecución

Pasos para replicar el proyecto en un entorno Linux.

### Requisitos Previos

  * GHC (Compilador de Haskell)
  * Cabal (Gestor de paquetes)
  * Python 3

### Paso 1: Compilación del Entorno Haskell

Desde la carpeta raíz `hasktorch/`:

1.  **Configurar dependencias:**
    ```bash
    ./setup-cabal.sh
    ```
2.  **Compilar el proyecto:**
    ```bash
    cabal build examples
    ```
3.  **Instalar el binario:**
    Copiamos el ejecutable a la carpeta del ejemplo para facilitar su acceso.
    ```bash
    cabal install load-torchscript --install-method=copy --installdir=./examples/load-torchscript
    ```

### Paso 2: Configuración de Python

Desde la carpeta `examples/load-torchscript/`:

1.  **Crear entorno virtual:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
2.  **Instalar librerías:**
    ```bash
    pip install torch torchvision opencv-python
    ```
3.  **Descargar el modelo:**
    Ejecutamos el script que baja el modelo ResNet-18 y lo convierte a formato TorchScript. Red Neuronal Convolucional (CNN)
    ```bash
    python3 gen_resnet.py
    ```

### Paso 3: Ejecución
Por si desea probar el ejecutable
```bash
(venv) m4r10@opensuse:~/Documents/hasktorch/examples/load-torchscript> cabal run load-torchscript -- ./resnet_model.pt ./elephant.jpg
"--labels--"
["African_elephant","tusker","water_buffalo","Indian_elephant","warthog"]
"--scores--"
Tensor Float [1,5] [[ 0.6939   ,  0.1715   ,  4.2850e-2,  4.2793e-2,  1.3161e-2]]


m4r10@opensuse:~/Documents/Cosas de haskell/hasktorch/examples/load-torchscript> ./load-torchscript ./resnet_model.pt ./chamaleon.jpeg 
"--labels--"
["African_chameleon","tree_frog","green_lizard","common_iguana","leatherback_turtle"]
"--scores--"
Tensor Float [1,5] [[ 0.8308   ,  2.2529e-2,  2.0475e-2,  1.8490e-2,  1.5495e-2]]
```


Para iniciar el sistema de detección en tiempo real:

```bash
python3 webcam_torch.py
```

  * **Uso:** Apunta la cámara a un objeto y presiona **`Espacio`**. El sistema congelará el frame, lo analizará con Haskell y mostrará la predicción.

-----

## 5\. Retos de Desarrollo y Soluciones

Durante el desarrollo de este proyecto, nos enfrentamos a desafíos significativos de ingeniería de software:

1.  **El "Infierno de Enlaces" (Linking Hell):** El mayor obstáculo fue lograr que el código Haskell compilara contra las librerías de C++ (`libtorch`). Se resolvió utilizando la configuración correcta de `cabal.project` y scripts de configuración automática.
2.  **Interoperabilidad de Procesos:** Al llamar al ejecutable de Haskell desde Python, fallaba silenciosamente.
      * **Diagnóstico:** Python creaba un entorno "limpio" que no incluía la variable `LD_LIBRARY_PATH`, por lo que el binario no encontraba las librerías dinámicas de C++.
      * **Solución:** Implementamos una llamada al sistema usando `subprocess.run(..., shell=True)`, lo que fuerza al programa a heredar el entorno completo de la shell del usuario, resolviendo las dependencias dinámicas.
3.  **Sincronización:** Inicialmente, usábamos archivos temporales que se borraban demasiado rápido. Se solucionó implementando un sistema de escritura síncrona en disco (`_frame_to_detect.jpg`) antes de invocar al proceso de Haskell.