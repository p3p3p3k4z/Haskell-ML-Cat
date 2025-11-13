import os
import cv2
import numpy as np
import subprocess
import sys

# --- 1. Importar Tkinter para el explorador de archivos ---
from tkinter import Tk, filedialog

# --- 2. Núcleo de Extracción (Idéntico) ---
def extract_color_histogram(image_path, bins=(8, 8, 8)):
    """
    Carga una imagen y calcula su histograma de color 3D (RGB) normalizado.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"ERROR: No se pudo cargar la imagen '{image_path}'.", file=sys.stderr)
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hist = cv2.calcHist([image_rgb], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist.flatten()
    except Exception as e:
        print(f"Error extrayendo histograma de '{image_path}': {e}", file=sys.stderr)
        return None

# --- 3. Main del script (¡Actualizado!) ---
def main():
    print("--- Script de Prueba Local del Clasificador ---")
    
    # --- ¡NUEVO! Abrir explorador de archivos ---
    print("Abriendo explorador de archivos para seleccionar imagen...")
    
    # Ocultar la ventana raíz de tkinter
    root = Tk()
    root.withdraw()
    
    test_image_path = filedialog.askopenfilename(
        title="Selecciona una imagen de prueba",
        filetypes=(("Archivos de imagen", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                   ("Todos los archivos", "*.*"))
    )
    
    # Salir si el usuario presiona "Cancelar"
    if not test_image_path:
        print("No se seleccionó ninguna imagen. Saliendo.")
        sys.exit(0)
    
    # --- El resto del script es igual que antes ---
    
    # Ruta al ejecutable de Haskell
    HASKELL_EXECUTABLE = "./fun_cat_classifier"

    # Validar que el ejecutable de Haskell existe
    if not os.path.exists(HASKELL_EXECUTABLE):
        print(f"¡ERROR! No se encuentra el ejecutable de Haskell en '{HASKELL_EXECUTABLE}'.", file=sys.stderr)
        print("Asegúrate de haber corrido 'cabal install --installdir=. fun_cat_classifier'", file=sys.stderr)
        sys.exit(1)

    print(f"\nProcesando imagen: '{test_image_path}'")

    # 1. Extraer características
    features = extract_color_histogram(test_image_path)
    if features is None:
        print("La extracción de características falló. Saliendo.", file=sys.stderr)
        sys.exit(1)

    print(f"Características extraídas (primeros 5): {features[:5]}...")

    # 2. Formatear el vector
    query_string = ",".join(map(str, features))

    # 3. Llamar a Haskell
    print(f"\nLlamando a Haskell: '{HASKELL_EXECUTABLE} {query_string[:50]}...'")
    try:
        process = subprocess.run(
            [HASKELL_EXECUTABLE, query_string],
            capture_output=True,
            text=True,
            check=True
        )
        
        # 4. Capturar y mostrar la salida
        raw_output = process.stdout.strip()
        prediction = raw_output.split('\n')[-1]
        
        print("\n--- Salida Completa de Haskell ---")
        print(raw_output)
        print("----------------------------------")
        print(f"\nPredicción Final de Haskell: ¡{prediction}!")

    except subprocess.CalledProcessError as e:
        print(f"\nERROR: El programa de Haskell terminó con un error.", file=sys.stderr)
        print(f"Salida de error de Haskell (stderr): \n{e.stderr}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR inesperado al llamar a Haskell: {e}", file=sys.stderr)
        sys.exit(1)

# Punto de entrada
if __name__ == "__main__":
    main()
