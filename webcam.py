import cv2
import subprocess
import os
import numpy as np
import time

# --- 1. CONFIGURACIÓN GLOBAL ---
HASKELL_EXECUTABLE = "./fun_cat_classifier" 
CAPTURE_FILE = "_frame.jpg" 

# --- 2. FUNCIÓN DE EXTRACCIÓN (De data_extractor.py) ---
def extract_color_histogram(image_path, bins=(8, 8, 8)):
    """
    Carga una imagen y calcula su histograma de color 3D (RGB) normalizado.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"ERROR: No se pudo cargar la imagen temporal '{image_path}'")
            return None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hist = cv2.calcHist([image_rgb], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist.flatten()
    except Exception as e:
        print(f"Error extrayendo histograma: {e}")
        return None

# --- 3. FUNCIÓN DE DETECCIÓN (Llama a Haskell) ---

def detect_cat(feature_vector):
    """
    Llama a 'fun_cat_classifier'
    pasándole el vector de características como un string.
    """
    
    query_string = ",".join(map(str, feature_vector))
    
    try:
        # --- ¡EL ARREGLO! ---
        # Se eliminó el 'timeout=...'
        # Ahora Python esperará indefinidamente a que Haskell termine.
        result = subprocess.run(
            [HASKELL_EXECUTABLE, query_string],
            capture_output=True,
            text=True
            # SIN TIMEOUT
        )
        
        output = result.stdout.strip()
        
        if result.returncode != 0:
            print(f"Error de Haskell (stderr): {result.stderr.strip()}")
            return "Error de Haskell"
        
        # Tu Main.hs imprime la predicción en la última línea
        prediction = output.split('\n')[-1]
        return prediction

    except Exception as e:
        print(f"Error en detección: {e}")
        return f"Error: {e}"

# --- 4. BUCLE PRINCIPAL ---

def main_loop():
    """
    Abre la webcam y ejecuta el bucle de detección.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara")
        return
        
    print("Presiona 'q' para salir, 'espacio' para analizar el fotograma")
    status = "Presiona 'espacio'"
    color = (255, 255, 255) # Blanco

    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        
        display_frame = frame.copy()
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
            
        if key == ord(' '): 
            print("¡Analizando fotograma! (Esto puede tardar...)")
            status = "Analizando... (Haskell KNN)"
            color = (255, 255, 0) # Amarillo
            
            cv2.putText(display_frame, status, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow('Clasificador KNN de Gatos (Haskell)', display_frame)
            cv2.waitKey(1) 
            
            # 1. Guarda la captura
            cv2.imwrite(CAPTURE_FILE, frame)
            
            # 2. Extrae características (Python)
            vector = extract_color_histogram(CAPTURE_FILE)
            
            prediction = "Error: Vector nulo"
            if vector is not None:
                # 3. Llama a Haskell (y espera el tiempo que sea necesario)
                prediction = detect_cat(vector)
            
            status = f"DETECTADO: {prediction}"
            is_error = "Error" in prediction
            color = (0, 0, 255) if is_error else (0, 255, 0)
            print(f"Resultado: {status}")
        
        # Dibuja el estado actual en el frame
        cv2.putText(display_frame, status, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow('Clasificador KNN de Gatos (Haskell)', display_frame)
        
    cap.release()
    cv2.destroyAllWindows()
    print("Cerrando programa.")
    if os.path.exists(CAPTURE_FILE):
        os.remove(CAPTURE_FILE)

# --- 5. PUNTO DE ENTRADA ---
if __name__ == "__main__":
    
    if not os.path.exists(HASKELL_EXECUTABLE):
        print(f"Error: No se encuentra el ejecutable '{HASKELL_EXECUTABLE}'")
    elif not os.path.exists("training_data.csv"):
        print("Error: No se encuentra 'training_data.csv'")
    else:
        main_loop()