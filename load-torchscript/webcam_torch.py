import cv2
import subprocess
import os
import time

# --- 1. CONFIGURACIÓN ---

# Directorio donde está este script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Rutas a los archivos (Construimos rutas absolutas)
# Asegúrate de que el nombre de la carpeta sea correcto ('load-torchscript')
HASKELL_EXE = os.path.join(SCRIPT_DIR, "load-torchscript")
MODEL_PATH = os.path.join(SCRIPT_DIR, "resnet_model.pt")
CAPTURE_FILE = os.path.join(SCRIPT_DIR, "_webcam_capture.jpg")

def check_files():
    """Verifica que existan el ejecutable y el modelo."""
    if not os.path.exists(HASKELL_EXE):
        print(f"Error: No encuentro el ejecutable en: {HASKELL_EXE}")
        return False
    if not os.path.exists(MODEL_PATH):
        print(f"Error: No encuentro el modelo en: {MODEL_PATH}")
        return False
    return True

def get_prediction_from_haskell(image_path):
    """
    Ejecuta el comando y parsea la salida de forma robusta.
    """
    
    # 1. Construimos el comando con comillas para manejar espacios
    command_string = f'"{HASKELL_EXE}" "{MODEL_PATH}" "{image_path}"'
    
    try:
        # 2. Ejecutamos con shell=True
        result = subprocess.run(
            command_string,
            shell=True,
            capture_output=True,
            text=True,
            timeout=5
        )
        
        # 3. Analizamos la salida
        output = result.stdout.strip()
        
        if result.returncode != 0:
            print(f"Error stderr: {result.stderr}")
            return "Error Haskell"

        # --- CORRECCIÓN EN EL PARSEO ---
        lines = output.splitlines()
        found_index = -1
        
        # Buscamos en qué línea aparece el texto "--labels--"
        for i, line in enumerate(lines):
            if "--labels--" in line:
                found_index = i
                break
        
        if found_index != -1 and found_index + 1 < len(lines):
            # La lista de animales está en la línea SIGUIENTE
            labels_line = lines[found_index + 1]
            
            # labels_line es algo como: ["African_chameleon","tree_frog",...]
            # Limpiamos corchetes y comillas para sacar el primer nombre
            # Dividimos por comas y tomamos el primero
            first_part = labels_line.split(',')[0] 
            
            # Limpiamos caracteres extraños (corchetes, comillas)
            clean_label = first_part.replace('[', '').replace('"', '').replace("'", "").strip()
            
            return clean_label
        else:
            # Si no encontramos la etiqueta, devolvemos lo que haya (útil para depurar)
            return "No labels found"

    except subprocess.TimeoutExpired:
        return "Timeout"
    except Exception as e:
        print(f"Excepción Python: {e}")
        return "Error Python"

def main_loop():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la webcam.")
        return

    print("--- Webcam Detector ---")
    print(f"Usando ejecutable: {HASKELL_EXE}")
    print("Presiona 'ESPACIO' para analizar. 'q' para salir.")

    status = "Listo. Presiona Espacio."
    color = (255, 255, 255)

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Mostrar el frame
        display = frame.copy()
        
        # Dibujar texto de estado
        cv2.putText(display, status, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow("Detector Haskell", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        
        if key == ord(' '):
            # 1. Indicar que estamos procesando
            print("Analizando...")
            status = "Procesando..."
            color = (0, 255, 255) # Amarillo
            cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow("Detector Haskell", display)
            cv2.waitKey(1) # Forzar refresco de ventana

            # 2. Guardar foto
            cv2.imwrite(CAPTURE_FILE, frame)

            # 3. Llamar a Haskell
            prediction = get_prediction_from_haskell(CAPTURE_FILE)
            
            print(f"Resultado Haskell: {prediction}")
            
            # 4. Actualizar estado
            status = f"Detectado: {prediction}"
            color = (0, 255, 0) # Verde

    cap.release()
    cv2.destroyAllWindows()
    # Limpieza
    if os.path.exists(CAPTURE_FILE):
        os.remove(CAPTURE_FILE)

if __name__ == "__main__":
    if check_files():
        main_loop()