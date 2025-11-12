import os
import cv2  # Esta es la biblioteca OpenCV
import numpy as np
import csv
import sys

def extract_color_histogram(image_path, bins=(8, 8, 8)):
    """
    Carga una imagen y calcula su histograma de color 3D (RGB).
    Luego, normaliza el histograma para que los valores estén entre 0 y 1.
    """
    try:
        # Cargar la imagen
        image = cv2.imread(image_path)
        if image is None:
            print(f"Advertencia: No se pudo cargar la imagen {image_path}", file=sys.stderr)
            return None

        # Convertir la imagen a espacio de color RGB (OpenCV carga en BGR por defecto)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Calcular el histograma (8x8x8 = 512 características)
        hist = cv2.calcHist([image_rgb], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])

        # Normalizar el histograma (MIN-MAX para que esté entre 0 y 1)
        # Esto es crucial para la Distancia Euclidiana
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        # Aplanar el histograma 3D en un vector 1D (de 512 elementos)
        return hist.flatten()

    except Exception as e:
        print(f"Error procesando {image_path}: {e}", file=sys.stderr)
        return None

def main():
    """
    Función principal para recorrer las carpetas de datos,
    extraer características y guardar en un CSV.
    """
    
    # --- RUTA RELATIVA ---
    # Asume que el script se ejecuta desde /Haskell-ML-Cat
    # y los datos están en /Haskell-ML-Cat/Cat_Breed
    base_dir = "Cat_Breed" 
    
    # El archivo CSV se guardará en la carpeta actual (Haskell-ML-Cat)
    output_csv = "training_data.csv"
    
    histogram_bins = (8, 8, 8) # 512 características en total

    print(f"Iniciando extracción de características desde: {base_dir}")
    print(f"Guardando datos en: {output_csv}")

    # Abrir el archivo CSV para escribir
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Iterar sobre cada carpeta de raza (ej. 'Abyssinian', 'NotCat', ...)
        for label in os.listdir(base_dir):
            label_path = os.path.join(base_dir, label)
            
            # Asegurarse de que es un directorio (¡Error corregido aquí!)
            if not os.path.isdir(label_path):
                continue
            
            print(f"\nProcesando clase: {label}")
            
            count = 0
            # Iterar sobre cada imagen en la carpeta de la raza
            for image_name in os.listdir(label_path):
                image_path = os.path.join(label_path, image_name)
                
                # Extraer el vector de características (histograma)
                features = extract_color_histogram(image_path, histogram_bins)
                
                if features is not None:
                    # Escribir la fila en el CSV: "Etiqueta,f1,f2,...,f512"
                    writer.writerow([label] + features.tolist())
                    count += 1
                    
                    if count % 50 == 0:
                        print(f"  ... {count} imágenes procesadas de {label}")

            print(f"  ¡Listo! Se procesaron {count} imágenes para {label}.")

    print(f"\n¡Extracción completada! (Fase 1)")
    print(f"Datos guardados en {output_csv}")

# Punto de entrada estándar de Python
if __name__ == "__main__":
    main()