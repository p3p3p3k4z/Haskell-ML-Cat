import os
import cv2
import numpy as np
import csv
import sys
import random

def extract_color_histogram(image_path, bins=(8, 8, 8)):
    """
    Carga una imagen y calcula su histograma de color 3D (RGB) normalizado.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Advertencia: No se pudo cargar la imagen {image_path}", file=sys.stderr)
            return None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hist = cv2.calcHist([image_rgb], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist.flatten()
    except Exception as e:
        print(f"Error procesando {image_path}: {e}", file=sys.stderr)
        return None

def main():
    """
    Función principal para recorrer las carpetas de datos,
    extraer características y guardar en un CSV.
    """
    
    base_dir = "Cat_Breed" 
    output_csv = "training_data.csv"
    histogram_bins = (8, 8, 8)
    
    MAX_IMAGES_PER_CAT_BREED = 500
    MAX_IMAGES_PER_NOTCAT_SUBCLASS = 50 
    
    print(f"Iniciando extracción de características desde: {base_dir}")
    print(f"Guardando datos en: {output_csv}")
    print(f"¡Modo de Muestreo Estratificado!")
    print(f"  - Razas de Gato: Máx {MAX_IMAGES_PER_CAT_BREED} por raza.")
    print(f"  - Clase 'NotCat': Máx {MAX_IMAGES_PER_NOTCAT_SUBCLASS} por sub-clase (ej. perros, pájaros...).")

    total_rows = 0

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Iterar sobre las carpetas principales (Abyssinian, ..., NotCat)
        for label in os.listdir(base_dir):
            label_path = os.path.join(base_dir, label)
            
            if not os.path.isdir(label_path):
                continue
            
            # --- LÓGICA DE MUESTREO DIVIDIDA ---
            
            if label == "NotCat":
                # --- A. Lógica para NOTCAT (Muestreo Estratificado) ---
                print(f"\nProcesando 'NotCat' (Muestreo Estratificado)...")
                count_notcat_total = 0
                
                # Asumimos que NotCat tiene subcarpetas (ej. Dogs, Birds, Cars)
                for subclass_name in os.listdir(label_path):
                    subclass_path = os.path.join(label_path, subclass_name)
                    
                    if not os.path.isdir(subclass_path):
                        # Ignorar archivos sueltos en NotCat/ (ej. .DS_Store)
                        continue 
                    
                    print(f"  Procesando sub-clase: {subclass_name}")
                    
                    # 1. Obtener todas las imágenes de la sub-carpeta
                    all_images = os.listdir(subclass_path)
                    
                    # 2. Mezclarlas aleatoriamente
                    random.shuffle(all_images)
                    
                    # 3. Seleccionar las primeras 30
                    selected_images = all_images[:MAX_IMAGES_PER_NOTCAT_SUBCLASS]
                    
                    print(f"    ... {len(selected_images)} imágenes seleccionadas (de {len(all_images)})")
                    
                    for image_name in selected_images:
                        image_path = os.path.join(subclass_path, image_name)
                        features = extract_color_histogram(image_path, histogram_bins)
                        if features is not None:
                            # ¡IMPORTANTE! La etiqueta sigue siendo "NotCat"
                            writer.writerow(["NotCat"] + features.tolist())
                            count_notcat_total += 1
                
                print(f"  ¡Listo! Se procesaron {count_notcat_total} imágenes para 'NotCat' en total.")
                total_rows += count_notcat_total

            else:
                # --- B. Lógica para RAZAS DE GATO (Muestreo Aleatorio Simple) ---
                print(f"\nProcesando raza: {label}")
                
                all_images = os.listdir(label_path)
                random.shuffle(all_images)
                selected_images = all_images[:MAX_IMAGES_PER_CAT_BREED]
                
                print(f"  ... {len(selected_images)} imágenes seleccionadas (de {len(all_images)})")
                count_breed = 0
                
                for image_name in selected_images:
                    image_path = os.path.join(label_path, image_name)
                    features = extract_color_histogram(image_path, histogram_bins)
                    if features is not None:
                        writer.writerow([label] + features.tolist())
                        count_breed += 1
                        
                print(f"  ¡Listo! Se procesaron {count_breed} imágenes para {label}.")
                total_rows += count_breed

    print(f"\n¡Extracción completada!")
    print(f"Datos guardados en {output_csv} con un total de {total_rows} filas.")

if __name__ == "__main__":
    main()