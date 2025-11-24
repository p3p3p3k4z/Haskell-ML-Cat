import pygame
import cv2
import numpy as np
import subprocess
import os
import sys
import time
import threading
import random

# ==========================================
# 1. CONFIGURACIÓN GLOBAL
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Rutas de Ejecución ---
KNN_EXE = os.path.join(BASE_DIR, "fun_cat_classifier")

TORCH_DIR = os.path.join(BASE_DIR, "load-torchscript")
TORCH_EXE = os.path.join(TORCH_DIR, "load-torchscript")
TORCH_MODEL = os.path.join(TORCH_DIR, "resnet_model.pt")

# --- Recursos ---
IMAGE_DATA_DIR = os.path.join(BASE_DIR, "Cat_Breed")
CAPTURE_FILE = os.path.join(BASE_DIR, "_temp_capture.jpg")
SOUND_ANALYSIS_FILE = os.path.join(BASE_DIR, "analysis.wav")
SOUND_SUCCESS_FILE = os.path.join(BASE_DIR, "success.wav")
SOUND_FAILURE_FILE = os.path.join(BASE_DIR, "failure.wav")
WALLPAPER_FILE = os.path.join(BASE_DIR, "background.jpeg")

# --- UI Config ---
SCREEN_W, SCREEN_H = 800, 600
COLOR_BG = (30, 30, 35)
COLOR_TEXT = (240, 240, 240)

# Colores
COLOR_KNN = (50, 255, 100)     
COLOR_KNN_DIM = (20, 100, 40)  
COLOR_TORCH = (255, 120, 0)    
COLOR_TORCH_DIM = (100, 50, 0) 
BTN_COLOR = (70, 70, 70)       
BTN_HOVER = (100, 100, 100)    

# ==========================================
# 2. ESTADO GLOBAL
# ==========================================

state = {
    "screen": "MENU",
    "model": None,
    "source": None,
    "image_path": None,
    "result_main": "",
    "result_sub": "",
    "result_color": COLOR_TEXT,
    "result_time": 0.0,
    "anim_images": [],
    "webcam_cap": None,
    "thread_running": False,
    "wallpaper": None,
    
    # --- SONIDOS (Cargados en memoria) ---
    "snd_analysis": None,
    "snd_success": None,
    "snd_failure": None,
    "sound_channel": None # Para detener el loop
}

def change_screen(new_screen):
    state["screen"] = new_screen

# ==========================================
# 3. LÓGICA DE BACKEND
# ==========================================

def extract_color_histogram(image_path, bins=(8, 8, 8)):
    """
    Lógica exacta de 'simulator.py'.
    Crucial para que el KNN funcione correctamente.
    """
    try:
        image = cv2.imread(image_path)
        if image is None: return None
        # Convertir a RGB (OpenCV usa BGR por defecto)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Calcular histograma 3D
        hist = cv2.calcHist([image_rgb], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
        # Normalizar (MinMax 0-1)
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist.flatten()
    except: return None

def run_knn(image_path):
    start_t = time.time()
    
    # 1. Extracción (Python)
    features = extract_color_histogram(image_path)
    if features is None: return "Error", "Imagen inválida", (255, 0, 0), 0
    
    try:
        # 2. Clasificación (Haskell)
        query_str = ",".join(map(str, features))
        res = subprocess.run([KNN_EXE, query_str], capture_output=True, text=True)
        end_t = time.time()
        
        label = res.stdout.strip().split('\n')[-1]
        
        # Lógica de visualización de simulator.py
        if "NotCat" in label or "Error" in label:
            return "NO ES UN GATO", f"Predicción: {label}", (255, 80, 80), end_t - start_t
        else:
            return "¡ES UN GATO!", f"Raza: {label}", (80, 255, 80), end_t - start_t
            
    except Exception as e:
        return "Error", str(e), (255, 0, 0), 0

def run_torch(image_path):
    start_t = time.time()
    # Comando robusto con comillas
    cmd = f'"{TORCH_EXE}" "{TORCH_MODEL}" "{image_path}"'
    try:
        # shell=True para librerías C++
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=BASE_DIR)
        end_t = time.time()
        output = res.stdout.strip()
        label = "Desconocido"
        
        # Parseo robusto (busca la línea que contiene --labels-- sin importar comillas)
        lines = output.splitlines()
        idx = -1
        for i, l in enumerate(lines):
            if "--labels--" in l:
                idx = i
                break
        
        if idx != -1 and idx + 1 < len(lines):
            # Limpieza total de caracteres extraños
            label = lines[idx+1].replace('[','').replace(']','').replace('"','').replace("'", "").split(',')[0].strip()
        
        return "OBJETO DETECTADO", f"Clase: {label}", (255, 200, 50), end_t - start_t
            
    except Exception as e:
        return "Error Torch", str(e), (255, 0, 0), 0

def analysis_worker():
    state["thread_running"] = True
    
    # Reproducir sonido de análisis en bucle (si está cargado)
    if state["snd_analysis"]:
        state["sound_channel"] = state["snd_analysis"].play(loops=-1)
        
    time.sleep(1.5) # Pausa para animación y sonido
    
    if state["model"] == 'KNN':
        m, s, c, t = run_knn(state["image_path"])
    else:
        m, s, c, t = run_torch(state["image_path"])
        
    state["result_main"], state["result_sub"], state["result_color"], state["result_time"] = m, s, c, t
    
    # Detener sonido análisis
    if state["sound_channel"]: 
        state["sound_channel"].stop()
        state["sound_channel"] = None

    # Sonido final
    is_fail = "NO" in m or "Error" in m
    snd = state["snd_failure"] if is_fail else state["snd_success"]
    if snd: snd.play()
    
    state["screen"] = "RESULT"
    state["thread_running"] = False

# ==========================================
# 4. INTERFAZ (PYGAME)
# ==========================================

def load_assets():
    # Imágenes
    if os.path.exists(WALLPAPER_FILE):
        try:
            img = pygame.image.load(WALLPAPER_FILE)
            state["wallpaper"] = pygame.transform.scale(img, (SCREEN_W, SCREEN_H))
        except: pass
    
    if os.path.exists(IMAGE_DATA_DIR):
        for root, _, files in os.walk(IMAGE_DATA_DIR):
            for f in files:
                if f.lower().endswith(('jpg', 'png')):
                    state["anim_images"].append(os.path.join(root, f))
                    
    # Sonidos (Cargarlos aquí evita que el GC los elimine)
    if os.path.exists(SOUND_ANALYSIS_FILE): state["snd_analysis"] = pygame.mixer.Sound(SOUND_ANALYSIS_FILE)
    if os.path.exists(SOUND_SUCCESS_FILE): state["snd_success"] = pygame.mixer.Sound(SOUND_SUCCESS_FILE)
    if os.path.exists(SOUND_FAILURE_FILE): state["snd_failure"] = pygame.mixer.Sound(SOUND_FAILURE_FILE)

def open_file():
    try:
        res = subprocess.run(['zenity', '--file-selection'], capture_output=True, text=True)
        return res.stdout.strip() or None
    except: return None

# --- DIBUJO ---

def draw_background(screen):
    if state["wallpaper"]:
        screen.blit(state["wallpaper"], (0,0))
        overlay = pygame.Surface((SCREEN_W, SCREEN_H))
        overlay.set_alpha(150)
        overlay.fill((0,0,0))
        screen.blit(overlay, (0,0))
    else:
        screen.fill(COLOR_BG)

def draw_text(surf, text, x, y, font, color=COLOR_TEXT, shadow=True):
    if shadow:
        sh = font.render(text, True, (0,0,0))
        r_sh = sh.get_rect(center=(x+2, y+2))
        surf.blit(sh, r_sh)
    obj = font.render(text, True, color)
    rect = obj.get_rect(center=(x, y))
    surf.blit(obj, rect)

def draw_btn(surf, rect, text, font, base_color, action=None):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()[0]
    hover = rect.collidepoint(mouse)
    
    color = [min(c + 60, 255) for c in base_color] if hover else base_color
    border_col = (255, 255, 255) if hover else (100, 100, 100)
    
    pygame.draw.rect(surf, color, rect, border_radius=12)
    pygame.draw.rect(surf, border_col, rect, 3 if hover else 1, border_radius=12)
    
    txt = font.render(text, True, (255,255,255))
    txt_r = txt.get_rect(center=rect.center)
    surf.blit(txt, txt_r)
    
    if hover and click and action:
        time.sleep(0.2)
        action()

# --- PANTALLAS ---

def update_menu(screen, fonts):
    draw_background(screen)
    draw_text(screen, "Clasificador gato VS Detector de cosas", SCREEN_W//2, 60, fonts['medium'], (255, 255, 255))
    draw_text(screen, "Selecciona tu Modelo", SCREEN_W//2, 110, fonts['medium'], (200, 200, 200))

    pygame.draw.line(screen, (100,100,100), (SCREEN_W//2, 160), (SCREEN_W//2, 550), 2)

    # KNN
    cx_knn = SCREEN_W // 4
    draw_text(screen, "KNN CLASICO", cx_knn, 180, fonts['medium'], COLOR_KNN)
    draw_text(screen, "Haskell Puro (Listas)", cx_knn, 210, fonts['small'], (180, 255, 180))
    
    draw_btn(screen, pygame.Rect(cx_knn-100, 260, 200, 50), "Abrir Archivo", fonts['small'], COLOR_KNN_DIM,
             lambda: start_analysis('KNN', 'FILE'))
    draw_btn(screen, pygame.Rect(cx_knn-100, 330, 200, 50), "Usar Webcam", fonts['small'], COLOR_KNN_DIM,
             lambda: start_webcam('KNN'))

    # Torch
    cx_torch = 3 * SCREEN_W // 4
    draw_text(screen, "DEEP LEARNING", cx_torch, 180, fonts['medium'], COLOR_TORCH)
    draw_text(screen, "Hasktorch (Tensores)", cx_torch, 210, fonts['small'], (255, 200, 150))

    draw_btn(screen, pygame.Rect(cx_torch-100, 260, 200, 50), "Abrir Archivo", fonts['small'], COLOR_TORCH_DIM,
             lambda: start_analysis('TORCH', 'FILE'))
    draw_btn(screen, pygame.Rect(cx_torch-100, 330, 200, 50), "Usar Webcam", fonts['small'], COLOR_TORCH_DIM,
             lambda: start_webcam('TORCH'))

def update_webcam(screen, fonts):
    draw_background(screen)
    ret, frame = state["webcam_cap"].read()
    
    cam_rect = pygame.Rect(0, 0, 500, 375)
    cam_rect.center = (SCREEN_W//2, SCREEN_H//2 - 20)
    
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.transpose(frame, (1, 0, 2)) # Horizontal
        surf = pygame.surfarray.make_surface(frame)
        surf = pygame.transform.scale(surf, (cam_rect.width, cam_rect.height))
        screen.blit(surf, cam_rect)
        col = COLOR_KNN if state["model"] == 'KNN' else COLOR_TORCH
        pygame.draw.rect(screen, col, cam_rect, 4)

    draw_text(screen, f"MODO WEBCAM: {state['model']}", SCREEN_W//2, 40, fonts['medium'])
    info_y = cam_rect.bottom + 30
    draw_text(screen, "[ESPACIO] CAPTURAR", SCREEN_W//2 - 150, info_y, fonts['medium'], (100, 255, 100))
    draw_text(screen, "[ESC] SALIR", SCREEN_W//2 + 150, info_y, fonts['medium'], (255, 100, 100))

    keys = pygame.key.get_pressed()
    if keys[pygame.K_SPACE]:
        # Capturar frame original BGR
        ret_cap, frame_cap = state["webcam_cap"].read()
        if ret_cap:
            cv2.imwrite(CAPTURE_FILE, frame_cap)
            start_analysis(state["model"], 'WEBCAM_CAP')
            state["webcam_cap"].release()
            
    if keys[pygame.K_ESCAPE]:
        state["webcam_cap"].release()
        state["screen"] = "MENU"

def update_analysis(screen, fonts):
    draw_background(screen)
    draw_text(screen, "PROCESANDO...", SCREEN_W//2, SCREEN_H//2 - 200, fonts['big'], (255, 255, 50))
    
    if state["anim_images"]:
        semilla_tiempo = pygame.time.get_ticks() // 200 #Tiempo de cada img
        random.seed(semilla_tiempo) 

        path = random.choice(state["anim_images"])#Selecciona aleatoriamente la img
        try:
            img = pygame.image.load(path)
            img = pygame.transform.scale(img, (300, 225))
            r = img.get_rect(center=(SCREEN_W//2, SCREEN_H//2))
            screen.blit(img, r)
            pygame.draw.rect(screen, (255,255,255), r, 4)
        except: pass

    if not state["thread_running"]:
        state["screen"] = "RESULT"

def update_result(screen, fonts):
    draw_background(screen)
    
    # Imagen (Arriba)
    try:
        if os.path.exists(state["image_path"]):
            img = pygame.image.load(state["image_path"])
            scale = min(400/img.get_width(), 300/img.get_height())
            img = pygame.transform.scale(img, (int(img.get_width()*scale), int(img.get_height()*scale)))
            r = img.get_rect(center=(SCREEN_W//2, 200)) 
            screen.blit(img, r)
            col = COLOR_KNN if state["model"] == 'KNN' else COLOR_TORCH
            pygame.draw.rect(screen, col, r, 4)
    except: pass

    # Resultados (Subidos)
    base_y = 360 
    draw_text(screen, "RESULTADO:", SCREEN_W//2, base_y, fonts['medium'], (200,200,200))
    draw_text(screen, state["result_main"], SCREEN_W//2, base_y + 45, fonts['big'], state["result_color"])
    draw_text(screen, state["result_sub"], SCREEN_W//2, base_y + 90, fonts['medium'], (255, 255, 255))
    draw_text(screen, f"Tiempo: {state['result_time']:.4f} seg", SCREEN_W//2, base_y + 130, fonts['small'], (180,180,180))

    draw_btn(screen, pygame.Rect(SCREEN_W//2 - 100, SCREEN_H - 80, 200, 50), "VOLVER", fonts['medium'], BTN_COLOR,
             lambda: change_screen("MENU"))

# --- ACCIONES ---

def start_webcam(model):
    state["model"] = model
    state["screen"] = "WEBCAM"
    state["webcam_cap"] = cv2.VideoCapture(0)

def start_analysis(model, source):
    state["model"] = model
    if source == 'FILE':
        f = open_file()
        if not f: return
        state["image_path"] = f
    elif source == 'WEBCAM_CAP':
        state["image_path"] = CAPTURE_FILE
    
    state["screen"] = "ANALYSIS"
    threading.Thread(target=analysis_worker).start()

def main():
    pygame.init()
    pygame.font.init() 
    
    state["display_surf"] = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Haskell ML Cat")
    load_assets()
    
    fonts = {
        'big': pygame.font.SysFont("Arial", 48, bold=True),
        'medium': pygame.font.SysFont("Arial", 32, bold=True),
        'small': pygame.font.SysFont("Arial", 18)
    }
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if state["webcam_cap"]: state["webcam_cap"].release()
                pygame.quit(); sys.exit()
        
        screen = state["display_surf"]
        if state["screen"] == "MENU": update_menu(screen, fonts)
        elif state["screen"] == "WEBCAM": update_webcam(screen, fonts)
        elif state["screen"] == "ANALYSIS": update_analysis(screen, fonts)
        elif state["screen"] == "RESULT": update_result(screen, fonts)
        
        pygame.display.flip()
        pygame.time.Clock().tick(30)

if __name__ == "__main__":
    main()