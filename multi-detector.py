import pygame
import cv2
import numpy as np
import subprocess
import os
import sys
import time
import threading
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- RUTAS (Bare Metal) ---
TORCH_DIR = os.path.join(BASE_DIR, "load-torchscript") 
TORCH_EXE = os.path.join(TORCH_DIR, "load-torchscript-new") 
TORCH_MODEL = os.path.join(TORCH_DIR, "resnet_model.pt")

# --- MEDIOS ---
IMAGE_DATA_DIR = os.path.join(BASE_DIR, "Cat_Breed")
CAPTURE_FILE = os.path.join(BASE_DIR, "_temp_capture.jpg")
SOUND_ANALYSIS_FILE = os.path.join(BASE_DIR, "analysis.wav")
SOUND_SUCCESS_FILE = os.path.join(BASE_DIR, "success.wav")
SOUND_FAILURE_FILE = os.path.join(BASE_DIR, "failure.wav")
WALLPAPER_FILE = os.path.join(BASE_DIR, "background.jpeg")

# --- COLORES Y ESTILO ---
SCREEN_W, SCREEN_H = 900, 650 
COLOR_BG = (20, 20, 25)       
COLOR_TITLE = (180, 60, 255)  # MORADO NEÓN 
COLOR_DETECT = (0, 255, 0)    # VERDE PURO (Para el marco)
COLOR_TEXT = (240, 240, 240)
COLOR_GRID = (0, 255, 255)    # Cian para la guía de webcam

state = {
    "screen": "MENU",
    "image_path": None,
    "detections": [],
    "result_time": 0.0,
    "anim_images": [],
    "webcam_cap": None,
    "thread_running": False,
    "wallpaper": None,
    "snd_analysis": None,
    "snd_success": None,
    "snd_failure": None,
    "sound_channel": None
}

def change_screen(new_screen):
    state["screen"] = new_screen

# LÓGICA DE VISIÓN (HASKELL BACKEND)
def create_crops(image_path):
    img = cv2.imread(image_path)
    if img is None: return [], []
    
    h, w, _ = img.shape
    crops_meta = []
    crop_files = []

    # 5 Zonas (Sliding Window)
    zones = [
        {"id": "centro", "rect": (w//4, h//4, w//2, h//2)}, 
        {"id": "tl", "rect": (0, 0, w//2, h//2)},
        {"id": "tr", "rect": (w//2, 0, w//2, h//2)},
        {"id": "bl", "rect": (0, h//2, w//2, h//2)},
        {"id": "br", "rect": (w//2, h//2, w//2, h//2)}
    ]

    for i, z in enumerate(zones):
        x, y, cw, ch = z["rect"]
        if cw <= 0 or ch <= 0: continue
        
        crop_img = img[y:y+ch, x:x+cw]
        fname = os.path.join(BASE_DIR, f"_temp_crop_{i}.jpg")
        cv2.imwrite(fname, crop_img)
        crop_files.append(fname)
        crops_meta.append(z["rect"])

    return crop_files, crops_meta

def run_haskell_batch(image_files):
    start_t = time.time()
    
    if not os.path.exists(TORCH_EXE):
        print(f"ERROR: Binario no encontrado en {TORCH_EXE}")
        return [], 0

    cmd = [TORCH_EXE, TORCH_MODEL] + image_files
    
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=TORCH_DIR)
        
        if res.returncode != 0:
            print(f"Haskell Error: {res.stderr}")
        
        labels = res.stdout.strip().split('\n')
        end_t = time.time()
        return labels, end_t - start_t
        
    except Exception as e:
        print(f"System Error: {e}")
        return [], 0

def analysis_worker():
    state["thread_running"] = True
    if state["snd_analysis"]:
        state["sound_channel"] = state["snd_analysis"].play(loops=-1)
        
    time.sleep(1.2) 
    
    crop_files, crop_rects = create_crops(state["image_path"])
    
    if crop_files:
        labels, time_taken = run_haskell_batch(crop_files)
        state["result_time"] = time_taken
        
        final_detections = []
        ignore_list = ["background", "window_shade", "shower_curtain", "velvet", "swimming_trunks", "jean", "book_jacket"]
        
        for i, label in enumerate(labels):
            if i < len(crop_rects):
                clean = label.strip()
                if len(clean) > 2 and clean not in ignore_list:
                    final_detections.append({"label": clean, "rect": crop_rects[i]})
        
        state["detections"] = final_detections
    
    if state["sound_channel"]: 
        state["sound_channel"].stop()
        state["sound_channel"] = None

    snd = state["snd_success"] if state["detections"] else state["snd_failure"]
    if snd: snd.play()
    
    state["screen"] = "RESULT"
    state["thread_running"] = False

# INTERFAZ
def load_assets():
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
    try:
        if os.path.exists(SOUND_ANALYSIS_FILE): state["snd_analysis"] = pygame.mixer.Sound(SOUND_ANALYSIS_FILE)
        if os.path.exists(SOUND_SUCCESS_FILE): state["snd_success"] = pygame.mixer.Sound(SOUND_SUCCESS_FILE)
        if os.path.exists(SOUND_FAILURE_FILE): state["snd_failure"] = pygame.mixer.Sound(SOUND_FAILURE_FILE)
    except: pass

def get_font(name, size, bold=False):
    try: return pygame.font.SysFont(name, size, bold=bold)
    except: return pygame.font.Font(None, size)

def open_file():
    try:
        res = subprocess.run(['zenity', '--file-selection', '--file-filter=*.jpg *.png'], capture_output=True, text=True)
        return res.stdout.strip() or None
    except: return None

# --- COMPONENTES UI ---

def draw_background(screen):
    if state["wallpaper"]:
        screen.blit(state["wallpaper"], (0,0))
        overlay = pygame.Surface((SCREEN_W, SCREEN_H))
        overlay.set_alpha(230) 
        overlay.fill((15, 15, 20))
        screen.blit(overlay, (0,0))
    else:
        screen.fill(COLOR_BG)
    
    # Cuadrícula tecnológica sutil
    for x in range(0, SCREEN_W, 40):
        pygame.draw.line(screen, (30, 30, 40), (x, 0), (x, SCREEN_H), 1)
    for y in range(0, SCREEN_H, 40):
        pygame.draw.line(screen, (30, 30, 40), (0, y), (SCREEN_W, y), 1)

def draw_shadow_text(surf, text, font, center_pos, color=COLOR_TEXT):
    if not font: return
    shadow = font.render(text, True, (0,0,0))
    s_rect = shadow.get_rect(center=(center_pos[0]+2, center_pos[1]+2))
    surf.blit(shadow, s_rect)
    txt = font.render(text, True, color)
    t_rect = txt.get_rect(center=center_pos)
    surf.blit(txt, t_rect)

def draw_gummy_button(surf, rect, text, font, base_color, action=None):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()[0]
    hover = rect.collidepoint(mouse)
    
    top_color = [min(c + 40, 255) for c in base_color] if hover else base_color
    side_color = [max(c - 40, 0) for c in base_color]
    
    offset_y = 5 
    if hover and click:
        offset_y = 2 
            
    # Sombra 3D
    shadow_rect = pygame.Rect(rect.x, rect.y + offset_y, rect.width, rect.height)
    pygame.draw.rect(surf, side_color, shadow_rect, border_radius=10)
    
    # Cara superior
    top_rect = pygame.Rect(rect.x, rect.y + (5 - offset_y), rect.width, rect.height)
    pygame.draw.rect(surf, top_color, top_rect, border_radius=10)
    
    if hover:
        pygame.draw.rect(surf, (255, 255, 255), top_rect, 2, border_radius=10)
    
    txt_surf = font.render(text, True, (255, 255, 255))
    txt_rect = txt_surf.get_rect(center=top_rect.center)
    surf.blit(txt_surf, txt_rect)
    
    return hover and click

# --- PANTALLAS ---

def update_menu(screen, fonts):
    draw_background(screen)
    
    # Título 
    draw_shadow_text(screen, "HaskTorch Vision", fonts['title'], (SCREEN_W//2, 130), COLOR_TITLE)
    draw_shadow_text(screen, "Deep Learning Multi-Detector", fonts['subtitle'], (SCREEN_W//2, 180), (180, 180, 200))
    
    cx, cy = SCREEN_W // 2, SCREEN_H // 2
    
    # Botones Gummy
    if draw_gummy_button(screen, pygame.Rect(cx-150, cy-40, 300, 70), "ANALIZAR ARCHIVO", fonts['btn'], (80, 60, 150)):
        time.sleep(0.15)
        start_analysis('FILE')

    if draw_gummy_button(screen, pygame.Rect(cx-150, cy+60, 300, 70), "WEBCAM EN VIVO", fonts['btn'], (0, 150, 130)):
        time.sleep(0.15)
        start_webcam()

    draw_shadow_text(screen, "Powered by Haskell & ResNet-18", fonts['small'], (SCREEN_W//2, SCREEN_H-40), (80, 80, 80))

def update_webcam(screen, fonts):
    draw_background(screen)
    ret, frame = state["webcam_cap"].read()
    
    cam_w, cam_h = 640, 480
    cam_rect = pygame.Rect((SCREEN_W - cam_w)//2, (SCREEN_H - cam_h)//2, cam_w, cam_h)
    
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.transpose(frame, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        surf = pygame.transform.scale(surf, (cam_w, cam_h))
        screen.blit(surf, cam_rect)
        
        # --- GUÍA VISUAL (GRID) ---
        # Dibujamos las líneas que muestran las 5 zonas
        cx, cy = cam_rect.centerx, cam_rect.centery
        
        # 1. Marco externo
        pygame.draw.rect(screen, COLOR_GRID, cam_rect, 2)
        
        # 2. Cruz central (divide en 4 cuadrantes)
        pygame.draw.line(screen, COLOR_GRID, (cam_rect.left, cy), (cam_rect.right, cy), 1)
        pygame.draw.line(screen, COLOR_GRID, (cx, cam_rect.top), (cx, cam_rect.bottom), 1)
        
        # 3. Zona Central (El cuadro del medio)
        cw, ch = cam_w // 2, cam_h // 2
        center_rect = pygame.Rect(0, 0, cw, ch)
        center_rect.center = (cx, cy)
        pygame.draw.rect(screen, COLOR_TITLE, center_rect, 2) # Morado para destacar el centro
        
        # Texto de ayuda
        draw_shadow_text(screen, "ZONA CENTRAL", fonts['small'], (cx, cy - ch//2 - 15), COLOR_TITLE)

    draw_shadow_text(screen, "MODO ESCANER", fonts['subtitle'], (SCREEN_W//2, 50), COLOR_GRID)
    draw_shadow_text(screen, "[ESPACIO] Capturar   |   [ESC] Menu", fonts['medium'], (SCREEN_W//2, SCREEN_H - 60), (255, 255, 255))

    keys = pygame.key.get_pressed()
    if keys[pygame.K_SPACE]:
        ret_cap, frame_cap = state["webcam_cap"].read()
        if ret_cap:
            cv2.imwrite(CAPTURE_FILE, frame_cap)
            start_analysis('WEBCAM_CAP')
            state["webcam_cap"].release()
    if keys[pygame.K_ESCAPE]:
        state["webcam_cap"].release()
        state["screen"] = "MENU"

def update_analysis(screen, fonts):
    draw_background(screen)
    ticks = pygame.time.get_ticks()
    
    dots = "." * ((ticks // 400) % 4)
    draw_shadow_text(screen, f"PROCESANDO TENSORES{dots}", fonts['subtitle'], (SCREEN_W//2, 100), COLOR_TITLE)
    
    if state["anim_images"]:
        semilla = ticks // 120 
        random.seed(semilla) 
        path = random.choice(state["anim_images"])
        try:
            img = pygame.image.load(path)
            img = pygame.transform.scale(img, (450, 350))
            r = img.get_rect(center=(SCREEN_W//2, SCREEN_H//2))
            screen.blit(img, r)
            
            # Efecto de escaneo
            pygame.draw.rect(screen, COLOR_TITLE, r, 4)
            scan_y = r.top + (ticks % 350)
            pygame.draw.line(screen, (255, 255, 255), (r.left, scan_y), (r.right, scan_y), 3)
        except: pass

    if not state["thread_running"]:
        state["screen"] = "RESULT"

def update_result(screen, fonts):
    draw_background(screen)
    
    try:
        if os.path.exists(state["image_path"]):
            orig_img = cv2.imread(state["image_path"])
            orig_h, orig_w, _ = orig_img.shape
            
            img_disp = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            img_disp = np.transpose(img_disp, (1, 0, 2))
            surf = pygame.surfarray.make_surface(img_disp)
            
            max_h = 450
            scale = max_h / orig_h
            target_w = int(orig_w * scale)
            surf = pygame.transform.scale(surf, (target_w, max_h))
            img_rect = surf.get_rect(center=(SCREEN_W//2, SCREEN_H//2))
            
            # Dibujar imagen
            screen.blit(surf, img_rect)
            pygame.draw.rect(screen, (100, 100, 100), img_rect, 2)
            
            # --- DIBUJAR DETECCIONES (SOLO BORDES) ---
            if state["detections"]:
                for det in state["detections"]:
                    label = det["label"].upper()
                    ox, oy, ow, oh = det["rect"]
                    
                    # Escalar
                    sx = img_rect.left + int(ox * scale)
                    sy = img_rect.top + int(oy * scale)
                    sw = int(ow * scale)
                    sh = int(oh * scale)
                    
                    # 1. Dibujar Perímetro (SOLO LINEA, SIN RELLENO)
                    # width=3 hace que sea solo el marco
                    pygame.draw.rect(screen, COLOR_DETECT, (sx, sy, sw, sh), 3)
                    
                    # 2. Etiqueta encima del marco
                    lbl_surf = fonts['small'].render(label, True, (0,0,0)) # Texto negro
                    # Fondo verde para el texto
                    lbl_bg = pygame.Rect(sx, sy-24, lbl_surf.get_width()+16, 24)
                    pygame.draw.rect(screen, COLOR_DETECT, lbl_bg, border_top_left_radius=4, border_top_right_radius=4)
                    
                    # Pintar texto
                    screen.blit(lbl_surf, (sx+8, sy-20))
            else:
                draw_shadow_text(screen, "No se detectaron objetos conocidos.", fonts['medium'], (SCREEN_W//2, img_rect.bottom + 40), (255, 100, 100))

    except Exception as e: print(e)

    if draw_gummy_button(screen, pygame.Rect(SCREEN_W - 220, SCREEN_H - 80, 200, 60), "FINALIZAR", fonts['btn'], (60, 60, 70)):
        time.sleep(0.15)
        change_screen("MENU")

# --- MAIN ---

def start_webcam():
    state["screen"] = "WEBCAM"
    state["webcam_cap"] = cv2.VideoCapture(0)

def start_analysis(source):
    if source == 'FILE':
        f = open_file()
        if not f: return
        state["image_path"] = f
    elif source == 'WEBCAM_CAP':
        state["image_path"] = CAPTURE_FILE
    
    state["screen"] = "ANALYSIS"
    state["detections"] = [] 
    threading.Thread(target=analysis_worker).start()

def main():
    pygame.init()
    pygame.font.init() 
    pygame.mixer.init()
    
    state["display_surf"] = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("HaskTorch Vision - Detector")
    load_assets()
    
    fonts = {
        'title': get_font("arial", 60, bold=True),
        'subtitle': get_font("arial", 28),
        'btn': get_font("arial", 24, bold=True),
        'medium': get_font("arial", 22),
        'small': get_font("arial", 16, bold=True)
    }
    
    clock = pygame.time.Clock()
    
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
        clock.tick(30)

if __name__ == "__main__":
    main()