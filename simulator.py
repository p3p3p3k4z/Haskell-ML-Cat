import os
import cv2
import numpy as np
import subprocess
import sys
import random
import threading
import pygame
import shutil

# --- 1. CONFIGURACIÓN DEL SIMULADOR ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
HASKELL_EXECUTABLE = "./fun_cat_classifier"
IMAGE_DATA_DIR = "Cat_Breed"

# ---  Paleta de Colores ---
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_GREY = (50, 50, 50)
COLOR_DARK_GREY = (30, 30, 30) 
COLOR_GREEN = (0, 200, 0)
COLOR_GREEN_HOVER = (50, 255, 50)
COLOR_RED = (200, 0, 0)
COLOR_RED_HOVER = (255, 50, 50) 
COLOR_YELLOW = (255, 255, 0)
COLOR_BLUE = (0, 150, 255)
COLOR_BLUE_HOVER = (100, 200, 255) 


# --- 2. LÓGICA DE CLASIFICACIÓN (Sin cambios) ---

def extract_color_histogram(image_path, bins=(8, 8, 8)):
    try:
        image = cv2.imread(image_path)
        if image is None: return None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hist = cv2.calcHist([image_rgb], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist.flatten()
    except Exception:
        return None

def run_haskell_classifier(image_path):
    try:
        features = extract_color_histogram(image_path)
        if features is None:
            return "Error: No se pudo procesar la imagen"
        query_string = ",".join(map(str, features))
        process = subprocess.run(
            [HASKELL_EXECUTABLE, query_string],
            capture_output=True, text=True, check=True
        )
        prediction = process.stdout.strip().split('\n')[-1]
        return prediction
    except subprocess.CalledProcessError as e:
        return f"Error de Haskell: {e.stderr}"
    except Exception as e:
        return f"Error: {e}"

# --- 3. HILO DE ANÁLISIS  ---
g_prediction_result = None
g_is_thread_running = False

def analysis_thread_function(image_path):
    global g_prediction_result, g_is_thread_running
    g_is_thread_running = True
    g_prediction_result = run_haskell_classifier(image_path)
    g_is_thread_running = False

# --- 4. CLASE PRINCIPAL DEL SIMULADOR ---
class CatSimulator:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Analizador Funcional de Gatos (Haskell + Pygame)")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Arial", 72, bold=True)
        self.font_medium = pygame.font.SysFont("Arial", 48)
        self.font_small = pygame.font.SysFont("Arial", 24)
        
        self.zenity_available = shutil.which("zenity") is not None
        if not self.zenity_available:
            print("ADVERTENCIA: 'zenity' no se encontró. No se podrá seleccionar archivo.")

        self.load_assets()
        
        # --- Definición de botones ---
        self.btn_select = pygame.Rect(SCREEN_WIDTH//2 - 150, 400, 300, 70)
        self.btn_analyze = pygame.Rect(SCREEN_WIDTH//2 - 150, 500, 300, 70)
        self.btn_return = pygame.Rect(SCREEN_WIDTH//2 - 150, 500, 300, 70)

        self.state = 'IDLE'
        self.selected_image_path = None
        self.selected_image_surf = None
        self.animation_timer = 0
        self.result_timer = 0

    def load_assets(self):
        print("Cargando sonidos...")
        try:
            self.sound_success = pygame.mixer.Sound("success.wav")
            self.sound_failure = pygame.mixer.Sound("failure.wav")
            self.sound_analysis = pygame.mixer.Sound("analysis.wav")
        except pygame.error as e:
            print(f"¡ADVERTENCIA! No se pudieron cargar archivos de sonido: {e}", file=sys.stderr)
            self.sound_success = None
            self.sound_failure = None
            self.sound_analysis = None
            
        print("Buscando imágenes de muestra para animación...")
        self.sample_image_paths = []
        for root, _, files in os.walk(IMAGE_DATA_DIR):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.sample_image_paths.append(os.path.join(root, file))
        if not self.sample_image_paths:
            print(f"¡ADVERTENCIA! No se encontraron imágenes en {IMAGE_DATA_DIR}.", file=sys.stderr)
        else:
            print(f"Se encontraron {len(self.sample_image_paths)} imágenes de muestra.")

    def draw_button(self, rect, text, base_color, hover_color, mouse_pos):
        color = base_color
        if rect.collidepoint(mouse_pos):
            color = hover_color 
            
        pygame.draw.rect(self.screen, color, rect, border_radius=10)
        text_surf = self.font_medium.render(text, True, COLOR_BLACK)
        text_rect = text_surf.get_rect(center=rect.center)
        self.screen.blit(text_surf, text_rect)

    def draw_text(self, text, y_pos, font, color=COLOR_WHITE):
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect(center=(SCREEN_WIDTH // 2, y_pos))
        self.screen.blit(text_surf, text_rect)

    def handle_events(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.state in ['IDLE', 'LOADED']:
                if self.btn_select.collidepoint(event.pos):
                    self.select_image()
            if self.state == 'LOADED':
                if self.btn_analyze.collidepoint(event.pos):
                    self.start_analysis()
            
            if self.state == 'RESULT':
                if self.btn_return.collidepoint(event.pos):
                    self.state = 'IDLE'
                    self.selected_image_surf = None
                    self.selected_image_path = None

    def select_image(self):
        if not self.zenity_available: return
        print("Abriendo explorador de archivos (Zenity)...")
        command = [
            'zenity', '--file-selection',
            '--title=Selecciona una imagen de Gato (o No-Gato)',
            '--file-filter=*.jpg *.jpeg *.png'
        ]
        try:
            process = subprocess.run(command, capture_output=True, text=True, check=True)
            file_path = process.stdout.strip()
            if file_path:
                print(f"Imagen seleccionada: {file_path}")
                self.selected_image_path = file_path
                img = pygame.image.load(file_path)
                img_rect = img.get_rect()
                scale_w = 400 / img_rect.width
                scale_h = 300 / img_rect.height
                scale = min(scale_w, scale_h)
                new_width = int(img_rect.width * scale)
                new_height = int(img_rect.height * scale)
                self.selected_image_surf = pygame.transform.scale(img, (new_width, new_height))
                self.state = 'LOADED'
        except subprocess.CalledProcessError:
            print("Selección de archivo cancelada.")
        except Exception as e:
            print(f"Error al ejecutar Zenity: {e}")

    def start_analysis(self):
        global g_prediction_result
        if not g_is_thread_running:
            print("Iniciando análisis... (llamando a Haskell en segundo plano)")
            self.state = 'ANALYZING'
            self.animation_timer = pygame.time.get_ticks()
            g_prediction_result = None
            
            # --- ¡NUEVO! Reproducir sonido de análisis en bucle ---
            if self.sound_analysis:
                self.sound_analysis.play(loops=-1)
                
            threading.Thread(
                target=analysis_thread_function,
                args=(self.selected_image_path,)
            ).start()

    def update(self):
        if self.state == 'ANALYZING' and not g_is_thread_running:
            print(f"¡Análisis completo! Resultado: {g_prediction_result}")
            self.state = 'RESULT'
            self.result_timer = pygame.time.get_ticks()
            
            if self.sound_analysis:
                self.sound_analysis.stop()

            if g_prediction_result.lower() == 'notcat':
                if self.sound_failure: self.sound_failure.play()
            else:
                if self.sound_success: self.sound_success.play()

    def draw(self):
        self.screen.fill(COLOR_DARK_GREY)
        mouse_pos = pygame.mouse.get_pos() # Obtener pos del ratón para hover
        
        if self.state == 'IDLE':
            self.draw_text("Analizador Funcional de Gatos", 100, self.font_medium)
            if self.zenity_available:
                self.draw_text("Por favor, selecciona una imagen", 250, self.font_small)
                self.draw_button(self.btn_select, "Seleccionar", COLOR_GREEN, COLOR_GREEN_HOVER, mouse_pos)
            else:
                self.draw_text("ERROR: 'zenity' no encontrado.", 200, self.font_medium, COLOR_RED)
                self.draw_text("Instala 'zenity' para continuar", 250, self.font_small)

        elif self.state == 'LOADED':
            # Mostrar la imagen seleccionada con un borde
            if self.selected_image_surf:
                img_rect = self.selected_image_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 100))
                self.screen.blit(self.selected_image_surf, img_rect)
                pygame.draw.rect(self.screen, COLOR_WHITE, img_rect, 2) # Borde
            
            self.draw_button(self.btn_select, "Cambiar", COLOR_BLUE, COLOR_BLUE_HOVER, mouse_pos)
            self.draw_button(self.btn_analyze, "Analizar", COLOR_GREEN, COLOR_GREEN_HOVER, mouse_pos)

        elif self.state == 'ANALYZING':
            self.draw_text("Analizando...", 100, self.font_large, COLOR_YELLOW)
            
            # Animación de ráfaga 
            if self.sample_image_paths:
                if pygame.time.get_ticks() - self.animation_timer > 100:
                    self.animation_timer = pygame.time.get_ticks()
                    random_path = random.choice(self.sample_image_paths)
                    try:
                        img = pygame.image.load(random_path)
                        img = pygame.transform.scale(img, (400, 300))
                        self.random_img_surf = img
                    except Exception: pass
            if hasattr(self, 'random_img_surf'):
                img_rect = self.random_img_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50))
                self.screen.blit(self.random_img_surf, img_rect)

        elif self.state == 'RESULT':
            global g_prediction_result
            is_cat = g_prediction_result.lower() != 'notcat' and "Error" not in g_prediction_result
            
            # Efecto de parpadeo
            elapsed = pygame.time.get_ticks() - self.result_timer
            bg_color = COLOR_DARK_GREY
            if (elapsed // 250) % 2 == 0 and elapsed < 2000: # Parpadea por 2 segundos
                bg_color = COLOR_RED if not is_cat else COLOR_YELLOW
            self.screen.fill(bg_color)

            # Mostrar la imagen del usuario
            if self.selected_image_surf:
                img_rect = self.selected_image_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 100))
                self.screen.blit(self.selected_image_surf, img_rect)
                pygame.draw.rect(self.screen, (COLOR_RED if not is_cat else COLOR_GREEN), img_rect, 4)

            # Mostrar el texto del resultado
            if is_cat:
                self.draw_text("¡ES UN GATO!", 350, self.font_large, COLOR_GREEN)
                self.draw_text(f"Predicción: {g_prediction_result}", 420, self.font_medium, COLOR_WHITE)
            else:
                self.draw_text("¡NO ES UN GATO!", 350, self.font_large, COLOR_RED)
                self.draw_text(f"Resultado: {g_prediction_result}", 420, self.font_medium, COLOR_WHITE)

            # Botón de Volver a Analizar
            self.draw_button(self.btn_return, "Regresar", COLOR_BLUE, COLOR_BLUE_HOVER, mouse_pos)

        pygame.display.flip()

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                self.handle_events(event)
            self.update()
            self.draw()
            self.clock.tick(30)
        pygame.quit()
        sys.exit()

# --- 5. PUNTO DE ENTRADA  ---
if __name__ == "__main__":
    if not os.path.exists(HASKELL_EXECUTABLE):
        print(f"¡ERROR FATAL! No se encuentra el ejecutable '{HASKELL_EXECUTABLE}'.", file=sys.stderr)
        print("Asegúrate de haber corrido 'cabal install --installdir=. fun_cat_classifier'", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(IMAGE_DATA_DIR):
        print(f"¡ERROR FATAL! No se encuentra el directorio de datos '{IMAGE_DATA_DIR}'.", file=sys.stderr)
        sys.exit(1)
    
    sim = CatSimulator()
    sim.run()