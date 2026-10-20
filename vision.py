import cv2
import numpy as np
import onnxruntime as ort
import time
from picamera2 import Picamera2
from gpiozero import Buzzer
import smbus

# --- Bibliothèques pour l'écran OLED ---
try:
    import board
    import busio
    import adafruit_ssd1306
    from PIL import Image, ImageDraw
    OLED_DISPO = True
except ImportError as e:
    print(f"❌ ERREUR : Bibliothèque OLED manquante ({e}). Tapez: pip install adafruit-circuitpython-ssd1306")
    OLED_DISPO = False

# ==========================================
# 1. INITIALISATION MATÉRIEL
# ==========================================

try:
    buzzer = Buzzer(18)
except:
    buzzer = None

# -- Écran OLED (I2C 0x3C) --
oled = None
if OLED_DISPO:
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        oled = adafruit_ssd1306.SSD1306_I2C(128, 64, i2c, addr=0x3C)
        oled.fill(0)
        oled.show()
        print("✅ Écran OLED initialisé avec succès.")
    except Exception as e:
        print(f"❌ ERREUR de connexion à l'écran OLED : {e}")

def mettre_a_jour_ecran(clignotant, freinage):
    if not oled: return
    image = Image.new("1", (oled.width, oled.height))
    draw = ImageDraw.Draw(image)
    
    if freinage:
        draw.rectangle((40, 10, 50, 40), fill=255); draw.rectangle((40, 45, 50, 55), fill=255)
        draw.rectangle((60, 10, 70, 40), fill=255); draw.rectangle((60, 45, 70, 55), fill=255)
        draw.rectangle((80, 10, 90, 40), fill=255); draw.rectangle((80, 45, 90, 55), fill=255)
    elif clignotant == 'gauche':
        draw.polygon([(60, 10), (20, 32), (60, 54)], fill=255); draw.rectangle((60, 22, 100, 42), fill=255)
    elif clignotant == 'droite':
        draw.polygon([(68, 10), (108, 32), (68, 54)], fill=255); draw.rectangle((28, 22, 68, 42), fill=255)
    
    oled.image(image)
    oled.show()

# -- Capteur MPU-6050 (I2C 0x68) --
try:
    bus = smbus.SMBus(1)
    MPU_ADDR = 0x68
    bus.write_byte_data(MPU_ADDR, 0x6B, 0)
    print("✅ Capteur MPU-6050 initialisé.")
    MPU_DISPO = True
except Exception as e:
    print(f"❌ ERREUR de connexion au capteur MPU-6050 : {e}")
    MPU_DISPO = False

def lire_mpu6050():
    if not MPU_DISPO: return None, False
    try:
        accel_x = bus.read_byte_data(MPU_ADDR, 0x3B) << 8 | bus.read_byte_data(MPU_ADDR, 0x3C)
        accel_y = bus.read_byte_data(MPU_ADDR, 0x3D) << 8 | bus.read_byte_data(MPU_ADDR, 0x3E)
        if accel_x > 32768: accel_x -= 65536
        if accel_y > 32768: accel_y -= 65536
        
        frein = accel_x < -12000
        cligno = 'droite' if accel_y > 10000 else 'gauche' if accel_y < -10000 else None
        return cligno, frein
    except:
        return None, False

# ==========================================
# 2. INITIALISATION IA ET CAMERA
# ==========================================
print("Initialisation de la PiCamera2...")
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

print("Chargement du modèle YOLOv8 ONNX...")
session = ort.InferenceSession("modele.onnx", providers=['CPUExecutionProvider'])
model_inputs = session.get_inputs()
target_w, target_h = 640, 640
SEUIL_DANGER_SURFACE = 40000

print("--- Système Prêt ! Appuyez sur 'q' pour quitter. ---")

try:
    while True:
        clignotant, freinage = lire_mpu6050()
        mettre_a_jour_ecran(clignotant, freinage)

        frame_raw = picam2.capture_array()
        frame_rgb = cv2.cvtColor(frame_raw, cv2.COLOR_RGBA2RGB) if frame_raw.shape[2] == 4 else frame_raw.copy()
        
        # Inférence IA
        img_input = np.expand_dims(np.transpose(cv2.resize(frame_rgb, (target_w, target_h)).astype(np.float32) / 255.0, (2, 0, 1)), axis=0)
        predictions = np.squeeze(session.run(None, {model_inputs[0].name: img_input})[0]).T

        danger = False
        for pred in predictions:
            scores = pred[4:]
            class_id = np.argmax(scores)
            if scores[class_id] > 0.4:
                danger = True
                # Dessiner le rectangle de danger sur la vidéo
                cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]
                x1 = int((cx - w/2) * (frame_rgb.shape[1] / target_w))
                y1 = int((cy - h/2) * (frame_rgb.shape[0] / target_h))
                x2 = int((cx + w/2) * (frame_rgb.shape[1] / target_w))
                y2 = int((cy + h/2) * (frame_rgb.shape[0] / target_h))
                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)

        if danger and buzzer: buzzer.on()
        elif buzzer: buzzer.off()

        # Restauration de l'affichage vidéo !
        cv2.imshow("Vision IA Casque", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nArrêt par l'utilisateur.")

finally:
    if buzzer: buzzer.off()
    if oled: oled.fill(0); oled.show()
    picam2.stop()
    cv2.destroyAllWindows()
