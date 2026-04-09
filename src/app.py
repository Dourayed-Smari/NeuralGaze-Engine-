from modules.vision_engine import NeuralVisionEngine
from modules.mouse_engine import KinematicMouse
import time

def main():
    print("Démarrage du Neural Gaze Engine...")
    vision_engine = NeuralVisionEngine()
    mouse_engine = KinematicMouse()
    
    frame_count = 0
    start_time = time.time()
    last_fps_time = time.time()
    
    print("Moteur de vision initialisé avec succès !")
    print("Contrôles Système :")
    print("  [C] - Initialiser la Calibration (au lieu de ESPACE)")
    print("  [M] - Changer de mode de suivi Dynamique/Statique (au lieu de A)")
    print("  [L] - Afficher/Masquer le maillage facial IA")
    print("  [Q] - Quitter le système")
    
    try:
        while True:
            gaze_coords, click_type = vision_engine.get_gaze()
            
            if gaze_coords and vision_engine.calibrated:
                mouse_engine.set_dynamic_mode(vision_engine.dynamic_mode)
                mouse_engine.move_mouse(gaze_coords)
                
                if click_type == "left":
                    print("--> Action: Clic GAUCHE détecté !")
                    mouse_engine.click("left")
                elif click_type == "right":
                    print("--> Action: Clic DROIT détecté !")
                    mouse_engine.click("right")
            
            frame_count += 1
            current_time = time.time()
            
            if current_time - last_fps_time > 3.0:
                elapsed = current_time - start_time
                fps = frame_count / elapsed
                mode = "DYNAMIC KINEMATICS" if vision_engine.dynamic_mode else "STATIC TRACKING"
                print(f"Fluidité IO: {fps:.1f} FPS | Mode: {mode} | Système Actif")
                last_fps_time = current_time
            
            if vision_engine.should_quit():
                break
                
    except KeyboardInterrupt:
        print("\nArrêt manuel du système...")
    except Exception as e:
        print(f"Erreur critique: {e}")
    finally:
        vision_engine.cleanup()
        print("Moteur Neural Gaze fermé en sécurité.")

if __name__ == "__main__":
    main()