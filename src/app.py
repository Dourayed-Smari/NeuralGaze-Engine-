from modules.vision_engine import NeuralVisionEngine
from modules.mouse_engine import KinematicMouse
import time

def main():
    print("Starting NeuralGaze Engine...")
    vision_engine = NeuralVisionEngine()
    mouse_engine = KinematicMouse()
    
    frame_count = 0
    start_time = time.time()
    last_fps_time = time.time()
    
    print("NeuralGaze AI Vision initialized successfully!")
    print("System Controls:")
    print("  [C] - Initialize Calibration")
    print("  [M] - Toggle Tracking Mode (Dynamic/Static)")
    print("  [L] - Toggle AI Facial Mesh HUD")
    print("  [Q] - Quit System")
    
    try:
        while True:
            gaze_coords, click_type = vision_engine.get_gaze()
            
            if gaze_coords is not None and vision_engine.calibrated:
                mouse_engine.set_dynamic_mode(vision_engine.dynamic_mode)
                mouse_engine.move_mouse(gaze_coords)
                
                if click_type == "left":
                    print("--> Action: LEFT Click detected!")
                    mouse_engine.click("left")
                elif click_type == "right":
                    print("--> Action: RIGHT Click detected!")
                    mouse_engine.click("right")
            
            frame_count += 1
            current_time = time.time()
            
            if current_time - last_fps_time > 3.0:
                elapsed = current_time - start_time
                fps = frame_count / elapsed
                mode = "DYNAMIC KINEMATICS" if vision_engine.dynamic_mode else "STATIC TRACKING"
                print(f"I/O Fluidity: {fps:.1f} FPS | Mode: {mode} | System Active")
                last_fps_time = current_time
            
            if vision_engine.should_quit():
                break
                
    except KeyboardInterrupt:
        print("\nManual system shutdown...")
    except Exception as e:
        print(f"Critical Error: {e}")
    finally:
        vision_engine.cleanup()
        print("NeuralGaze Engine safely closed.")

if __name__ == "__main__":
    main()
