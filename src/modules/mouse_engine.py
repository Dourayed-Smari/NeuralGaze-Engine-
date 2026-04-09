import pyautogui
import time
import numpy as np

class KinematicMouse:
    def __init__(self):
        self.screen_w, self.screen_h = pyautogui.size()
        pyautogui.FAILSAFE, pyautogui.PAUSE = False, 0
        self.last_x, self.last_y = self.screen_w // 2, self.screen_h // 2
        self.smoothing, self.dynamic_mode = 0.2, False
        self.last_move_time = time.time()
        self.min_move_interval = 0.001
        self.dead_zone_x, self.dead_zone_y = 0.004, 0.004

    def set_dynamic_mode(self, enabled):
        self.dynamic_mode = enabled
        self.smoothing = 0.25 if enabled else 0.2
        self.dead_zone_x = self.dead_zone_y = 0.002 if enabled else 0.004

    def move_mouse(self, gaze_coords):
        # CORRECTION : is not None pour éviter l'erreur NumPy
        if gaze_coords is None: return
        
        now = time.time()
        if now - self.last_move_time < self.min_move_interval: return
        gx, gy = gaze_coords
        if abs(gx - 0.5) < self.dead_zone_x and abs(gy - 0.5) < self.dead_zone_y: return
        tx, ty = (1 - gx) * self.screen_w, gy * self.screen_h
        sf = 0.35 if self.dynamic_mode else 0.3
        sx, sy = self.last_x + (tx - self.last_x) * sf, self.last_y + (ty - self.last_y) * sf
        try:
            pyautogui.moveTo(int(sx), int(sy))
            self.last_x, self.last_y, self.last_move_time = sx, sy, now
        except: pass

    def click(self, button="left"):
        try: pyautogui.click(button=button)
        except: pass
