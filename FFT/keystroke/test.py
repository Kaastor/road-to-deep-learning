import time
from pynput import keyboard

class KeyStrokeTimer:
    def __init__(self):
        self.last_time = None
        self.timings = []
        self.typed_text = ''
        self.last_key = None
        self.last_key_time = None

    def save_timings(self):
        with open('keystroke_timings_m3.txt', 'w') as f:
            for timing in self.timings:
                f.write(f"{timing:.4f}\n")
        print("Timings saved to 'keystroke_timings.txt'")

    def on_key_press(self, key):
        current_time = time.time()

        # Check for double Enter press
        if key == keyboard.Key.enter and self.last_key == key:
            time_since_last_key = current_time - self.last_key_time if self.last_key_time else float('inf')
            if time_since_last_key < 0.5:  # 0.5 seconds threshold for double press
                self.save_timings()
                return False  # Stop the listener

        # Convert key to its string representation
        try:
            key_char = key.char
        except AttributeError:
            key_char = str(key)

        # Handle special keys
        if key == keyboard.Key.space:
            key_char = ' '
        elif key == keyboard.Key.enter:
            key_char = '\n'
        elif key == keyboard.Key.backspace:
            self.typed_text = self.typed_text[:-1]
            print("\033[A                             \033[A")  # Move up and clear line
            key_char = ''

        # Append the key character to the typed text
        self.typed_text += key_char

        if self.last_time:
            interval = current_time - self.last_time
            self.timings.append(interval)
            print(f"{self.typed_text} (Last key delay: {interval:.4f} seconds)")
        else:
            print(self.typed_text)

        self.last_time = current_time
        self.last_key = key
        self.last_key_time = current_time

    def start(self):
        with keyboard.Listener(on_press=self.on_key_press) as listener:
            listener.join()


import numpy as np
import matplotlib.pyplot as plt

def compute_fft_from_file(filename):
    # Load data from file
    timings = np.loadtxt(filename)

    # Compute the FFT
    fft_values = np.fft.fft(timings)

    # Compute the frequencies for the FFT values
    frequencies = np.fft.fftfreq(len(timings))
    return frequencies

# Provide the path to your file here
file_path = "keystroke_timings.txt"
file_path_m2 = "keystroke_timings_m2.txt"
file_path_p = "keystroke_timings_p.txt"
file_path_p2 = "keystroke_timings_p2.txt"
file_path_m3 = "keystroke_timings_m3.txt"
przemek = compute_fft_from_file(file_path_p)
michal = compute_fft_from_file(file_path)
michal1 = compute_fft_from_file(file_path_m2)
przemek2 = compute_fft_from_file(file_path_p2)
michal2 = compute_fft_from_file(file_path_m3)

def filter_array(arr):
    avg = np.mean(arr)
    std = np.std(arr)
    lower_bound = avg - 2*std
    upper_bound = avg + 2.5*std

    return [x for x in arr if x <= upper_bound]

przemek = filter_array(przemek)
przemek2 = filter_array(przemek2)
michal = filter_array(michal)
michal1 = filter_array(michal1)
michal2 = filter_array(michal2)

n = min(len(michal), len(przemek), len(michal1), len(przemek2), len(michal2))

przemek = przemek[:n]
michal = michal[:n]
michal1 = michal1[:n]
michal2 = michal2[:n]
przemek2 = przemek2[:n]

import math

def euclidean_distance(p, q):
    if len(p) != len(q):
        raise ValueError("Both points must have the same dimension")
    return math.sqrt(sum((pi - qi)**2 for pi, qi in zip(p, q)))

print(euclidean_distance(przemek, przemek2))
print()
print(euclidean_distance(michal1, michal))
print(euclidean_distance(michal, michal2))
print(euclidean_distance(michal1, michal2))
print()
print(euclidean_distance(przemek2, michal2))
print(euclidean_distance(przemek, michal))

# if __name__ == "__main__":
#     timer = KeyStrokeTimer()
#     timer.start()
