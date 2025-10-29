from pathlib import Path
import numpy as np
from ifit.dark_manager import acquire_startup_darks, load_dark_library

# --- fake spectrometer stub ---
class FakeSpectrometer:
    def __init__(self):
        self.integration_time = 50
        self.serial_number = "FAKE123"
        self.pixels = 2048
    def update_integration_time(self, ms):
        self.integration_time = ms
    def acquire_spectrum(self, coadds=1, save=False):
        wl = np.linspace(300, 400, self.pixels)
        noise = np.random.normal(0, 30, self.pixels)
        intensity = (1000 / wl) + noise + 100  # fake baseline + noise
        spec = np.vstack([wl, intensity])
        info = {"integration_time": self.integration_time}
        return spec, info

# --- test the dark manager ---
fake_spec = FakeSpectrometer()
out_dir = Path("DarkTest")

# acquire darks at 50, 100, 150 ms
lib = acquire_startup_darks(fake_spec, out_dir, [50, 100, 150])
print("Saved darks:", lib.darks.keys())

# load them back
lib2 = load_dark_library(fake_spec, out_dir)
print("Loaded darks:", lib2.darks.keys())

# simulate a measurement at 100 ms and subtract
wl = np.linspace(300, 400, fake_spec.pixels)
meas = np.vstack([wl, np.linspace(1000, 500, fake_spec.pixels) + 100])
corr = lib2.subtract(meas, 100)
print("Before mean:", np.mean(meas[1]), "After mean:", np.mean(corr[1]))
