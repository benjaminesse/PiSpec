# -*- coding: utf-8 -*-
"""
Dark spectrum acquisition and management for PiSpec.

Features:
  - Acquire one dark spectrum at a list of integration times (ms)
  - Save each dark as CSV and a compact NPZ index for fast lookup
  - Load cached darks on startup
  - Return the exact-match dark for a given integration time (or nearest)
  - Subtract the dark from a measured spectrum safely (wavelength check + interp)
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DarkLibrary:
    root: Path
    serial: str
    # map: integration_time_ms -> (wavelengths, dark_counts)
    darks: Dict[int, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    @property
    def index_path(self) -> Path:
        return self.root / f"{self.serial}_dark_index.npz"

    # ---------------- I/O ----------------

    def load(self) -> None:
        """Load cached darks from NPZ if available."""
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            logger.warning("No dark index found at %s", self.index_path)
            return
        data = np.load(self.index_path, allow_pickle=True)
        times = data["times"].tolist()
        wls = data["wavelengths"].tolist()
        vals = data["values"].tolist()
        self.darks = {int(t): (np.array(w), np.array(v)) for t, w, v in zip(times, wls, vals)}
        logger.info("Loaded %d dark spectra for %s", len(self.darks), self.serial)

    def save(self) -> None:
        if not self.darks:
            logger.warning("No darks to save for %s", self.serial)
            return
        times = np.array(sorted(self.darks.keys()), dtype=int)
        wavelengths = [self.darks[t][0] for t in times]
        values = [self.darks[t][1] for t in times]
        self.root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.index_path, times=times, wavelengths=wavelengths, values=values)
        logger.info("Saved dark index with %d entries -> %s", len(times), self.index_path)

    # ------------- Acquisition -------------

    def acquire(
        self,
        spectro,
        integration_times_ms: List[int],
        coadds: int = 1,
        save_csv: bool = True,
    ) -> None:
        """
        Acquire 1 dark spectrum at each specified integration time.
        The spectrometer must be optically dark (shutter closed / lens cap on).
        """
        self.root.mkdir(parents=True, exist_ok=True)
        acquired = 0
        for it in integration_times_ms:
            try:
                spectro.update_integration_time(int(it))
                spec, info = spectro.acquire_spectrum(coadds=coadds, save=False)
                wl = np.asarray(spec[0, :], dtype=float)
                y = np.asarray(spec[1, :], dtype=float)
                self.darks[int(it)] = (wl, y)
                acquired += 1

                if save_csv:
                    fname = self.root / f"{self.serial}_dark_{int(it)}ms.csv"
                    header = (
                        "PiSpec Dark Spectrum\n"
                        f"Serial: {self.serial}\n"
                        f"Integration time (ms): {int(it)}\n"
                        f"Pixels: {getattr(spectro, 'pixels', len(wl))}\n"
                        "Wavelength (nm),Intensity (arb)"
                    )
                    np.savetxt(fname, np.column_stack([wl, y]), delimiter=",", header=header)
                    logger.info("Saved dark CSV: %s", fname)

            except Exception as e:
                logger.exception("Failed dark acquisition at %sms: %s", it, e)

        if acquired:
            self.save()
        else:
            logger.warning("No darks acquired.")

    # ------------- Access & subtraction -------------

    def get(self, integration_time_ms: int, *, nearest_ok: bool = False
            ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        it = int(integration_time_ms)
        if it in self.darks:
            return self.darks[it]
        if not nearest_ok or not self.darks:
            return None
        # choose nearest time
        keys = np.array(sorted(self.darks.keys()), dtype=int)
        idx = int(np.argmin(np.abs(keys - it)))
        nearest = int(keys[idx])
        logger.warning("Exact dark for %sms not found; using nearest %sms", it, nearest)
        return self.darks[nearest]

    def subtract(self, spectrum: np.ndarray, integration_time_ms: int,
                 *, clip_floor: float | None = 0.0) -> np.ndarray:
        """
        Subtract appropriate dark spectrum from a [2 x N] spectrum array.
        Returns a new 2xN array [wavelengths; corrected_intensities].
        """
        if spectrum is None or spectrum.ndim != 2 or spectrum.shape[0] != 2:
            raise ValueError("Spectrum must be 2xN array [wavelengths; intensities]")

        wl_meas = spectrum[0, :]
        y_meas = spectrum[1, :].astype(float, copy=True)

        dark = self.get(integration_time_ms, nearest_ok=False)
        if dark is None:
            raise KeyError(f"No dark available for {integration_time_ms} ms")
        wl_dark, y_dark = dark

        # If wavelengths differ slightly, interpolate dark onto measurement grid
        if wl_dark.shape != wl_meas.shape or not np.allclose(wl_dark, wl_meas, rtol=0, atol=1e-6):
            y_dark_interp = np.interp(wl_meas, wl_dark, y_dark)
        else:
            y_dark_interp = y_dark

        y_corr = y_meas - y_dark_interp
        if clip_floor is not None:
            y_corr = np.maximum(y_corr, clip_floor)
        return np.vstack([wl_meas, y_corr])


# -------- Convenience helpers for run_pispec.py --------

def acquire_startup_darks(spectro, out_dir: Path,
                          times_ms: List[int] | None = None, coadds: int = 1
                          ) -> DarkLibrary:
    """Convenience: build library by acquiring darks at startup."""
    if times_ms is None:
        times_ms = list(range(50, 301, 10))  # 50..300 step 10 (ms) default
    serial = getattr(spectro, 'serial_number', 'UNKNOWN')
    lib = DarkLibrary(root=Path(out_dir), serial=serial)
    lib.acquire(spectro, times_ms, coadds=coadds, save_csv=True)
    return lib


def load_dark_library(spectro, out_dir: Path) -> DarkLibrary:
    """Load previously saved dark library from disk."""
    serial = getattr(spectro, 'serial_number', 'UNKNOWN')
    lib = DarkLibrary(root=Path(out_dir), serial=serial)
    lib.load()
    return lib
