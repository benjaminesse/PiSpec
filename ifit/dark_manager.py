# -*- coding: utf-8 -*-
"""
Dark spectrum acquisition and management for PiSpec.

Features (acquisition-only):
  - Acquire one dark spectrum at a list of integration times (ms)
  - Save each dark as CSV and a compact NPZ index for fast lookup
  - Load cached darks on startup (for verification/inspection)
  - Return the exact-match dark for a given integration time (or nearest)
"""

from __future__ import annotations
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
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
        """Write NPZ index for quick reload."""
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

        # Refuse to run if no device present (wrapper exposes .spectro when connected)
        if getattr(spectro, "spectro", None) is None:
            raise RuntimeError("No spectrometer connected (spectro.spectro is None)")

        # Remember current IT and restore afterwards
        prev_it = int(getattr(spectro, "integration_time", 0) or 0)

        acquired = 0
        try:
            for it in integration_times_ms:
                try:
                    spectro.update_integration_time(int(it))

                    # Your wrapper exposes get_spectrum(fname, gps=...)
                    tmp_fname = str(self.root / f"__tmp_dark_{int(it):03d}.txt")
                    # gps=None is fine for darks
                    [x, y_arr], info = spectro.get_spectrum(tmp_fname, gps=None)

                    wl = np.asarray(x, dtype=float)
                    y = np.asarray(y_arr, dtype=float)
                    self.darks[int(it)] = (wl, y)
                    acquired += 1

                    # Clean up the temporary measurement file (we only need arrays)
                    try:
                        os.remove(tmp_fname)
                    except OSError:
                        pass

                    if save_csv:
                        # Timestamp for file naming
                        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")[:-3]

                        fname = self.root / f"{timestamp}_dark_{int(it)}ms.csv"
                        header = (
                            "PiSpec Dark Spectrum\n"
                            f"serial_number; {spectro.serial_number}\n"
                            f"spectrum_number;{spectro.spec_no}\n"
                            f"timestamp; {timestamp}\n"
                            f"integration_time; {int(it)}\n"
                            f"coadds; {spectro.coadds}\n"
                            f"Pixels; {getattr(spectro, 'pixels', len(wl))}\n"
                            "Wavelength (nm),Intensity (arb)"
                        )
                        np.savetxt(fname, np.column_stack([wl, y]), delimiter=",", header=header)
                        logger.info("Saved dark CSV: %s", fname)


                except Exception as e:
                    logger.exception("Failed dark acquisition at %sms: %s", it, e)

        finally:
            # Restore previous integration time (best effort)
            try:
                if prev_it > 0:
                    spectro.update_integration_time(prev_it)
            except Exception:
                logger.warning("Could not restore previous integration time to %d ms", prev_it)

        if acquired:
            self.save()
        else:
            logger.warning("No darks acquired.")

    # ------------- Access (optional) -------------

    def get(
        self,
        integration_time_ms: int,
        *,
        nearest_ok: bool = False
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return (wavelengths, dark_counts) for an IT; optional nearest fallback."""
        it = int(integration_time_ms)
        if it in self.darks:
            return self.darks[it]
        if not nearest_ok or not self.darks:
            return None
        keys = np.array(sorted(self.darks.keys()), dtype=int)
        idx = int(np.argmin(np.abs(keys - it)))
        nearest = int(keys[idx])
        logger.warning("Exact dark for %sms not found; using nearest %sms", it, nearest)
        return self.darks[nearest]


# -------- Convenience helpers for run_pispec.py --------

def acquire_startup_darks(
    spectro,
    out_dir: Path,
    times_ms: List[int] | None = None,
    coadds: int = 1
) -> DarkLibrary:
    """Convenience: build library by acquiring darks at startup."""
    if times_ms is None:
        times_ms = list(range(50, 301, 10))  # default 50..300 step 10 (ms)
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
