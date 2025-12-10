import sys
import glob
import logging
import re               # NEW
from collections import defaultdict  # NEW

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from ifit.parameters import Parameters
from ifit.spectral_analysis import Analyser
from ifit.load_spectra import read_spectrum, average_spectra


# =============================================================================
# Setup log output to standard output
# =============================================================================

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
date_fmt = '%H:%M:%S'
formatter = logging.Formatter('%(asctime)s - %(message)s', date_fmt)
handler.setFormatter(formatter)
logger.addHandler(handler)

# =============================================================================
# Define analysis files
# =============================================================================

# Main file path to spectra
fpath = r'c:\Users\w04272ar\OneDrive - The University of Manchester\UOM PhD\2026_01_Guatemala\PiSpec\Example PiSpec Results\20251210_094703\spectra'

# Dark spectra path
dpath = r"c:\Users\w04272ar\OneDrive - The University of Manchester\UOM PhD\2026_01_Guatemala\PiSpec\Example PiSpec Results\20251210_094703\dark"

# Set the location to save the spectra
save_path = r'C:\Users\w04272ar\OneDrive - The University of Manchester\UOM PhD\2026_01_Guatemala\PiSpec\Example PiSpec Results\20251210_094703\ifit_results.csv'

# Set the spectra type
spec_type = 'iFit'

# Set the dark, reference and measurement spectra numbers
meas_fnames = glob.glob(f'{fpath}/spectrum_*')
dark_fnames = glob.glob(f'{dpath}/*ms*')

# Sort the files
dark_fnames.sort()
meas_fnames.sort()

# Set backslashes to forward slashes (if using windows)
dark_fnames = [f.replace('\\', '/') for f in dark_fnames]
meas_fnames = [f.replace('\\', '/') for f in meas_fnames][:]

# Control whether plotting is turned on or off
plotting_flag = True

# =============================================================================
# Parameter Setup
# =============================================================================

# Create parameter dictionary
params = Parameters()

# Add the gases
params.add('SO2',  value=1.0e16, vary=True, xpath='Ref/SO2_295K.txt')
params.add('O3',   value=1.0e19, vary=True, xpath='Ref/O3_Voigt_223K.txt')
params.add('Ring', value=0.1,    vary=True, xpath='Ref/Ring.txt')

# Add background polynomial parameters
params.add('bg_poly0', value=0.0, vary=True)
params.add('bg_poly1', value=0.0, vary=True)
params.add('bg_poly2', value=0.0, vary=True)
params.add('bg_poly3', value=1.0, vary=True)

# Add intensity offset parameters
params.add('offset', value=0.0, vary=True)

# Add wavelength stretch and shift parameters
params.add('shift0', value=0.0, vary=True)
params.add('shift1', value=0.1, vary=True)

# Add ILS parameters
params.add('fwem', value=0.674, vary=False)
params.add('k',    value=2.53,  vary=False)
params.add('a_w',  value=-0.12, vary=False)
params.add('a_k',  value=-1.3,  vary=False)

# Generate the analyser
analyser = Analyser(
    params,
    fit_window=[310, 320],
    frs_path='Ref/sao2010.txt',
    stray_flag=True,
    stray_window=[280, 290],
)

print(params.pretty_print(cols='all'))

print('Preparing dark spectra by integration time...\n')

print('Preparing dark spectra by integration time...\n')

from collections import defaultdict
import numpy as np

# -------------------------------------------------------------------------
# Helper: read a dark file (header + 2-column data)
# -------------------------------------------------------------------------
def read_dark_file(fname):
    """
    Reads a PiSpec dark file:
      - Parses integration_time from header line '# integration_time; 50'
      - Loads wavelength + intensity data from the body (2 columns)
    Returns: x (nm), y (counts), int_time_ms (float or None)
    """
    int_time_ms = None

    # First pass: parse header for integration_time
    with open(fname, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                # reached data
                break
            if 'integration_time' in line:
                # e.g. '# integration_time; 50'
                # strip leading '#', then split on ';'
                parts = line.lstrip('#').strip().split(';')
                if len(parts) >= 2:
                    try:
                        int_time_ms = float(parts[1])
                    except ValueError:
                        pass

    # Second pass: load data (2 columns) ignoring comment lines
    # Most likely comma-separated; if your files are tab-separated this
    # still works because delimiter=None falls back to any whitespace.
    data = np.loadtxt(fname, comments='#', delimiter=',')
    if data.ndim == 1:
        # single row: make it 2D
        data = data.reshape(1, -1)

    x = data[:, 0]
    y = data[:, 1]

    return x, y, int_time_ms

# -------------------------------------------------------------------------
# Group darks by integration time and average them
# -------------------------------------------------------------------------
dark_groups = defaultdict(list)   # it_ms -> list of (x, y) arrays

for fname in dark_fnames:
    try:
        x_d, y_d, int_time_d = read_dark_file(fname)
    except Exception as e:
        logger.warning(f"Error reading dark file {fname}: {e}")
        continue

    if int_time_d is None:
        logger.warning(f"No integration_time found in dark metadata: {fname}")
        continue

    try:
        it_ms = int(round(float(int_time_d)))
    except Exception:
        logger.warning(
            f"Could not convert dark integration_time '{int_time_d}' to int for {fname}"
        )
        continue

    dark_groups[it_ms].append((x_d, y_d))

dark_spectra = {}  # it_ms -> (x_dark, dark_avg)

for it_ms, xy_list in dark_groups.items():
    xs = [xy[0] for xy in xy_list]
    ys = [xy[1] for xy in xy_list]

    # Use the first wavelength grid as reference
    x0 = xs[0]

    # Interpolate any mismatched grids onto x0
    ys_interp = []
    for x_d, y_d in zip(xs, ys):
        if len(x_d) != len(x0) or not np.allclose(x_d, x0):
            ys_interp.append(np.interp(x0, x_d, y_d))
        else:
            ys_interp.append(y_d)

    dark_avg = np.mean(ys_interp, axis=0)
    print(dark_avg)
    dark_spectra[it_ms] = (x0, dark_avg)
    logger.info(f"Averaged {len(xy_list)} dark spectra for {it_ms} ms")

if not dark_spectra:
    logger.warning("No dark spectra loaded – proceeding without dark subtraction!")

print('Done!\n')


# =============================================================================
# Generate the figure
# =============================================================================

if plotting_flag:
    print('Making plot canvas...')

    # Make the figure and define the subplot grid
    fig = plt.figure(figsize=[10, 6.4])
    gs = GridSpec(2, 2)

    # Define axes
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])

    # Define plot lines
    l0, = ax0.plot([], [], 'C0x-')  # Measured spectrum
    l1, = ax0.plot([], [], 'C1-')   # Model fit

    l2, = ax1.plot([], [], 'C0x-')  # Residual

    l3, = ax2.plot([], [], 'C0x-')  # Measured OD
    l4, = ax2.plot([], [], 'C1-')   # Fit OD

    l5, = ax3.plot([], [], 'C2o-')  # Time series

    print('Done!\n')

# =============================================================================
# Run analysis
# =============================================================================

print('Beginning analysis...')

# Make a list of column names
cols = ['Number', 'Time']
for par in params:
    cols += [par, f'{par}_err']
cols += ['fit_quality', 'int_lo', 'int_hi', 'int_av', 'Lat', 'Lon', 'Alt', "integration_time"]

with open(save_path, 'w') as w:
    # Write header
    for c in cols:
        w.write(f'{c},')
    w.write('\n')

    for i, fname in enumerate(tqdm(meas_fnames)):

        # Read the spectrum
        x, y, spec_info, read_err = read_spectrum(fname, spec_type)

        # Get integration time from metadata (assumed in ms)
        int_time = spec_info.get("integration_time", None)
        if int_time is None:
            logger.warning(f"No integration_time found in metadata for {fname}; "
                           "no dark subtraction performed.")
            y_corr = y
        else:
            # Force to integer ms for matching
            try:
                int_time_ms = int(round(float(int_time)))
            except Exception:
                logger.warning(f"Could not convert integration_time '{int_time}' "
                               f"to int for {fname}; no dark subtraction performed.")
                int_time_ms = None

            if int_time_ms is not None and dark_spectra:
                # Prefer exact match; if not available, pick nearest integration time
                if int_time_ms in dark_spectra:
                    x_dark, dark = dark_spectra[int_time_ms]
                else:
                    # Nearest available as a fallback
                    available = np.array(list(dark_spectra.keys()))
                    nearest = int(available[np.argmin(np.abs(available - int_time_ms))])
                    x_dark, dark = dark_spectra[nearest]
                    logger.warning(
                        f"No dark for {int_time_ms} ms; using nearest dark at {nearest} ms"
                    )

                # Ensure dark spectrum is on the same grid
                if len(dark) != len(y) or not np.allclose(x_dark, x):
                    # Interpolate dark onto measurement wavelength grid
                    dark_interp = np.interp(x, x_dark, dark)
                else:
                    dark_interp = dark

                y_corr = y - dark_interp
            else:
                y_corr = y

        # Perform the fitting (use dark-corrected spectrum)
        fit = analyser.fit_spectrum(
            [x, y_corr],
            update_params=True,
            interp_method='linear',
            calc_od=['SO2'],
            prefit_shift=0,
        )

        # Write the results to the CSV
        w.write(f'{i},{spec_info["timestamp"].strftime("%Y-%m-%dT%H-%M-%S-%f")[:-3]},')

        for par in fit.params.values():
            w.write(f'{par.fit_val},{par.fit_err},')

        w.write(
            f'{fit.nerr},{fit.int_lo},{fit.int_hi},{fit.int_av},'
            f'{spec_info["lat"]},{spec_info["lon"]},{spec_info["alt"]},{int_time}\n'
        )

        if plotting_flag:
            l0.set_data(fit.grid, fit.spec)
            l1.set_data(fit.grid, fit.fit)
            l2.set_data(fit.grid, fit.resid)
            l3.set_data(fit.grid, fit.meas_od['SO2'])
            l4.set_data(fit.grid, fit.synth_od['SO2'])

            for ax in [ax0, ax1, ax2, ax3]:
                ax.relim()
                ax.autoscale_view()

            plt.pause(0.01)
            if i == 0:
                plt.tight_layout()

    if plotting_flag:
        plt.show()

    print('Done!\n')

print('Done!')
