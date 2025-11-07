from __future__ import division
import os
import sys
import utm
import time
import yaml
import serial
import logging
import threading
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
import serial.tools.list_ports
from multiprocessing import Process, Queue
# from gpiozero import DigitalInputDevice

from pymavlink import mavutil
os.environ['MAVLINK20'] = '1'

from ifit.spectrometers import Spectrometer
from ifit.gps import GPS
from ifit.load_spectra import read_spectrum
from ifit.parameters import Parameters
from ifit.spectral_analysis import Analyser
from ifit.dark_manager import acquire_startup_darks, load_dark_library

logger = logging.getLogger()


def analyse_spec(spec_fname, analyser, fpath, q1, q2):
    """."""
    # Read in the spectrum
    x, y, info, err = read_spectrum(spec_fname, spec_type='iFit')

    # Fit the spectrum
    fit = analyser.fit_spectrum(
        spectrum=[x, y],
        update_params=True,
        resid_limit=20,
        int_limit=[0, 60000],
        prefit_shift=1,
        interp_method='linear'
    )

    # Convert lat/lon to UTM
    utm_coords = utm.from_latlon(info['lat'], info['lon'])

    # Colate results and add to the queue
    conv = 2.54e15
    res = [
        info['timestamp'], info['lat'], info['lon'], info['alt'],
        utm_coords[0], utm_coords[1], utm_coords[2], utm_coords[3],
        fit.params['SO2'].fit_val, fit.params['SO2'].fit_err,
        fit.params['SO2'].fit_val/conv, fit.params['SO2'].fit_err/conv,
        info['integration_time'], np.max(fit.spec)
    ]

    # To send results over telemetry use the following
    # mav_connection.mav.named_value_float_send(
    #     int(time.mktime(info['timestamp'].timetuple())),
    #     'So2_SCD'.encode('utf-8'),
    #     fit.params['SO2'].fit_val/conv
    # )

    head, tail = os.path.split(spec_fname)
    spectra_fname = f"{head}/spectra/{tail.replace('spectrum', 'spectra')}"

    q1.put(res)
    q2.put([info['timestamp'], fit.params['SO2'].fit_val/conv])

# ------------- MAVLink helpers  ----------------
def send_status(conn, text, severity=None):
    """Send a short status string to the GCS (STATUSTEXT, <=50 chars)."""
    if conn is None:
        return
    try:
        if severity is None:
            severity = mavutil.mavlink.MAV_SEVERITY_INFO
        s = str(text)[:50]
        conn.mav.statustext_send(severity, s.encode('utf-8'))
    except Exception as e:
        logger.warning("STATUSTEXT send failed: %s", e)


def heartbeat_thread(conn, period=2):
    """Send heartbeats + pings so the link stays alive."""
    while True:
        try:
            conn.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0
            )
            conn.mav.ping_send(int(time.time()*1e6), 0, 0, 0)
        except Exception as e:
            logger.warning("Heartbeat error: %s", e)
        time.sleep(period)


def link_monitor_thread(conn, lost_after=10):
    """
    Watch for inbound MAVLink (e.g., GCS heartbeats).
    If nothing received for 'lost_after' seconds -> LOST.
    On first message after loss -> RESTORED.
    """
    last_rx = time.time()
    lost = False
    while True:
        try:
            msg = conn.recv_match(blocking=False)
            if msg is not None:
                last_rx = time.time()
                if lost:
                    send_status(conn, "MAVLink link restored")
                    lost = False
            if (time.time() - last_rx) > lost_after and not lost:
                send_status(conn, "MAVLink link lost", mavutil.mavlink.MAV_SEVERITY_WARNING)
                lost = True
        except Exception as e:
            logger.warning("Link monitor error: %s", e)
        time.sleep(0.5)


def listener(q1, save_fname):
    """."""
    # Handle writing the results file
    with open(save_fname, 'w') as w:

        # Write the header
        h = 'Time,Lat,Lon,Alt,X,Y,ZoneNum,ZoneLett,SO2_SCD_mol,SO2_err_mol,' \
            + 'SO2_SCD_ppmm,SO2_err_ppmm,IntegrationTime,Intensity'
        w.write(h + '\n')
        print('Time\tLat\tLon\tAlt\tSO2_SCD_ppmm\tSO2_err_ppmm')

        while True:
            # Unpack the results
            res = q1.get()
            if res == 'kill':
                break
            else:
                msg = str(res[0])
                for r in res[1:]:
                    msg += f',{r}'
                w.write(msg + '\n')
                w.flush()
                print(
                    f'{res[0]}\t{res[1]}\t{res[2]}\t{res[3]}\t{res[10]}\t'
                    f'{res[11]}'
                )

def mavlink_listener(q2, mav_connection):

    conv = 2.54e15
    last_text = 0  # for optional human-readable throttled messages

    while True:

        res = [q2.get() for _ in range(100) if not q2.empty()]

        if len(res) != 0:

            so2_vals = [r[1] for r in res]
            timestamp = [r[0] for r in res]
            so2_val = np.nanmean(so2_vals)

            mav_connection.mav.named_value_float_send(
                int(time.mktime(timestamp[-1].timetuple())),
                'So2_SCD'.encode('utf-8'),
                so2_val
            )

            # (Optional) occasional human-readable text to GCS (every 5 s)
            now = time.time()
            if now - last_text > 5:
                try:
                    send_status(mav_connection, f"SO2_SCD {so2_val:.2e} mol/m^2")
                except Exception:
                    pass
                last_text = now

        time.sleep(0.5)

# =============================================================================
def gps_time_sync(gps):
    """Syncs the position and time with the GPS. Returns (ok, lat, lon)."""
    logger.info('Starting GPS sync...')

    # Get a fix from the GPS
    position = gps.get_position(time_to_wait=7200)

    if position is not None:
        ts, lat, lon, alt = position
        tstamp = ts.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f'Updating system time: {tstamp}')
        tstr = ts.strftime('%a %b %d %H:%M:%S UTC %Y')
        subprocess.call(f'sudo date -s "{tstr}"', shell=True)

        # Log the scanner location
        logger.info(
            'Scanner position:\n'
            f'Latitude:   {lat}\n'
            f'Longitutde: {lon}\n'
            f'Altitude:   {alt}\n'
        )
        return True, lat, lon
    else:
        logger.warning('GPS fix failed')
        return False, None, None

# =============================================================================
# Run main script
# =============================================================================

def run():
    """Run main program loop."""
    # Get the logger

    # Setup logger to standard output
    logger.setLevel(logging.INFO)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_formatter = logging.Formatter(
        '%(asctime)s - %(message)s', '%H:%M:%S')
    stdout_handler.setFormatter(stdout_formatter)
    logger.addHandler(stdout_handler)

# =============================================================================
#   Connect to MAVLink first so we can report status
# =============================================================================
    connection_string = '/dev/serial0'
    mav_connection = None
    try:
        mav_connection = mavutil.mavlink_connection(
            connection_string,
            baud=115200,
            source_system=1,
            source_component=0
        )
        send_status(mav_connection, "MAVLink connected")
        # Start heartbeat + link monitor threads
        threading.Thread(target=heartbeat_thread, args=(mav_connection, 2), daemon=True).start()
        threading.Thread(target=link_monitor_thread, args=(mav_connection, 10), daemon=True).start()
    except Exception as e:
        logger.exception("Failed to open MAVLink: %s", e)

# =============================================================================
#   Sync with GPS
# =============================================================================

    # Connect to the GPS
    gps = GPS()

    # Sync time & position
    gps_ok, glat, glon = gps_time_sync(gps)
    if gps_ok:
        send_status(mav_connection, f"GPS OK {glat:.1f},{glon:.1f}")
    else:
        send_status(mav_connection, "GPS fix failed", mavutil.mavlink.MAV_SEVERITY_WARNING)

    # Get the timestamp
    nowtime = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')

# =============================================================================
#   Connect to the spectrometer
# =============================================================================

    try:
        spectro = Spectrometer()
        send_status(mav_connection, "Spectrometer connected")
    except Exception as e:
        logger.exception("Spectrometer init failed: %s", e)
        send_status(mav_connection, "Spectrometer init failed", mavutil.mavlink.MAV_SEVERITY_ERROR)
        return

# =============================================================================
#   Create folders outputs
# =============================================================================

    # Create the results folder
    fpath = f'/home/pi/PiSpec/Results/{nowtime}'
    if not os.path.isdir(fpath):
        os.makedirs(fpath)
    if not os.path.isdir(f'{fpath}/spectra'):
        os.makedirs(f'{fpath}/spectra')
    if not os.path.isdir(f'{fpath}/dark'):
        os.makedirs(f'{fpath}/dark')

    # Add file handler to logger (after fpath is known)
    f_handler = logging.FileHandler(f'{fpath}/log.txt')
    f_handler.setLevel(logging.INFO)
    f_formatter = logging.Formatter('%(asctime)s - %(message)s', '%H:%M:%S')
    f_handler.setFormatter(f_formatter)
    logger.addHandler(f_handler)


    # Read in settings
    default_config = {'TargetIntensity': 50000,
                      'MinIntTime': 50,
                      'MaxIntTime': 300,
                      'IntTimeStep': 10,
                      'GPSCOMPort': 0,
                      'FitWindow': [310, 320]}
    try:
        with open('/home/pi/PiSpec/pispec_settings.yml', 'r') as ymlfile:
            load_config = yaml.load(ymlfile, Loader=yaml.FullLoader)
            config = {**default_config, **load_config}
    except FileNotFoundError:
        config = default_config

    # Set the target intensity
    target_int = config['TargetIntensity']

    # Construct an array of intergration times
    int_times = np.arange(config['MinIntTime'],
                          config['MaxIntTime'] + config['IntTimeStep'],
                          config['IntTimeStep'])


    # --- Build the integration-time grid from YAML ---
    min_it  = int(config.get('MinIntTime', 50))
    max_it  = int(config.get('MaxIntTime', 300))
    it_step = int(config.get('IntTimeStep', 10))

    int_time_grid = list(range(min_it, max_it + 1, it_step))

# =============================================================================
#   Acquire or load darks on boot (with status)
# =============================================================================
    dark_dir = Path(f'{fpath}/dark')

    try:
        send_status(mav_connection, "Dark acquisition start")
        dark_lib = acquire_startup_darks(
            spectro,
            out_dir=dark_dir,
            times_ms=int_time_grid,
            coadds=1   # one dark per integration time
        )
        logging.info("Startup dark acquisition complete at %s", int_time_grid)
        send_status(mav_connection, "Dark acquisition done")
    except Exception as e:
        logging.exception("Dark acquisition failed: %s; trying to load existing index", e)
        send_status(mav_connection, "Dark acquisition failed", mavutil.mavlink.MAV_SEVERITY_ERROR)
        dark_lib = load_dark_library(spectro, dark_dir)

    if not getattr(dark_lib, "darks", None):
        logging.warning("Dark library is empty. Proceeding WITHOUT dark subtraction.")
        send_status(mav_connection, "Dark lib empty", mavutil.mavlink.MAV_SEVERITY_WARNING)
    else:
        logging.info("Dark library contains %d entries.", len(dark_lib.darks))
        send_status(mav_connection, f"Darks ready: {len(dark_lib.darks)}")

    # Initialise a process list
    # processes = []

# =============================================================================
#   Set up iFit analyser
# =============================================================================

    # Create parameter dictionary
    params = Parameters()

    for name, info in config['FitParameters'].items():
        info['value'] = float(info['value'])
        params.add(name, **info)

    # Generate the analyser
    analyser = Analyser(params,
                        fit_window=config['FitWindow'],
                        frs_path=config['FRSPath'],
                        model_padding=1.0,
                        model_spacing=0.01,
                        flat_flag=False,
                        stray_flag=True,
                        stray_window=[280, 290],
                        ils_type='Manual')

    # Report fitting parameters
    logger.info(params.pretty_print(cols=['name', 'value', 'vary', 'xpath']))

    # Initialise a counter
    i = 0

    # Generate the writing queue
    save_fname = f'{fpath}/so2_output.csv'
    q1 = Queue()
    listen1 = Process(target=listener, args=[q1, save_fname])
    listen1.daemon = True
    listen1.start()

    # Generate the mavlink queue (keeps your float sending intact)
    q2 = Queue()
    listen2 = Process(target=mavlink_listener, args=[q2, mav_connection])
    listen2.daemon = True
    listen2.start()

    # Start switched OFF
    control_file = 'controlON'
    if os.path.isfile(control_file):
        os.remove(control_file)

    logger.info('PiSpec ready!')
    send_status(mav_connection, "PiSpec running")

    acquiring_announced = False

    
    while True:
        try:
            # Format the spectrum name and read
            spec_fname = f'{fpath}/spectra/spectrum_{i:05d}.txt'
            try:
                [x, y], info = spectro.get_spectrum(spec_fname, gps=gps)
            except Exception as e:
                logger.exception("Spectrometer read failed: %s", e)
                send_status(mav_connection, "Spectrometer read failed", mavutil.mavlink.MAV_SEVERITY_ERROR)
                time.sleep(0.5)
                continue

            # Find the maximum intensity
            max_int = np.max(y)

            # Scale the intensity to the target
            scale = target_int / max_int if max_int > 0 else 1.0

            # Scale the integration time by this factor
            int_time = spectro.integration_time * scale

            # Find the nearest value
            diff = ((int_times - int_time)**2)**0.5
            idx = np.where(diff == min(diff))[0][0]
            new_int_time = int(int_times[idx])

            # Update the integration time
            if new_int_time != spectro.integration_time:
                spectro.update_integration_time(new_int_time)
                send_status(mav_connection, f"IntTime -> {new_int_time} ms")

            # Clear any finished processes from the processes list
            try:
                p = Process(
                    target=analyse_spec,
                    args=[spec_fname, analyser, fpath, q1, q2]
                )
                p.daemon = True
                p.start()
                if not acquiring_announced:
                    send_status(mav_connection, "Acquiring spectra")
                    acquiring_announced = True
            except Exception as e:
                logger.exception("Analysis start failed: %s", e)
                send_status(mav_connection, "Analysis start failed", mavutil.mavlink.MAV_SEVERITY_ERROR)

            i += 1

        except KeyboardInterrupt:
            q1.put('kill')
            send_status(mav_connection, "PiSpec stopping", mavutil.mavlink.MAV_SEVERITY_WARNING)
            break
        except Exception as e:
            logger.exception("Unhandled loop error: %s", e)
            send_status(mav_connection, "Unhandled error", mavutil.mavlink.MAV_SEVERITY_ERROR)
            time.sleep(0.5)

    logger.info('Program ended')

    # Complete processes
    listen1.join()
    listen2.join()


if __name__ == '__main__':
    run()
