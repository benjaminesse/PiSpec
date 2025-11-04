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
from gpiozero import LED

from pymavlink import mavutil
os.environ['MAVLINK20'] = '1'

from ifit.spectrometers import Spectrometer
from ifit.gps import GPS
from ifit.load_spectra import read_spectrum
from ifit.parameters import Parameters
from ifit.spectral_analysis import Analyser
from ifit.dark_manager import acquire_startup_darks, load_dark_library

logger = logging.getLogger()

# ---- LED status helper -------------------------------------------------------
from contextlib import contextmanager

class StatusLeds:
    """
    Solid ON = step successful.
    OFF      = step not yet run or failed.
    On failure in any step: that LED OFF + error LED ON.
    """
    def __init__(self, gps_pin=27, spec_pin=21, dark_pin=13, collect_pin=26, err_pin=19):
        self._construct_leds(gps_pin, spec_pin, dark_pin, collect_pin, err_pin)

    def _construct_leds(self, gps_pin, spec_pin, dark_pin, collect_pin, err_pin):
        try:
            self.leds = {
                "gps":          LED(gps_pin),
                "spec":         LED(spec_pin),
                "dark":         LED(dark_pin),
                "collect":      LED(collect_pin),  
            }
            self.error = LED(err_pin)
        except Exception:
            # Stub for testing off-Pi
            class _Stub:
                def __init__(self, pin): self.pin=pin
                def on(self):  print(f"[LED {self.pin}] ON")
                def off(self): print(f"[LED {self.pin}] OFF")
                def close(self): pass
            self.leds = {
                "gps":     _Stub(gps_pin),
                "spec":    _Stub(spec_pin),
                "dark":    _Stub(dark_pin),
                "collect": _Stub(collect_pin),
            }
            self.error = _Stub(err_pin)

        self.all_off()

    def all_off(self):
        for led in self.leds.values(): led.off()
        self.error.off()

    def ok(self, name: str):
        self.leds[name].on()

    def fail(self, name: str):
        self.leds[name].off()
        self.error.on()

    @contextmanager
    def step(self, name: str):
        try:
            yield
        except Exception:
            self.fail(name)
            raise
        else:
            self.ok(name)

    def clear_error(self):
        self.error.off()

    def close(self):
        for led in self.leds.values():
            led.off(); led.close()
        self.error.off(); self.error.close()
# ------------------------------------------------------------------------------

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

        time.sleep(0.5)


def heartbeat_thread(conn, timeout=2):
    """Send heartbeats to router"""
    while True:
        conn.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0, 0, 0
        )
        conn.mav.ping_send(
            0, 1, 126, 0
        )
        time.sleep(timeout)
    threads.remove(threading.current_thread())

def gps_time_sync(gps):
    """Syncs the position and time with the GPS."""
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

    else:
        logger.warning('GPS fix failed')

# =============================================================================
# Run main script
# =============================================================================

def run():
    # --- logger setup (keep your existing logger init above) ---
    logger.setLevel(logging.INFO)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_formatter = logging.Formatter('%(asctime)s - %(message)s', '%H:%M:%S')
    stdout_handler.setFormatter(stdout_formatter)
    logger.addHandler(stdout_handler)

   # ===================== LED status ==========================
    gps_pin     = int(os.getenv('gps_led_pin',        '27'))  # BCM numbering
    spec_pin    = int(os.getenv('spectro_led_pin',    '21'))
    dark_pin    = int(os.getenv('dark_led_pin',       '13'))
    collect_pin = int(os.getenv('collect_led_pin',    '26'))   # <--- NEW LED
    err_pin     = int(os.getenv('error_led_pin',      '19'))

    # Constructor now receives 5 pins
    leds = StatusLeds(gps_pin, spec_pin, dark_pin, collect_pin, err_pin)

    try:
        # ================== GPS connect + time sync ==============
        with leds.step("gps"):
            gps = GPS()                 # may raise if not available
            gps_time_sync(gps)          # your existing function (logs position/time)
            # Quick confirmation of a fix
            position = gps.get_position(time_to_wait=10)
            assert position is not None, "No GPS fix within 10s"
            logger.info("GPS sync confirmed")

        # ================== Spectrometer connect =================
        with leds.step("spec"):
            spectro = Spectrometer()     # may raise if not available
            # Optionally verify it responds
            _ = spectro.integration_time  # touch a property to force driver call
            logger.info("Spectrometer connected")

        # ================== Results folder =======================
        nowtime = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
        fpath = f'/home/pi/PiSpec/Results/{nowtime}'
        os.makedirs(f'{fpath}/spectra', exist_ok=True)
        os.makedirs(f'{fpath}/dark',    exist_ok=True)

        # ================== Config ===============================
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

        target_int = config['TargetIntensity']
        int_times = np.arange(config['MinIntTime'],
                              config['MaxIntTime'] + config['IntTimeStep'],
                              config['IntTimeStep'])

        min_it  = int(config.get('MinIntTime', 50))
        max_it  = int(config.get('MaxIntTime', 300))
        it_step = int(config.get('IntTimeStep', 10))
        int_time_grid = list(range(min_it, max_it + 1, it_step))

        # ================== Darks acquisition ====================
                # ================== Darks acquisition ====================
        DARK_DIR = Path(f'{fpath}/dark')
        with leds.step("dark"):
            try:
                dark_lib = acquire_startup_darks(
                    spectro,
                    out_dir=DARK_DIR,
                    times_ms=int_time_grid,
                    coadds=1
                )
                if not getattr(dark_lib, "darks", None):
                    raise RuntimeError("Dark library empty after acquisition.")
                logger.info("Startup dark acquisition complete")
            except Exception:
                logger.exception("Dark acquisition failed; trying to load existing index")
                dark_lib = load_dark_library(spectro, DARK_DIR)
                if not getattr(dark_lib, "darks", None):
                    raise RuntimeError("No darks available (acquire and load both failed).")
                logger.info("Loaded existing dark library")

        # ================== Turn spectra aquistion LED ===========
        leds.ok("collect")
        logger.info("Starting spectral acquisition — spectra LED on")

        # ================== MAVLink ==============================
        connection_string = '/dev/serial0'
        mav_connection = mavutil.mavlink_connection(
            connection_string, baud=115200, source_system=1, source_component=0
        )

        hb_thread = threading.Thread(
            target=heartbeat_thread, name='HB_thread',
            args=(mav_connection, 5,), daemon=True
        )

        # ============== File logging handler =====================
        f_handler = logging.FileHandler(f'{fpath}/log.txt')
        f_handler.setLevel(logging.INFO)
        f_formatter = logging.Formatter('%(asctime)s - %(message)s', '%H:%M:%S')
        f_handler.setFormatter(f_formatter)
        logger.addHandler(f_handler)

        processes = []

        # ============== iFit analyser ============================
        params = Parameters()
        for name, info in config['FitParameters'].items():
            info['value'] = float(info['value'])
            params.add(name, **info)

        analyser = Analyser(params,
                            fit_window=config['FitWindow'],
                            frs_path=config['FRSPath'],
                            model_padding=1.0,
                            model_spacing=0.01,
                            flat_flag=False,
                            stray_flag=True,
                            stray_window=[280, 290],
                            ils_type='Manual')

        logger.info(params.pretty_print(cols=['name', 'value', 'vary', 'xpath']))

        i = 0
        save_fname = f'{fpath}/so2_output.csv'
        q1 = Queue(); listen1 = Process(target=listener, args=[q1, save_fname]); listen1.daemon = True; listen1.start()
        q2 = Queue(); listen2 = Process(target=mavlink_listener, args=[q2, mav_connection]); listen2.daemon = True; listen2.start()

        control_file = 'controlON'
        if os.path.isfile(control_file):
            os.remove(control_file)

        logger.info('PiSpec ready!')

        while True:
            try:
                spec_fname = f'{fpath}/spectra/spectrum_{i:05d}.txt'
                [x, y], info = spectro.get_spectrum(spec_fname, gps=gps)

                max_int = np.max(y)
                scale = target_int / max_int
                int_time = spectro.integration_time * scale
                diff = ((int_times - int_time)**2)**0.5
                idx = np.where(diff == min(diff))[0][0]
                new_int_time = int(int_times[idx])

                if new_int_time != spectro.integration_time:
                    spectro.update_integration_time(new_int_time)

                processes = [p for p in processes if p.is_alive()]

                if len(processes) < 1:
                    p = Process(target=analyse_spec, args=[spec_fname, analyser, fpath, q1, q2])
                    processes.append(p)
                    p.start()
                else:
                    logger.warning(f'Too many processes! Spectrum {i} not analysed')
                i += 1

            except KeyboardInterrupt:
                q1.put('kill')
                break

        logger.info('Program ended')
        listen1.join(); listen2.join()
        for p in processes: p.join()

    except Exception as e:
        # Any failure by this point: error LED is already ON by the step() context
        logger.exception("Fatal error in run(): %s", e)
    finally:
        # Keep LEDs showing final state or turn all off if you prefer
        # leds.all_off()
        try:
            leds.close()
        except Exception:
            pass



if __name__ == '__main__':
    run()
