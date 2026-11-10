"""Log PV readings from the Arduino/ESP32 over USB serial into data/solar_data.csv.

The microcontroller (hardware/solar_logger.ino) prints one CSV line per minute:
    voltage_V,current_mA,power_mW[,temperature_C,light_lux]
This script timestamps each line with the PC clock and appends it to the CSV,
so the dashboard can be refreshed at any time while logging continues.

Usage:
    python serial_logger.py --list              # show available ports
    python serial_logger.py --port COM5         # Windows
    python serial_logger.py --port /dev/ttyUSB0 # Linux / Raspberry Pi
"""
import argparse
import csv
import sys
import time
from datetime import datetime

import config

HEADER = [config.COL_TIME, config.COL_VOLT, config.COL_CURR, config.COL_POWER,
          config.COL_TEMP, config.COL_LUX]


def list_ports():
    from serial.tools import list_ports as lp
    ports = list(lp.comports())
    if not ports:
        print("No serial ports found. Is the board plugged in?")
    for p in ports:
        print(f"{p.device:12s} {p.description}")


def parse_line(line):
    """Return a list of floats, or None for comments / malformed lines."""
    line = line.strip()
    if not line or line.startswith("#") or line[0].isalpha():
        return None
    try:
        # empty fields = optional sensor not fitted -> None
        values = [float(v) if v.strip() else None for v in line.split(",")]
    except ValueError:
        return None
    if len(values) < 3 or None in values[:3]:
        return None
    return (values + [None, None])[:5]


def run(port, baud, out_path):
    import serial

    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out_path.exists() or out_path.stat().st_size == 0
    print(f"Logging {port} @ {baud} baud -> {out_path}  (Ctrl+C to stop)")

    while True:
        try:
            with serial.Serial(port, baud, timeout=5) as ser, \
                    open(out_path, "a", newline="") as f:
                writer = csv.writer(f)
                if new_file:
                    writer.writerow(HEADER)
                    new_file = False
                time.sleep(2)            # board resets when the port opens
                ser.reset_input_buffer()
                while True:
                    raw = ser.readline().decode("utf-8", errors="ignore")
                    values = parse_line(raw)
                    if values is None:
                        if raw.strip():
                            print(f"  [board] {raw.strip()}")
                        continue
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    writer.writerow([ts] + ["" if v is None else v for v in values])
                    f.flush()
                    print(f"{ts}  V={values[0]:.2f}  I={values[1]:.1f} mA  P={values[2]:.1f} mW")
        except KeyboardInterrupt:
            print("\nStopped.")
            return
        except Exception as exc:  # cable pulled, port busy, etc. -> retry
            print(f"Serial error: {exc}. Retrying in 10 s...", file=sys.stderr)
            time.sleep(10)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="serial port, e.g. COM5 or /dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=9600)
    ap.add_argument("--out", default=str(config.LOGGED_DATA_FILE))
    ap.add_argument("--list", action="store_true", help="list serial ports and exit")
    args = ap.parse_args()

    if args.list or not args.port:
        list_ports()
        if not args.port:
            print("\nRe-run with --port <PORT> to start logging.")
        return
    from pathlib import Path
    run(args.port, args.baud, Path(args.out))


if __name__ == "__main__":
    main()
