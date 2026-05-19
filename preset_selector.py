# preset_selector.py
import sys
import threading
import time

try:
    import msvcrt
    WINDOWS = True
except ImportError:
    WINDOWS = False

RESULT_FILE = "temp_selection.txt"

choices = {
    "1": "fastest",
    "2": "fast",
    "3": "balanced",
    "4": "smart",
    "5": "nuclear",
    "6": "gemma12b",
}

selected = [None]
done     = [False]
TIMEOUT  = 10


def countdown():
    remaining = TIMEOUT
    while remaining > 0 and not done[0]:
        # Write directly to CON (Windows console) to bypass any redirection
        sys.stderr.write(
            f"\r  Press 1-6 to switch preset, or ENTER to keep current  "
            f"({remaining}s) "
        )
        sys.stderr.flush()
        time.sleep(1)
        remaining -= 1
    if not done[0]:
        selected[0] = "KEEP"
        done[0]     = True


timer = threading.Thread(target=countdown, daemon=True)
timer.start()

if WINDOWS:
    while not done[0]:
        if msvcrt.kbhit():
            key = msvcrt.getwch()
            if key in choices:
                sys.stderr.write(
                    f"\r  ✓ Selected [{key}] → {choices[key].upper()}"
                    f"                              \n"
                )
                sys.stderr.flush()
                selected[0] = choices[key]
                done[0]     = True
                break
            elif key in ("\r", "\n"):
                sys.stderr.write(
                    f"\r  ✓ [ENTER] → keeping current preset"
                    f"                              \n"
                )
                sys.stderr.flush()
                selected[0] = "KEEP"
                done[0]     = True
                break
            else:
                sys.stderr.write(
                    f"\r  ✗ Unknown key — press 1-6 or ENTER"
                    f"                              "
                )
                sys.stderr.flush()
else:
    try:
        key = input()
        if key.strip() in choices:
            selected[0] = choices[key.strip()]
        else:
            selected[0] = "KEEP"
        done[0] = True
    except Exception:
        selected[0] = "KEEP"
        done[0]     = True

timer.join(timeout=TIMEOUT + 1)
sys.stderr.write("\n")
sys.stderr.flush()

result = selected[0] if selected[0] else "KEEP"

# Write result to file directly — stdout is no longer used for this
with open(RESULT_FILE, "w") as f:
    f.write(f"SELECTED:{result}\n")
