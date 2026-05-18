# preset_selector.py
import sys
import threading
import time

try:
    import msvcrt
    WINDOWS = True
except ImportError:
    WINDOWS = False

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
        print(
            f"\r  Your choice (or wait {remaining}s "
            f"to keep current preset): ",
            end="",
            flush=True
        )
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
                selected[0] = choices[key]
                done[0]     = True
                break
            elif key in ["\r", "\n"]:
                selected[0] = "KEEP"
                done[0]     = True
                break
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
print()
result = selected[0] if selected[0] else "KEEP"
print(f"SELECTED:{result}")
