"""Ask Windows not to idle-sleep while the pipeline runs (SetThreadExecutionState, released when this
process exits; no power settings are changed). Exits when the flag file appears. A closed lid still sleeps."""
import ctypes
import os
import sys
import time

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
flag = sys.argv[1]
if os.name == "nt":
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
while not os.path.exists(flag):
    time.sleep(60)
