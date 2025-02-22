import platform
import os
import multiprocessing

def gather_system_info():
    system_info = {
        "Operating System": platform.system(),
        "OS Version": platform.version(),
        "Architecture": platform.machine(),
        "CPU Cores": multiprocessing.cpu_count(),
        "Current Directory": os.getcwd()
    }
    
    return system_info