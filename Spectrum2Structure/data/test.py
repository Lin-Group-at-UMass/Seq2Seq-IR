import psutil
import platform
from datetime import datetime
import subprocess
from datetime import datetime

def get_size(bytes, suffix="B"):
    """
    Convert bytes to a more readable format (e.g., KB, MB, GB, etc.)
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f} {unit}{suffix}"
        bytes /= factor

# Get detailed CPU information using lscpu
def get_cpu_model():
    try:
        result = subprocess.run(['lscpu'], stdout=subprocess.PIPE)
        return result.stdout.decode('utf-8')
    except Exception as e:
        return str(e)

# System information
# System information
def system_info():
    print("=" * 20, "System Information", "=" * 20)
    uname = platform.uname()
    print(f"System: {uname.system}")
    print(f"Node Name: {uname.node}")
    print(f"Release: {uname.release}")
    print(f"Version: {uname.version}")
    print(f"Machine: {uname.machine}")
    print(f"Processor: {uname.processor}")


# CPU information
def cpu_info():
    print("=" * 20, "CPU Info", "=" * 20)
    print(f"Physical cores: {psutil.cpu_count(logical=False)}")
    print(f"Total cores: {psutil.cpu_count(logical=True)}")
    cpufreq = psutil.cpu_freq()
    print(f"Max Frequency: {cpufreq.max:.2f}Mhz")
    print(f"Min Frequency: {cpufreq.min:.2f}Mhz")
    print(f"Current Frequency: {cpufreq.current:.2f}Mhz")
    print("CPU Usage Per Core:")
    for i, percentage in enumerate(psutil.cpu_percent(percpu=True, interval=1)):
        print(f"Core {i}: {percentage}%")
    print(f"Total CPU Usage: {psutil.cpu_percent()}%")

    # Get detailed CPU model information
    print("=" * 20, "Detailed CPU Model", "=" * 20)
    print(get_cpu_model())

# Memory Information
def memory_info():
    print("="*20, "Memory Information", "="*20)
    svmem = psutil.virtual_memory()
    print(f"Total: {get_size(svmem.total)}")
    print(f"Available: {get_size(svmem.available)}")
    print(f"Used: {get_size(svmem.used)}")
    print(f"Percentage: {svmem.percent}%")
    print("="*20, "SWAP", "="*20)
    swap = psutil.swap_memory()
    print(f"Total: {get_size(swap.total)}")
    print(f"Free: {get_size(swap.free)}")
    print(f"Used: {get_size(swap.used)}")
    print(f"Percentage: {swap.percent}%")

# Main function to call all the information
def main():
    system_info()
    get_cpu_model()
    cpu_info()
    memory_info()

if __name__ == "__main__":
    main()
