import subprocess
import time

if __name__ == "__main__":
    try:
        # Start Cam1 and Cam2 as subprocesses
        cam1_process = subprocess.Popen(["python3", "cam1.py"])
        cam2_process = subprocess.Popen(["python3", "cam2.py"])

        # Wait until both finish (which happens when user presses 'q' in both windows)
        cam1_process.wait()
        cam2_process.wait()

    except KeyboardInterrupt:
        print("Exiting...")
        cam1_process.terminate()
        cam2_process.terminate()
