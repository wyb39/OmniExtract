import subprocess
import threading
import os
import sys

class ProcessManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._procs = {}

    def _normalize_key(self, key):
        return os.path.normcase(os.path.normpath(os.path.abspath(str(key))))

    def register(self, key, process):
        norm_key = self._normalize_key(key)
        with self._lock:
            self._procs[norm_key] = process

    def get(self, key):
        norm_key = self._normalize_key(key)
        with self._lock:
            return self._procs.get(norm_key)

    def start_python_code(self, python_code, key, capture_output=False, text=True, log_to_key_dir=False, log_filename="process.log"):
        kwargs = {}
        log_file = None
        if log_to_key_dir:
            norm_key = self._normalize_key(key)
            dir_path = norm_key if os.path.isdir(norm_key) else os.path.dirname(norm_key)
            if not dir_path:
                dir_path = os.getcwd()
            log_path = os.path.join(dir_path, log_filename)
            log_file = open(log_path, "a", encoding="utf-8", buffering=1) if text else open(log_path, "ab")
            kwargs["stdout"] = log_file
            kwargs["stderr"] = log_file
        else:
            if capture_output:
                kwargs["stdout"] = subprocess.PIPE
                kwargs["stderr"] = subprocess.PIPE
            else:
                kwargs["stdout"] = None
                kwargs["stderr"] = None
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["preexec_fn"] = os.setsid
        kwargs["text"] = text
        p = subprocess.Popen([sys.executable, "-c", python_code], **kwargs)
        self.register(key, p)
        if log_file:
            try:
                log_file.close()
            except Exception:
                pass
        return p

    def terminate(self, key):
        p = self.get(key)
        print(p)
        if not p:
            return False
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                try:
                    os.killpg(p.pid, 9)
                except Exception:
                    p.terminate()
            return True
        finally:
            with self._lock:
                self._procs.pop(self._normalize_key(key), None)

process_manager = ProcessManager()