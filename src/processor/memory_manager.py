import psutil
import cv2
import gc
from pathlib import Path

class MemoryManager:
    def __init__(self, logger):
        self.logger = logger
        self.memory_threshold = 80
        self.memory_critical = 90
        self.cache_dir = Path('.cache')

    def monitor_memory(self):
        """Monitor system memory usage"""
        memory = psutil.virtual_memory()
        if memory.percent > self.memory_critical:
            self.logger.warning("Critical memory usage detected! Attempting cleanup...")
            self.cleanup_memory()
        return memory.percent

    def cleanup_memory(self):
        """Perform memory cleanup operations"""
        gc.collect()
        cv2.destroyAllWindows()
        if self.cache_dir.exists():
            files = list(self.cache_dir.glob('*'))
            if len(files) > 10:
                files.sort(key=lambda x: x.stat().st_mtime)
                for f in files[:-10]:
                    f.unlink()