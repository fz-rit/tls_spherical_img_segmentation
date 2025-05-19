# tools/logger_setup.py
import logging
import time
from pathlib import Path
from tools.load_tools import CONFIG

time_str = time.strftime("%Y-%m-%d_%H-%M", time.localtime())
log_path = Path(CONFIG['root_dir']) / f"log_output_{'_'.join(CONFIG['model_name_ls'])}_{time_str}.log"

class Logger:
    _instance = None  # Singleton to avoid reinitialization

    def __new__(cls, level=logging.INFO):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._init_logger(level)
        return cls._instance

    def _init_logger(self, level):
        self.logger = logging.getLogger("global_logger")
        self.logger.setLevel(level)

        if not self.logger.handlers:
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

            # File handler
            file_handler = logging.FileHandler(log_path, mode='w')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(level)

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def info(self, msg): self.logger.info(msg)
    def debug(self, msg): self.logger.debug(msg)
    def warning(self, msg): self.logger.warning(msg)
    def error(self, msg): self.logger.error(msg)
    def critical(self, msg): self.logger.critical(msg)
