from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

from .config import Config

FILE_FORMAT = "%(asctime)s %(levelname)-7s [%(threadName)s] %(name)s: %(message)s"
CONSOLE_FORMAT = "%(asctime)s %(levelname)-7s [%(threadName)s] %(message)s"
JOURNAL_FORMAT = "%(levelname)s [%(threadName)s] %(name)s: %(message)s"

NOISY_LOGGERS = ("watchdog", "urllib3", "PIL", "requests", "absl")


def _syslog_priority(levelno: int) -> int:
    if levelno >= logging.CRITICAL:
        return 2
    if levelno >= logging.ERROR:
        return 3
    if levelno >= logging.WARNING:
        return 4
    if levelno >= logging.INFO:
        return 6
    return 7


class JournalFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        prefix = f"<{_syslog_priority(record.levelno)}>"
        return "\n".join(prefix + line for line in text.splitlines())


def setup_logging(cfg: Config, console: bool | None = None, level: str | None = None) -> Path | None:
    log_cfg = cfg.logging
    level_name = str(level or log_cfg.level).upper()
    numeric_level = getattr(logging, level_name, logging.INFO)
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(numeric_level)

    log_file: Path | None = None
    if log_cfg.file:
        log_dir = cfg.path("logs")
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / str(log_cfg.file_name)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=int(log_cfg.max_bytes),
                backupCount=int(log_cfg.backups),
                encoding="utf-8",
            )
            file_handler.setFormatter(logging.Formatter(FILE_FORMAT))
            root.addHandler(file_handler)
        except OSError as exc:
            log_file = None
            print(f"warning: cannot open log file in {log_dir}: {exc}", file=sys.stderr)

    use_console = log_cfg.console if console is None else console
    if use_console:
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(logging.Formatter(CONSOLE_FORMAT, datefmt="%H:%M:%S"))
        root.addHandler(stream)
    elif log_cfg.journal:
        journal = logging.StreamHandler(sys.stderr)
        journal.setFormatter(JournalFormatter(JOURNAL_FORMAT))
        root.addHandler(journal)

    if not root.handlers:
        root.addHandler(logging.NullHandler())

    quiet_level = max(numeric_level, logging.WARNING)
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(quiet_level)
    logging.captureWarnings(True)
    return log_file
