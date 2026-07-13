import logging
import sys
from typing import Any


class VerboseFilter(logging.Filter):
    """Filter out records marked with the verbose attribute."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not (hasattr(record, "verbose") and record.verbose)


def setup_logging(cfg: Any) -> logging.Logger:
    log_format = "[%(asctime)s] %(levelname)s: %(message)s"
    logging_cfg = getattr(cfg, "logging", None)
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper()),
        format=log_format,
        handlers=[],
        force=True,
    )
    if bool(getattr(logging_cfg, "suppress_httpx_logs", True)):
        logging.getLogger("httpx").setLevel(logging.WARNING)

    logger = logging.getLogger("MLEvolve")
    logger.handlers.clear()
    logger.propagate = False
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    if bool(getattr(logging_cfg, "write_brief_log", True)):
        brief_name = str(getattr(logging_cfg, "brief_log_filename", "MLEvolve.log"))
        file_handler = logging.FileHandler(cfg.log_dir / brief_name, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(log_format))
        file_handler.addFilter(VerboseFilter())
        logger.addHandler(file_handler)

    if bool(getattr(logging_cfg, "write_verbose_log", True)):
        verbose_name = str(getattr(logging_cfg, "verbose_log_filename", "MLEvolve.verbose.log"))
        verbose_file_handler = logging.FileHandler(cfg.log_dir / verbose_name, encoding="utf-8")
        verbose_file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(verbose_file_handler)

    if bool(getattr(logging_cfg, "write_console_log", True)):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(log_format))
        console_handler.addFilter(VerboseFilter())
        logger.addHandler(console_handler)
    return logger
