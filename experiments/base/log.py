import logging

logging.basicConfig(level=logging.DEBUG)
_exp_logger = logging.getLogger()
_exp_logger.setLevel(logging.INFO)


def get_logger():
    global _exp_logger
    return _exp_logger
