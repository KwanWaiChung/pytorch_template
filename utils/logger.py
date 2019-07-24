import logging
import logging.config
import os


s = os.path.join(os.path.dirname(__file__), "..", "log.txt")
logging.config.fileConfig(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "logging.conf"),
    #  disable_existing_loggers=False,
    defaults={
        "logfilename": os.path.join(os.path.dirname(__file__), "..", "log.txt")
    },
)


def getlogger(name=__name__):
    logger = logging.getLogger(name)
    return logger
