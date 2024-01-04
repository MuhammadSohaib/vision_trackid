import logging
from common.env_configurations import get_env_bool


class NoLineExceptionFormatter(logging.Formatter):
    """
    Remove multiple lines so gray log treats it as 1 msg.
    """

    def formatException(self, exc_info):
        result = super(NoLineExceptionFormatter, self).formatException(exc_info)
        return repr(result)

    def format(self, record):
        s = super(NoLineExceptionFormatter, self).format(record)
        if record.exc_text:
            s = s.replace("\n", "\r")
        return s


def set_logger_attributes(service_name):
    logs = get_env_bool("logs", False)
    level = logging.DEBUG if logs else logging.INFO
    handler = logging.StreamHandler()
    formatter = NoLineExceptionFormatter("%(levelname)s: PID %(process)d: %(message)s")
    handler.setFormatter(formatter)
    logger = logging.getLogger(service_name)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger
