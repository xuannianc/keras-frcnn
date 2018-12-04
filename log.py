import logging
import sys

logger = logging.getLogger('frcnn')
logger.setLevel(logging.DEBUG)  # default log level
format = logging.Formatter("%(asctime)s %(filename)-16s %(levelname)-8s %(lineno)-4d %(message)s")  # output format
sh = logging.StreamHandler(stream=sys.stdout)  # output to standard output
sh.setFormatter(format)
logger.addHandler(sh)