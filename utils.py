
import logging

import numpy as np
from skimage.color import gray2rgb
# import matplotlib.pyplot as plt
from skimage.exposure import equalize_hist

# Start logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.WARNING)


def apply_color_lut(image, color: str, do_equalize_hist=True):
    multiplier = {'red': [1, 0, 0], 'green': [0, 1, 0], 'blue': [0, 0, 1]}
    if do_equalize_hist:
        image = (equalize_hist(image) * 200).astype(np.uint8)
    return gray2rgb(image) * multiplier[color]


def to_color(image, color: str, do_equalize_hist=True):
    multiplier = {'red': [1, 0, 0], 'green': [0, 1, 0], 'blue': [0, 0, 1]}
    if do_equalize_hist:
        image = (equalize_hist(image) * 200).astype(np.uint8)
    return gray2rgb(image) * multiplier[color]


# # https://github.com/CellProfiler/python-bioformats/issues/137
# def init_javabridge_logger(level: str = "WARN"):
#     """This is so that Javabridge doesn't spill out a lot of DEBUG messages
#     during runtime.
#     From CellProfiler/python-bioformats.
#     """
#     rootLoggerName = javabridge.get_static_field("org/slf4j/Logger",
#                                                  "ROOT_LOGGER_NAME",
#                                                  "Ljava/lang/String;")
#     rootLogger = javabridge.static_call("org/slf4j/LoggerFactory",
#                                         "getLogger",
#                                         "(Ljava/lang/String;)Lorg/slf4j/Logger;",
#                                         rootLoggerName)
#     logLevel = javabridge.get_static_field("ch/qos/logback/classic/Level",
#                                            level,
#                                            "Lch/qos/logback/classic/Level;")
#     javabridge.call(rootLogger,
#                     "setLevel",
#                     "(Lch/qos/logback/classic/Level;)V",
#                     logLevel)


def main():
    pass


if __name__ == "__main__":
    main()
