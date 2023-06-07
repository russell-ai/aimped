# Author: AIMPED
# Date: 2023-March-11
# Description: This file contains utility functions

import json
import logging
import numpy as np
from .version import __version__


def get_version():
    """Returns the version of aimped library."""
    return f'aimped version: {__version__}'

def get_handler(log_file='KSERVE.log', log_level=logging.DEBUG):
    """Returns the logger object for logging the output of the server.
    parameters:
    ----------------
    log_file: str
    log_level: logging level
    return:
    ----------------
    logger: logging object
    """

    f_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')
    formatter = logging.Formatter(
        '[%(asctime)s %(filename)s:%(lineno)s] - %(message)s')
    f_handler.setFormatter(formatter)
    f_handler.setLevel(log_level)

    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.addHandler(f_handler)

    return logger

def cuda_info():
    """Returns the cuda information if cuda is available."""
    try:
        import torch
        if torch.cuda.is_available():
            cuda_info = {"cuda is available": torch.cuda.is_available(),
                         "device count": torch.cuda.device_count(),
                         "current device": torch.cuda.current_device(),
                         "device name": torch.cuda.get_device_name(0),
                         "Memory Usage": {"Allocated": round(torch.cuda.memory_allocated(0)/1024**3,1),
                                          "Cached": round(torch.cuda.memory_reserved(0)/1024**3,1)}}
            return cuda_info
        else:
            return "cuda not available"
    except:
        return "cuda information not available"


class NumpyFloatValuesEncoder(json.JSONEncoder):
    """This class is used to convert numpy float32 to float"""
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


# TODO: test these functions
if __name__ == '__main__':
    print(get_version())