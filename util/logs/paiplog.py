import sys, os, functools
from inspect import getframeinfo, stack

import logging
import os

## set log levle
LOGLEVEL = logging.ERROR  # NOTSET | DEBUG | INFO | WARNING | ERROR | CRITICAL = 0,10,20,30,40,50
HOME_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir) + '/'

class PaipLogFormatter(logging.Formatter):
    def format(self, record):
        if hasattr(record, 'func_name_override'):
            record.funcName = record.func_name_override
        if hasattr(record, 'file_name_override'):
            record.filename = record.file_name_override
        return super(PaipLogFormatter, self).format(record)


def get_logger(log_file_name, log_dir_param=None, log_sub_dir=""):
    """ Creates a Log File and returns Logger object """

    windows_log_dir = 'c:\\logs_dir\\'
    linux_log_dir = HOME_PATH + '../logs_dir/'

    # Build Log file directory, based on the OS and supplied input
    if log_dir_param is not None:
        log_dir = log_dir_param
    else:
        log_dir = windows_log_dir if os.name == 'nt' else linux_log_dir
        log_dir = os.path.join(log_dir, log_sub_dir)

    # Create Log file directory if not exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Build Log File Full Path
    logPath = log_file_name if os.path.exists(log_file_name) else os.path.join(log_dir, (str(log_file_name) + '.log'))

    # Create logger object and set the format for logging and other attributes
    logger = logging.Logger(log_file_name)
    logger.setLevel(LOGLEVEL)
    handler = logging.FileHandler(logPath, 'a+')
    """ Set the formatter of 'PaipLogFormatter' type as we need to log base function name and base file name """
    handler.setFormatter(PaipLogFormatter('%(asctime)s - %(levelname)-10s - %(filename)s - %(funcName)s - %(message)s'))
    logger.addHandler(handler)

    # Return logger object
    return logger


def paiplog(_func=None):
    def log_decorator_info(func):
        py_file_caller = getframeinfo(stack()[2][0])
        # print("====================")
        # print(py_file_caller.function)
        # print("====================")
        if py_file_caller.function != "<module>":
            @functools.wraps(func)
            def log_decorator_wrapper(self, *args, **kwargs):
                # Build logger object
                logger_obj = get_logger(
                    log_file_name='.'.join(os.path.basename(py_file_caller.filename).split('.')[:-1]),
                    log_sub_dir='')  # log_sub_dir='.'.join(py_file_caller.filename.split('.')[:-1]))

                args_passed_in_function = [repr(a) for a in args]
                kwargs_passed_in_function = [f"{k}={v!r}" for k, v in kwargs.items()]

                formatted_arguments = ", ".join(args_passed_in_function + kwargs_passed_in_function)

                extra_args = {'func_name_override': func.__name__,
                              'file_name_override': os.path.basename(py_file_caller.function)}
                logger_obj.info(f"Arguments: {formatted_arguments} - Begin function")
                try:
                    """ log return value from the function """
                    value = func(self, *args, **kwargs)
                    logger_obj.info(f"Returned: - End function {value!r}", extra=extra_args)
                except:
                    """log exception if occurs in function"""
                    logger_obj.error(f"Exception: {str(sys.exc_info()[1])}", extra=extra_args)
                    raise
                # Return function value
                return value

            # Return the pointer to the function
            return log_decorator_wrapper

        else:
            @functools.wraps(func)
            def log_decorator_wrapper(*args, **kwargs):
                # Build logger object
                logger_obj = get_logger(
                    log_file_name='.'.join(os.path.basename(py_file_caller.filename).split('.')[:-1]), log_sub_dir='')

                args_passed_in_function = [repr(a) for a in args]
                kwargs_passed_in_function = [f"{k}={v!r}" for k, v in kwargs.items()]

                formatted_arguments = ", ".join(args_passed_in_function + kwargs_passed_in_function)

                extra_args = {'func_name_override': func.__name__,
                              'file_name_override': os.path.basename(py_file_caller.filename)}
                logger_obj.info(f"Arguments: {formatted_arguments} - Begin function")
                try:
                    """ log return value from the function """
                    value = func(*args, **kwargs)
                    logger_obj.info(f"Returned: - End function {value!r}", extra=extra_args)
                except:
                    """log exception if occurs in function"""
                    logger_obj.error(f"Exception: {str(sys.exc_info()[1])}", extra=extra_args)
                    raise
                # Return function value
                return value

            # Return the pointer to the function
            return log_decorator_wrapper

    # Decorator was called with arguments, so return a decorator function that can read and return a function
    if _func is None:
        return log_decorator_info
    # Decorator was called without arguments, so apply the decorator to the function immediately
    else:
        return log_decorator_info(_func)

