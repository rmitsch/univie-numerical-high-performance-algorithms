import logging


def create_logger(name: str):
    """
    Initializes named logger.
    :param name:
    :return:
    """
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s : %(levelname)s : %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    return logging.getLogger(name)


def init_result_dict() -> dict:
    """
    Returns result dictionary as used in test functions.
    :return:
    """

    return {
        "m": [],
        "n": [],
        "mn": [],
        "time_own": [],
        "time_own_blocked": [],
        "time_scipy_update": [],
        "time_numpy_scratch": [],
        "res_norm_QR": [],
        "res_norm_Axb": []
    }