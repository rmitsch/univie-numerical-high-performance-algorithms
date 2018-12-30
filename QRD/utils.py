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