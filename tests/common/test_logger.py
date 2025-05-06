import pytest
from prt_rl.common.loggers import Logger, BlankLogger

def test_create_blank_logger():
    # Test Logger
    logger = Logger.create("blank")
    assert isinstance(logger, BlankLogger)

def test_create_invalid_logger():
    # Test invalid logger type
    with pytest.raises(ValueError):
        Logger.create("invalid_logger")