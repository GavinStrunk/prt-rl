import pytest
import torch
import numpy as np
import tempfile
from prt_rl.common.loggers import Logger, BlankLogger, FileLogger

def test_create_blank_logger():
    # Test Logger
    logger = Logger.create("blank")
    assert isinstance(logger, BlankLogger)

def test_create_invalid_logger():
    # Test invalid logger type
    with pytest.raises(ValueError):
        Logger.create("invalid_logger")


def test_file_logger_json_serialization_error():
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = FileLogger(output_dir=tmpdir)

        # Log scalar values that are not JSON serializable
        logger.log_scalar("float32_numpy", np.float32(3.14), iteration=0)
        logger.log_scalar("float32_tensor", torch.tensor(2.71, dtype=torch.float32), iteration=0)

        logger.close()

    assert True
