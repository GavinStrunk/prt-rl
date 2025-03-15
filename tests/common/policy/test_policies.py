import pytest
from prt_rl.common.policy import load_from_mlflow


@pytest.mark.skip(reason="Requires MLFlow server")
def test_mlflow_model_load():
    tracking_uri = 'http://localhost:5000'
    model_name = 'Robot Game'
    version = '2'

    policy = load_from_mlflow(tracking_uri=tracking_uri, model_name=model_name, model_version=version)
