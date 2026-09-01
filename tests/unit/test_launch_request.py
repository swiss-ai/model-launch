from swiss_ai_model_launch.launchers.launch_request import LaunchRequest
from swiss_ai_model_launch.launchers.model_catalog_entry import ModelCatalogEntry


def test_from_catalog_entry_carries_servekit_optims():
    entry = ModelCatalogEntry(model="vendor/model", framework="sglang", servekit_optims=True)
    request = LaunchRequest.from_catalog_entry(entry, replicas=1, time="02:00:00")
    assert request.servekit_optims is True


def test_from_catalog_entry_defaults_servekit_optims_to_false():
    entry = ModelCatalogEntry(model="vendor/model", framework="sglang")
    request = LaunchRequest.from_catalog_entry(entry, replicas=1, time="02:00:00")
    assert request.servekit_optims is False


def test_from_catalog_entry_carries_servekit_artifact_path():
    entry = ModelCatalogEntry(
        model="vendor/model",
        framework="sglang",
        servekit_optims=True,
        servekit_artifact_path="/scratch/artifact",
    )
    request = LaunchRequest.from_catalog_entry(entry, replicas=1, time="02:00:00")
    assert request.servekit_artifact_path == "/scratch/artifact"


def test_from_catalog_entry_defaults_servekit_artifact_path_to_none():
    entry = ModelCatalogEntry(model="vendor/model", framework="sglang")
    request = LaunchRequest.from_catalog_entry(entry, replicas=1, time="02:00:00")
    assert request.servekit_artifact_path is None
