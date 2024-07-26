import torch
import logging
import random
import time
from datetime import datetime
import wandb
import inspect

from google.cloud import storage

def retry_with_backoff(fn, retries=5, backoff_in_seconds=1, **kwargs):
    """Wrapper for a function to run with exponential backoff."""
    retries_left = retries
    while True:
        try:
            return fn(**kwargs)
        except Exception:
            if retries_left == 0:
                raise
            else:
                sleep = (backoff_in_seconds * 2 ** (retries + 1 - retries_left) +
                         random.uniform(0, 1))
                time.sleep(sleep)
                retries_left -= 1


ARTIFACT_BUCKET = "ml-infra-test-model-artifacts"

def save_checkpoint_to_gcs(model, optimizer, filename):
    """save out the optimizer and model state dicts
    to a checkpoint file.   we can use the optimizer if
    we wish to resume.   the model state file paired with
    the model architecture can be used to operationalize the model
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(ARTIFACT_BUCKET)
    blob = bucket.blob(filename)
    with blob.open("wb", ignore_flush=True) as f:
        torch.save(
            {
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
            }, f)


def checkpoint(model, optimizer, gcs_path) -> str:
    """
    save out the optimizer and model state dicts
    to a checkpoint file.   we can use the optimizer if
    we wish to resume.   the model state file paired with
    the model architecture can be used to operationalize the model
    """
    filename = f"{gcs_path}/model-checkpoint.pt"
    retry_with_backoff(
        save_checkpoint_to_gcs,
        model=model,
        optimizer=optimizer,
        filename=filename
    )
    return f"gs://{ARTIFACT_BUCKET}/{filename}"


def resume(model, filename, optimizer):
    """If a long training were interrupted, we could use the resume to continue training"""
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    
#### generate artifacts
def generate_gcs_filepath(wandb_run, model_name) -> str:
    """pull apart a wandb.run object to generate a unique
    filepath in GCS we can use for the run"""

    date_partition = datetime.utcfromtimestamp(wandb_run.start_time).strftime('%Y-%m-%d')
    filepath = f"{model_name}/{date_partition}/{wandb_run.name}-{wandb_run.id}"
    return filepath


def generate_model_gcs_uri_artifact(model_name: str, wandb_run: wandb.run, artifact_gcs_path: str):
    """generate a wandb artifact for the model gcs uri"""
    root_gcs_path = f"gs://{ARTIFACT_BUCKET}/{artifact_gcs_path}"
    artifact = wandb.Artifact(name=model_name, type="model")
    artifact.add_reference(uri=root_gcs_path)
    wandb_run.log_artifact(artifact)


def get_module_path(model_class):
    """get the path to the module file for a model class"""
    try:
        m = inspect.getmodule(model_class)
    except AttributeError:
        logger.info("--Model class does not have a module")
        raise AttributeError
    return m.__file__


def save_filename_to_gcs(gcs_uri: str, module_path: str) -> None:
    """save a filename to GCS"""
    storage_client = storage.Client()
    blob = storage_client.bucket(ARTIFACT_BUCKET).blob(gcs_uri)
    blob.upload_from_filename(module_path)


def send_model_artifact_to_gcs(model_class, artifact_gcs_filepath):
    """send a module to GCS as an artifact"""
    module_path = get_module_path(model_class)
    gcs_uri = f"{artifact_gcs_filepath}/model.py"

    retry_with_backoff(
        save_filename_to_gcs,
        gcs_uri=gcs_uri,
        module_path=module_path
    )

    
    
    
