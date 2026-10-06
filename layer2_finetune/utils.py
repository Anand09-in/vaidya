"""Shared utilities for Vaidya training scripts (credentials, S3 upload, MLflow)."""

import os, sys, logging
from pathlib import Path

log = logging.getLogger(__name__)


def setup_credentials():
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        from kaggle_secrets import UserSecretsClient
        s = UserSecretsClient()
        os.environ["AWS_ACCESS_KEY_ID"]     = s.get_secret("AWS_ACCESS_KEY_ID")
        os.environ["AWS_SECRET_ACCESS_KEY"] = s.get_secret("AWS_SECRET_ACCESS_KEY")
        os.environ["AWS_DEFAULT_REGION"]    = s.get_secret("AWS_DEFAULT_REGION")
        log.info("AWS credentials loaded from Kaggle secrets.")
    elif "google.colab" in sys.modules:
        from google.colab import userdata
        os.environ["AWS_ACCESS_KEY_ID"]     = userdata.get("AWS_ACCESS_KEY_ID")
        os.environ["AWS_SECRET_ACCESS_KEY"] = userdata.get("AWS_SECRET_ACCESS_KEY")
        os.environ["AWS_DEFAULT_REGION"]    = userdata.get("AWS_DEFAULT_REGION")
        log.info("AWS credentials loaded from Colab secrets.")
    else:
        os.environ.setdefault("AWS_PROFILE", "vaidya")
        log.info("Using AWS profile: vaidya")


def push_to_s3(run_dir: Path, run_name: str, s3_bucket: str):
    import boto3
    s3 = boto3.client("s3")
    bucket = s3_bucket.replace("s3://", "")

    uploaded = 0
    for f in run_dir.rglob("*"):
        if f.is_file():
            key = f"checkpoints/{run_name}/{f.relative_to(run_dir).as_posix()}"
            s3.upload_file(str(f), bucket, key)
            uploaded += 1
    log.info("Uploaded %d files → %s/checkpoints/%s/", uploaded, s3_bucket, run_name)

    mlruns_dir = Path("mlruns")
    if mlruns_dir.exists():
        for f in mlruns_dir.rglob("*"):
            if f.is_file():
                s3.upload_file(str(f), bucket, f"mlflow/{f.relative_to('.').as_posix()}")
        log.info("MLflow runs → %s/mlflow/", s3_bucket)


def setup_mlflow(experiment_name: str, s3_bucket: str):
    import mlflow
    mlflow.set_tracking_uri("./mlruns")
    try:
        mlflow.create_experiment(experiment_name,
                                 artifact_location=f"{s3_bucket}/mlflow")
    except Exception:
        pass
    mlflow.set_experiment(experiment_name)
    log.info("MLflow → ./mlruns  |  experiment: %s", experiment_name)
