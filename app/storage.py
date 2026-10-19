
"""Работа с объектным хранилищем RustFS через S3 API."""

import time

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from app.config import settings


S3_CONFIG = Config(
    signature_version="s3v4",
    s3={"addressing_style": "path"},
    retries={"max_attempts": 3, "mode": "standard"},
)


def get_s3_client(public: bool = False):
    endpoint = (
        settings.s3_public_endpoint_url
        if public
        else settings.s3_endpoint_url
    )

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        region_name=settings.aws_region,
        config=S3_CONFIG,
    )


def wait_for_s3(attempts: int = 30, pause: int = 2):
    for attempt in range(attempts):
        try:
            get_s3_client().list_buckets()
            return
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(pause)


def ensure_bucket():
    client = get_s3_client()

    try:
        client.head_bucket(Bucket=settings.s3_bucket)
    except ClientError as error:
        code = error.response.get(
            "Error", {}
        ).get("Code", "")

        if code not in {"404", "NoSuchBucket"}:
            raise

        client.create_bucket(Bucket=settings.s3_bucket)


def upload_file(path: str, key: str):
    client = get_s3_client()
    client.upload_file(
        path,
        settings.s3_bucket,
        key,
    )


def delete_file(key: str):
    client = get_s3_client()
    client.delete_object(
        Bucket=settings.s3_bucket,
        Key=key,
    )


def presigned_url(
    key: str,
    expires_in: int = 3600,
) -> str:
    client = get_s3_client(public=True)

    return client.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": settings.s3_bucket,
            "Key": key,
        },
        ExpiresIn=expires_in,
    )
