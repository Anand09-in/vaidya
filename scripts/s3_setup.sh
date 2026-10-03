#!/usr/bin/env bash
# One-time S3 bucket + IAM setup for Vaidya artifact store.
# Run locally before starting Phase 1.

set -euo pipefail

BUCKET="vaidya-artifacts"
REGION="ap-south-1"
IAM_USER="vaidya-s3-user"
POLICY_NAME="VaidyaS3Policy"

echo "=== Creating S3 bucket ==="
aws s3 mb "s3://${BUCKET}" --region "${REGION}"

echo "=== Enabling versioning (protects against accidental overwrites) ==="
aws s3api put-bucket-versioning \
    --bucket "${BUCKET}" \
    --versioning-configuration Status=Enabled

echo "=== Creating IAM user for Kaggle secrets ==="
aws iam create-user --user-name "${IAM_USER}"

echo "=== Attaching S3-only policy (least privilege) ==="
POLICY_DOC=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket",
        "s3:GetBucketLocation"
      ],
      "Resource": [
        "arn:aws:s3:::${BUCKET}",
        "arn:aws:s3:::${BUCKET}/*"
      ]
    }
  ]
}
EOF
)

aws iam put-user-policy \
    --user-name "${IAM_USER}" \
    --policy-name "${POLICY_NAME}" \
    --policy-document "${POLICY_DOC}"

echo "=== Creating access key (save these — shown only once!) ==="
aws iam create-access-key --user-name "${IAM_USER}"

echo ""
echo "=== Done. Add the access key + secret to Kaggle secrets as: ==="
echo "  AWS_ACCESS_KEY_ID"
echo "  AWS_SECRET_ACCESS_KEY"
echo "  AWS_DEFAULT_REGION  (value: ${REGION})"
echo ""
echo "=== Bucket structure will be: ==="
echo "  s3://${BUCKET}/data/          <- Parquet splits"
echo "  s3://${BUCKET}/checkpoints/   <- LoRA adapters"
echo "  s3://${BUCKET}/models/        <- merged / gptq / awq / gguf"
echo "  s3://${BUCKET}/mlflow/        <- MLflow artifact root"
echo "  s3://${BUCKET}/eval/          <- eval_results.json, profiler trace"
