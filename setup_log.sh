#!/bin/bash

# Set up AIM
aim_repo_path=logs
mlflow_uri=logs/mlflow

aimlflow sync --mlflow-tracking-uri=$mlflow_uri --aim-repo=$aim_repo_path &
aim up --repo=$aim_repo_path &
