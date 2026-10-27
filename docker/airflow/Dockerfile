# Airflow image with an ISOLATED virtualenv for ML code.
#
# Production lesson: Airflow's own dependencies (SQLAlchemy, pydantic, ...) are pinned
# by Airflow's constraints files. ML libraries pin their own versions. Installing both
# into one environment is a classic source of "works on my laptop" pain. We keep the
# scheduler's Python clean and run every ML step with /opt/ml-venv/bin/python instead —
# the same idea as KubernetesPodOperator / DockerOperator in a real cluster.
FROM apache/airflow:3.3.2-python3.12

USER root
COPY --from=ghcr.io/astral-sh/uv:0.9.28 /uv /usr/local/bin/uv
COPY requirements-ml.txt /tmp/requirements-ml.txt
RUN uv venv /opt/ml-venv --python /usr/python/bin/python3.12 \
 && uv pip install --python /opt/ml-venv/bin/python --no-cache -r /tmp/requirements-ml.txt \
 && chown -R airflow:0 /opt/ml-venv
USER airflow
