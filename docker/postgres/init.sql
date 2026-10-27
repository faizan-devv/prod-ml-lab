-- One Postgres instance, three logical databases.
-- In production these would usually be separate managed instances (RDS / Cloud SQL).
CREATE DATABASE mlflow;   -- MLflow runs, params, metrics, model registry
CREATE DATABASE airflow;  -- Airflow metadata: DAG runs, task instances, XComs
CREATE DATABASE feast;    -- Feast registry: entities, feature views, feature services
