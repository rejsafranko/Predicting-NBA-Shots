from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from datetime import datetime, timedelta
from sqlalchemy import create_engine

DB_ENGINE = create_engine("db-connection")


def check_new_data_count() -> int:
    with DB_ENGINE.connect() as connection:
        result = connection.execute(
            """
            SELECT
                COUNT(*)
            FROM db.shots
            WHERE is_new = True;
            """
        )
        new_data_count = result.scalar()
    return new_data_count


def should_retrain(new_data_count: int) -> str:
    return "retrain" if new_data_count >= 100 else "skip"


def train_model():
    def load_data():
        pass
    def load_model():
        pass
    features, labels = load_data()
    model = load_model()
    model.fit(features, labels)


def update_database():
    with DB_ENGINE.connect() as connection:
        connection.execute(
            """
            UPDATE db.shots
            SET is_new = False
            WHERE is_new = True;
            """
        )


DEFAULT_ARGS = {
    "owner": "rejsafranko",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "retrain_model_when_needed",
    default_args=DEFAULT_ARGS,
    description="Retrain the model if enough new data is stored in the database.",
    schedule_interval=timedelta(days=1),
)

t1 = PythonOperator(
    task_id="check_new_data_count",
    python_callable=check_new_data_count,
    provide_context=True,
    dag=dag,
)
t2 = BranchPythonOperator(
    task_id="decide_if_retrain",
    python_callable=lambda **kwargs: should_retrain(
        kwargs["ti"].xcom_pull(task_ids="check_new_data_count")
    ),
)
t3 = PythonOperator(task_id="train_model", python_callable=train_model, dag=dag)
t4 = PythonOperator(task_id="update_database", python_callable=update_database)
t5 = PythonOperator(
    task_id="skip_retraining",
    python_callable=lambda: print("No need for retraining."),
    dag=dag,
)

t1 >> t2 >> [t3, t5]
t3 >> t4
