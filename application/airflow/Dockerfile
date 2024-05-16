FROM apache/airflow:2.9.1
WORKDIR /opt/airflow
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY dags /opt/airflow/dags
EXPOSE 8080