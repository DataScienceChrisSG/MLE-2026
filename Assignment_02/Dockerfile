# Airflow 2.10.5 on Python 3.12 (same Python as Assignment 1), with Java for PySpark
FROM apache/airflow:2.10.5-python3.12

USER root
ENV DEBIAN_FRONTEND=noninteractive

# Java 17 (headless) for Spark, procps for 'ps' (used by Spark scripts)
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jre-headless procps && \
    rm -rf /var/lib/apt/lists/* && \
    # architecture-independent JAVA_HOME (works on Intel and Apple-silicon hosts)
    ln -s "$(dirname "$(dirname "$(readlink -f "$(which java)")")")" /usr/lib/jvm/java-home

ENV JAVA_HOME=/usr/lib/jvm/java-home
ENV PATH=$PATH:$JAVA_HOME/bin

USER airflow

# Pinned to the versions in Airflow 2.10.5's own constraints; apache-airflow is pinned too so pip cannot change it
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /requirements.txt
