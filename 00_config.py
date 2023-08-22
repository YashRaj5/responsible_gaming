# Databricks notebook source
useremail = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
username_sql_compatible = useremail.split('@')[0].replace('.', '_')

# COMMAND ----------

print("User e-mail: {}".format(useremail))
print("SQL campatible username: {}".format(username_sql_compatible))

# COMMAND ----------

config = {
    "database": "real_money_gaming",
    "experiment_path": f"/Users/{useremail}/real_money_gaming",
    "data_path": f"/tmp/real_money_gaming/data",
    "pipeline_path": f"/tmp/real_money_gaming/dlt",
    "pipeline_name": "real_money_gaming_pipeline"
}

# COMMAND ----------

# DBTITLE 1,Creating source data path
try:
    dbutils.fs.ls(f"dbfs:{config['data_path']}")
except:
    print("Raw data directory not found, creating the directory...")
    dbutils.fs.mkdirs(f"{config['data_path']}/raw")

# COMMAND ----------

# DBTITLE 1,Setting current database
_ = spark.sql(f"create database if not exists {config['database']}")
_ = spark.sql(f"use {config['database']}")

# COMMAND ----------

# DBTITLE 1,Setting Experiment
import mlflow
mlflow.set_experiment(config['experiment_path'])
