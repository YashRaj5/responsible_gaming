# Databricks notebook source
# MAGIC
# MAGIC %md This notebook sets up the companion cluster(s) to run the solution accelerator. It also creates the Workflow to illustrate the order of execution. Happy exploring! 
# MAGIC 🎉
# MAGIC
# MAGIC **Steps**
# MAGIC 1. Simply attach this notebook to a cluster and hit Run-All for this notebook. A multi-step job and the clusters used in the job will be created for you and hyperlinks are printed on the last block of the notebook. 
# MAGIC
# MAGIC 2. Run the accelerator notebooks: Feel free to explore the multi-step job page and **run the Workflow**, or **run the notebooks interactively** with the cluster to see how this solution accelerator executes. 
# MAGIC
# MAGIC     2a. **Run the Workflow**: Navigate to the Workflow link and hit the `Run Now` 💥. 
# MAGIC   
# MAGIC     2b. **Run the notebooks interactively**: Attach the notebook with the cluster(s) created and execute as described in the `job_json['tasks']` below.
# MAGIC
# MAGIC **Prerequisites** 
# MAGIC 1. You need to have cluster creation permissions in this workspace.
# MAGIC
# MAGIC 2. In case the environment has cluster-policies that interfere with automated deployment, you may need to manually create the cluster in accordance with the workspace cluster policy. The `job_json` definition below still provides valuable information about the configuration these series of notebooks should run with. 
# MAGIC
# MAGIC **Notes**
# MAGIC 1. The pipelines, workflows and clusters created in this script are not user-specific. Keep in mind that rerunning this script again after modification resets them for other users too.
# MAGIC
# MAGIC 2. If the job execution fails, please confirm that you have set up other environment dependencies as specified in the accelerator notebooks. Accelerators may require the user to set up additional cloud infra or secrets to manage credentials. 

# COMMAND ----------

# DBTITLE 1,Install util packages
# MAGIC %pip install mlflow git+https://github.com/databricks-academy/dbacademy@v1.0.13 git+https://github.com/databricks-industry-solutions/notebook-solution-companion@safe-print-html --quiet --disable-pip-version-check

# COMMAND ----------

from solacc.companion import NotebookSolutionCompanion

# COMMAND ----------

# MAGIC %run "./_resources/notebook_config"

# COMMAND ----------

pipeline_json = {
    "clusters": [
        {
            "label": "default",
            "autoscale": {
                "min_workers": 1,
                "max_workers": 5,
                "mode": "LEGACY"
            }
        }
    ],
    "development": True,
    "continuous": False,
    "channel": "CURRENT",
    "edition": "ADVANCED",
    "photon": False,
    "libraries": [
        {
            "notebook": {
                "path": "01_DLT"
            }
        }
    ],
    "name": config["pipeline_name"],
    "configuration": {
      "data.path": config["data_path"]
    },
    "storage": config["pipeline_path"],
    "target": config["database"]
}

# COMMAND ----------

solacc_config_database = "databricks_solacc"
dbsql_config_table = f"{solacc_config_database}.dbsql"

spark.sql(f"CREATE DATABASE IF NOT EXISTS {solacc_config_database} LOCATION '/databricks_solacc/'")
spark.sql(f"CREATE TABLE IF NOT EXISTS {dbsql_config_table} (path STRING, id STRING, solacc STRING)")

# COMMAND ----------

pipeline_id = NotebookSolutionCompanion().deploy_pipeline(pipeline_json, "", spark)

# COMMAND ----------

job_json = {
        "timeout_seconds": 28800,
        "max_concurrent_runs": 1,
        "tags": {
            "usage": "solacc_testing",
            "group": "CME"
        },
        "tasks": [
            {
                "pipeline_task": {
                    "pipeline_id": pipeline_id
                },
                "task_key": "rmg_01",
                "description": ""
            },
          {
                "job_cluster_key": "rmg_cluster",
                "libraries": [],
                "notebook_task": {
                    "notebook_path": f"02_EDA"
                },
                "task_key": "rmg_02",
                "description": "",
                "depends_on": [
                    {
                        "task_key": "rmg_01"
                    }
                ]
            },
          {
                "job_cluster_key": "rmg_cluster",
                "libraries": [],
                "notebook_task": {
                    "notebook_path": f"03_Feature_Store"
                },
                "task_key": "rmg_03",
                "description": "",
                "depends_on": [
                    {
                        "task_key": "rmg_02"
                    }
                ]
            },
          {
                "job_cluster_key": "rmg_cluster",
                "libraries": [],
                "notebook_task": {
                    "notebook_path": f"04_ML"
                },
                "task_key": "rmg_04",
                "description": "",
                "depends_on": [
                    {
                        "task_key": "rmg_03"
                    }
                ]
            }
        ],
        "job_clusters": [
            {
                "job_cluster_key": "rmg_cluster",
                "new_cluster": {
                    "spark_version": "11.3.x-cpu-ml-scala2.12",
                "spark_conf": {
                    "spark.databricks.delta.formatCheck.enabled": "false"
                    },
                    "num_workers": 2,
                    "node_type_id": {"AWS": "i3.xlarge", "MSA": "Standard_DS3_v2", "GCP": "n1-highmem-4"},
                    "custom_tags": {
                        "usage": "solacc_testing"
                    },
                }
            }
        ]
    }

# COMMAND ----------

dbutils.widgets.dropdown("run_job", "False", ["True", "False"])
run_job = dbutils.widgets.get("run_job") == "True"
nsc = NotebookSolutionCompanion()
nsc.deploy_compute(job_json, run_job=run_job)
nsc.deploy_dbsql("./Responsible_Gaming_Dashboard.dbdash", dbsql_config_table, spark)
