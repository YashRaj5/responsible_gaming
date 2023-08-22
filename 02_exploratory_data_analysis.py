# Databricks notebook source
# MAGIC %run ./00_config

# COMMAND ----------

# DBTITLE 1,bronze_clickstreams
# MAGIC %sql
# MAGIC select * from hive_metastore.real_money_gaming.bronze_clickstream;

# COMMAND ----------

# DBTITLE 1,silver bets table
# MAGIC %sql
# MAGIC select * from hive_metastore.real_money_gaming.silver_bets;

# COMMAND ----------

# DBTITLE 1,gold activity table
# MAGIC %sql
# MAGIC select * from hive_metastore.real_money_gaming.gold_daily_activity;

# COMMAND ----------

# MAGIC %sql
# MAGIC select
# MAGIC   count(customer_id)
# MAGIC from hive_metastore.real_money_gaming.gold_daily_activity
# MAGIC where is_high_risk = 1;

# COMMAND ----------

# DBTITLE 1,average number of bets per day
# MAGIC %sql
# MAGIC select
# MAGIC   month(date) as month,
# MAGIC   avg(num_bets)
# MAGIC from hive_metastore.real_money_gaming.gold_daily_activity
# MAGIC group by month
# MAGIC order by month

# COMMAND ----------

# DBTITLE 1,total money spent by users
# MAGIC %sql
# MAGIC

# COMMAND ----------


