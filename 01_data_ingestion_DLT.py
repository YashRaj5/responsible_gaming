# Databricks notebook source
from pyspark.sql.functions import col, count, countDistinct, min, mean, max, round, sum, lit
import dlt

# COMMAND ----------

schema = 'customer_id STRING, age_band STRING, gender STRING, date STRING, date_transaction_id INT, event_type STRING, game_type STRING, wager_amount FLOAT, win_loss STRING, win_loss_amount FLOAT, initial_balance FLOAT, ending_balance FLOAT, withdrawal_amount FLOAT, deposit_amount FLOAT'

# COMMAND ----------

# DBTITLE 1,bronze clickstream
@dlt.table
def bronze_clickstream():
  raw_data_path = f's3a://db-gtm-industry-solutions/data/CME/real_money_gaming/data/raw/*'
  return spark.read.csv(raw_data_path,schema=schema)

# COMMAND ----------

# MAGIC %md
# MAGIC We are creating silver table for each beacon type, viz.:
# MAGIC - Bets
# MAGIC - Deposits
# MAGIC - Flagged High Risk
# MAGIC - Registrations
# MAGIC - Withdrawals

# COMMAND ----------

# DBTITLE 1,for bets
@dlt.table
def silver_bets():
  return (dlt.read("bronze_clickstream").select('customer_id', 'date', 'date_transaction_id',
          'event_type','game_type','wager_amount','win_loss','win_loss_amount','initial_balance','ending_balance')
          .filter(col('event_type') == 'bet'))

# COMMAND ----------

# DBTITLE 1,for deposits
@dlt.table
def silver_deposits():
  return (dlt.read("bronze_clickstream").select('customer_id', 'date', 'date_transaction_id','event_type','initial_balance','ending_balance','deposit_amount')
         .filter(col('event_type') == 'deposit'))

# COMMAND ----------

# DBTITLE 1,for flagged high risk
@dlt.table
def silver_flagged_high_risk():
  return (dlt.read("bronze_clickstream").select('customer_id', 'date','event_type')
         .filter(col('event_type') == 'flagged_high_risk'))

# COMMAND ----------

# DBTITLE 1,for registrations
@dlt.table
def silver_registrations():
  return (dlt.read("bronze_clickstream").select('customer_id', 'date','event_type','gender','age_band')
         .filter(col('event_type') == 'register'))

# COMMAND ----------

# DBTITLE 1,for withdrawals
@dlt.table
def silver_withdrawals():
  return (dlt.read("bronze_clickstream").select('customer_id', 'date','date_transaction_id', 'event_type','initial_balance','ending_balance','withdrawal_amount')
         .filter(col('event_type') == 'withdrawal'))

# COMMAND ----------

# MAGIC %md
# MAGIC Combining all the table data into a single daily activity gold table

# COMMAND ----------

# DBTITLE 0,tt
@dlt.table
def gold_daily_activity():
  daily_betting_activity = (dlt.read('silver_bets').groupBy('customer_id','date')
                            .agg(count('date_transaction_id').alias('num_bets'),
                                sum('wager_amount').alias('total_wagered'),
                                min('wager_amount').alias('min_wager'),
                                max('wager_amount').alias('max_wager'),
                                round(mean('wager_amount'),2).alias('mean_wager'),
                                round(sum('win_loss_amount'),2).alias('winnings_losses')))
 
  daily_deposits = (dlt.read('silver_deposits').groupBy('customer_id','date')
                    .agg(count('event_type').alias('num_deposits'), sum('deposit_amount').alias('total_deposit_amt')))
 
  daily_withdrawals = (dlt.read('silver_withdrawals').groupBy('customer_id','date')
                       .agg(count('event_type').alias('num_withdrawals'), sum('withdrawal_amount').alias('total_withdrawal_amt')))
  
  
  daily_high_risk_flags = (dlt.read('silver_flagged_high_risk').withColumn('is_high_risk',lit(1)).drop('event_type'))
 
  return (daily_betting_activity.join(daily_deposits,on=['customer_id','date'],how='outer')
          .join(daily_withdrawals,on=['customer_id','date'],how='outer').join(daily_high_risk_flags,on=['customer_id', 'date'],how='outer').na.fill(0))
