import collections
import math

from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from sklearn.ensemble import IsolationForest
import numpy as np
import os

SPARK_HOME = "/opt/cloudera/parcels/CDH-6.3.0-1.cdh6.3.0.p0.1279813/lib/spark/"
PYSPARK_PYTHON = "/usr/bin/python"
PYSPARK_DRIVER_PYTHON = "/usr/bin/python"
os.environ['SPARK_HOME'] = SPARK_HOME
os.environ['PYSPARK_PYTHON'] = PYSPARK_PYTHON
os.environ['PYSPARK_DRIVER_PYTHON'] = PYSPARK_DRIVER_PYTHON

spark = SparkSession.builder.appName("test").getOrCreate()
m_df = spark.read.json("/data/malicious_data.log")
if __name__ == '__main__':

    def replace_null(dataframe):
        null = u'\u0000'
        dataframe = dataframe.withColumn("ccmdline", f.regexp_replace(dataframe.ccmdline, null, ' '))
        return dataframe

    get_param_num_udf = f.udf(lambda ccmdline : len(ccmdline.split()))

    def get_entropy(string):
        counter_char = collections.Counter(string)
        entropy = 0
        for c, ctn in counter_char.items():
            _p = float(ctn) / len(string)
            entropy += -1 * _p * math.log(_p, 2)
        return round(entropy, 7)

    get_entropy_udf = f.udf(lambda ccmdline : get_entropy(ccmdline))

    def get_sensitive_characters_num_udf(characters):
        return f.udf(lambda ccmdline : ccmdline.count(characters))

    def extract_features(dataframe):
        dataframe = dataframe.where("pname = 'java' and ccmdline != ''")

        dataframe = dataframe.withColumn("len", f.length("ccmdline") - 1)
        dataframe = dataframe.withColumn("paramnum", get_param_num_udf(dataframe.ccmdline))
        dataframe = dataframe.withColumn("entropy", get_entropy_udf(dataframe.ccmdline))
        dataframe = dataframe.withColumn("\'\' num", get_sensitive_characters_num_udf("\'")(dataframe.ccmdline) / 2)
        dataframe = dataframe.withColumn("\"\" num", get_sensitive_characters_num_udf("\"")(dataframe.ccmdline) / 2)
        dataframe = dataframe.withColumn("> num", get_sensitive_characters_num_udf(">")(dataframe.ccmdline))
        dataframe = dataframe.withColumn("; num", get_sensitive_characters_num_udf(";")(dataframe.ccmdline))
        dataframe = dataframe.withColumn("$ num", get_sensitive_characters_num_udf("$")(dataframe.ccmdline))
        dataframe = dataframe.withColumn("% num", get_sensitive_characters_num_udf("%")(dataframe.ccmdline))
        dataframe = dataframe.withColumn("| num", get_sensitive_characters_num_udf("|")(dataframe.ccmdline))
        dataframe = dataframe.withColumn("() num", get_sensitive_characters_num_udf("(")(dataframe.ccmdline))
        dataframe = dataframe.withColumn("{} num", get_sensitive_characters_num_udf("{")(dataframe.ccmdline))
        dataframe = dataframe.withColumn("-D num", get_sensitive_characters_num_udf("-D")(dataframe.ccmdline))
        dataframe = dataframe.withColumn("yum num", get_sensitive_characters_num_udf(" yum ")(dataframe.ccmdline))
        dataframe = dataframe.withColumn("echo num", get_sensitive_characters_num_udf(" echo ")(dataframe.ccmdline))
        dataframe = dataframe.withColumn("cat num", get_sensitive_characters_num_udf(" cat ")(dataframe.ccmdline))
        dataframe = dataframe.withColumn("ls num", get_sensitive_characters_num_udf(" ls ")(dataframe.ccmdline))
        dataframe = dataframe.withColumn("sudo num", get_sensitive_characters_num_udf(" sudo ")(dataframe.ccmdline))
        dataframe = dataframe.withColumn("chmod num", get_sensitive_characters_num_udf(" chmod ")(dataframe.ccmdline))
        dataframe = dataframe.withColumn("etc num", get_sensitive_characters_num_udf("etc")(dataframe.ccmdline))
        dataframe = dataframe.withColumn("passwd num", get_sensitive_characters_num_udf("passwd")(dataframe.ccmdline))
        dataframe = dataframe.withColumn(".sh num", get_sensitive_characters_num_udf(".sh")(dataframe.ccmdline))
        # dataframe.show(200)
        return dataframe

    def get_input_array(dataframe):
        dataframe = replace_null(dataframe)
        dataframe = extract_features(dataframe)
        dataframe = dataframe.drop("cexe","cname","cpid","pcmdline","pexe","pname","ppid")
        dataframe.show()

        array = np.array(dataframe.collect())
        return array

    X_test = get_input_array(m_df)
    clf = IsolationForest(n_estimators=200,contamination=0.3757,random_state=0,n_jobs=-1).fit(X_test[:,2:])
    y_pred_test = clf.predict(X_test[:,2:])
    abnormal = X_test[y_pred_test == -1]
    for json in abnormal:
        print json[1]
    print len(abnormal)