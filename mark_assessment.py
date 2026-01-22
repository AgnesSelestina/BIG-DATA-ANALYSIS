# Install PySpark
!pip install pyspark
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("BigDataAnalysis_CODTECH") \
    .getOrCreate()

print("✅ Spark Session Started Successfully")
# Load dataset using PySpark
df = spark.read.csv("/content/smart_lms_dataset_correlated.csv",
                    header=True, inferSchema=True)

print("✅ Dataset Loaded Successfully")
df.show(20)
print("📐 Total Rows:", df.count())
print("📊 Total Columns:", len(df.columns))

print("\n📌 Schema:")
df.printSchema()
from pyspark.sql.functions import col

print("🔎 Missing Values Count Per Column:")
for c in df.columns:
    print(c, ":", df.filter(col(c).isNull()).count())
from pyspark.sql.functions import when

# Drop rows missing more than 3 columns
df = df.dropna(thresh=len(df.columns)-3)

# Fill numeric columns with median (approximate for Big Data)
numeric_cols = ["attendance_rate", "time_spent_hours", "quiz_avg_score",
                "final_exam_score", "forum_posts", "badges_earned"]

for col_name in numeric_cols:
    median = df.approxQuantile(col_name, [0.5], 0.01)[0]
    df = df.fillna({col_name: median})

# Fill categorical with mode
mode_grade = df.groupBy("final_grade").count().orderBy("count", ascending=False).first()[0]
df = df.fillna({"final_grade": mode_grade})

print("✅ Missing Values Handled")
#remove duplicate records
before = df.count()
df = df.dropDuplicates()
after = df.count()

print("Rows before removing duplicates:", before)
print("Rows after removing duplicates :", after)
print("✅ Removed", before - after, "duplicate rows")
#Handle Inconsistent Data Ranges
from pyspark.sql.functions import when

df = df.withColumn("attendance_rate", when(col("attendance_rate") < 0, 0)
                                     .when(col("attendance_rate") > 1, 1)
                                     .otherwise(col("attendance_rate")))

df = df.withColumn("quiz_avg_score", when(col("quiz_avg_score") < 0, 0)
                                     .when(col("quiz_avg_score") > 1, 1)
                                     .otherwise(col("quiz_avg_score")))

df = df.withColumn("time_spent_hours", when(col("time_spent_hours") < 0, 0)
                                       .otherwise(col("time_spent_hours")))

print("✅ Inconsistent Numeric Ranges Fixed")
#Big Data Aggregations & Insights


# Grade Distribution
print("📊 Final Grade Distribution:")
df.groupBy("final_grade").count().show()

# Average Exam Score by Grade
print("📌 Average Final Exam Score by Grade:")
df.groupBy("final_grade").avg("final_exam_score").show()

# Average Attendance by Grade
print("📌 Average Attendance Rate by Grade:")
df.groupBy("final_grade").avg("attendance_rate").show()

# Top Engaged Students
print("🔥 Top 10 Most Engaged Students (by Time Spent):")
df.orderBy(col("time_spent_hours").desc()).show(10)

# Correlation Check (Big Data Style)
print("📈 Correlation between Time Spent and Exam Score:")
df.stat.corr("time_spent_hours", "final_exam_score")
# Convert Spark → Pandas for Visualization
import pandas as pd

grade_avg = df.groupBy("final_grade").avg("final_exam_score").toPandas()
attendance_avg = df.groupBy("final_grade").avg("attendance_rate").toPandas()
# DATA VISUALIZATION SECTION
# Convert Spark DataFrames to Pandas for visualization

grade_dist = df.groupBy("final_grade").count().toPandas()
grade_exam_avg = df.groupBy("final_grade").avg("final_exam_score").toPandas()
grade_att_avg = df.groupBy("final_grade").avg("attendance_rate").toPandas()

# Take a random sample for histograms & scatter (to avoid memory issue)
sample_df = df.sample(fraction=0.3, seed=42).toPandas()

# PIE CHART – Final Grade Distribution
import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))
plt.pie(grade_dist["count"], labels=grade_dist["final_grade"],
        autopct="%1.1f%%", startangle=90, shadow=True)

plt.title("📊 Final Grade Distribution (Pie Chart)")
plt.show()

# BAR GRAPH – Average Exam Score by Grade
import seaborn as sns

plt.figure(figsize=(7,5))
sns.barplot(x="final_grade", y="avg(final_exam_score)", data=grade_exam_avg)

plt.title("📌 Average Final Exam Score by Grade")
plt.xlabel("Final Grade")
plt.ylabel("Average Exam Score")
plt.show()

# BAR GRAPH – Average Attendance Rate by Grade
plt.figure(figsize=(7,5))
sns.barplot(x="final_grade", y="avg(attendance_rate)", data=grade_att_avg)

plt.title("📌 Average Attendance Rate by Grade")
plt.xlabel("Final Grade")
plt.ylabel("Attendance Rate")
plt.show()

#HISTOGRAM – Distribution of Final Exam Scores
plt.figure(figsize=(7,5))
plt.hist(sample_df["final_exam_score"], bins=20, color="skyblue", edgecolor="black")

plt.title("📊 Distribution of Final Exam Scores")
plt.xlabel("Final Exam Score")
plt.ylabel("Frequency")
plt.show()

# HISTOGRAM – Distribution of Time Spent on Platform
plt.figure(figsize=(7,5))
plt.hist(sample_df["time_spent_hours"], bins=20, color="orange", edgecolor="black")

plt.title("📊 Distribution of Time Spent (Hours)")
plt.xlabel("Time Spent (hours)")
plt.ylabel("Frequency")
plt.show()

# BOX PLOT – Attendance Rate by Final Grade
plt.figure(figsize=(7,5))
sns.boxplot(x="final_grade", y="attendance_rate", data=sample_df)

plt.title("📦 Box Plot: Attendance Rate by Final Grade")
plt.xlabel("Final Grade")
plt.ylabel("Attendance Rate")
plt.show()

# BOX PLOT – Final Exam Score by Final Grade
plt.figure(figsize=(7,5))
sns.boxplot(x="final_grade", y="final_exam_score", data=sample_df)

plt.title("📦 Box Plot: Final Exam Score by Grade")
plt.xlabel("Final Grade")
plt.ylabel("Final Exam Score")
plt.show()

# SCATTER PLOT – Time Spent vs Final Exam Score
plt.figure(figsize=(7,5))
sns.scatterplot(x="time_spent_hours", y="final_exam_score",
                hue="final_grade", data=sample_df)

plt.title("📈 Time Spent vs Final Exam Score")
plt.xlabel("Time Spent (hours)")
plt.ylabel("Final Exam Score")
plt.show()

# LINE GRAPH – Performance Trend Across Grades
plt.figure(figsize=(7,5))
plt.plot(grade_exam_avg["final_grade"],
         grade_exam_avg["avg(final_exam_score)"],
         marker="o")

plt.title("📉 Trend of Average Exam Score Across Grades")
plt.xlabel("Final Grade")
plt.ylabel("Average Exam Score")
plt.grid(True)
plt.show()

#HEATMAP – Correlation Between Important Features
import numpy as np

corr_data = sample_df[["attendance_rate", "time_spent_hours",
                       "quiz_avg_score", "final_exam_score",
                       "forum_posts", "badges_earned"]]

plt.figure(figsize=(8,6))
sns.heatmap(corr_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")

plt.title("🔥 Correlation Heatmap of Student Performance Features")
plt.show()
#to save cleaned output
df.write.csv("smart_lms_cleaned_bigdata", header=True, mode="overwrite")
print("✅ Cleaned Big Data Saved Successfully")
