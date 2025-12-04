import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

Hospital_Data = pd.read_csv("Clean_Hospital_Data.csv")
#---------------
# time series
#----------------

Hospital_Data["YearMonth"] = pd.to_datetime(
    Hospital_Data["Year"].astype(str) + "-" + Hospital_Data["Month"].astype(str) + "-01"
)
monthly_totals = Hospital_Data.groupby("YearMonth")["Total_Admissions"].sum().reset_index()

plt.figure(figsize=(10, 5))
plt.plot(monthly_totals["YearMonth"], monthly_totals["Total_Admissions"])
plt.xlabel("Year-Month")
plt.ylabel("Total Admissions")
plt.title("Monthly Total Hospital Admissions (2020 - 2024)")
plt.xticks(rotation=45)
plt.savefig("Monthly Total Hospital Admissions (2020 - 2024).png", bbox_inches='tight', dpi=300)
plt.clf()

#--------------
# hist
#-------------

plt.figure(figsize=(10, 5))
plt.hist(Hospital_Data["Total_Admissions"], 
         bins=20, color= "green", edgecolor = "black", linewidth = 1)
plt.title("Distribution of Total Admitted Patients")
plt.ylabel("Count")
plt.xlabel("Number of Patients")
plt.savefig("Distribution of Total Admitted Patients.png",bbox_inches='tight', dpi=300)
plt.clf()

