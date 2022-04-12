import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Seed = 1
TargetFS = 10

#Load the Advertising dataset and save radio as
#the features and sales as the labels:
    
df = pd.read_csv("casestudy.csv")

Data = df.to_numpy()

Column_Names = df.columns

#Calculate total revenue for each year
Data15 = Data[Data[:,3]==2015,:]
Data16 = Data[Data[:,3]==2016,:]
Data17 = Data[Data[:,3]==2017,:]

#Total revenue calculation:
TotalRev15 = sum(Data15[:,2])
TotalRev16 = sum(Data16[:,2])
TotalRev17 = sum(Data17[:,2])
TotalRev = sum(Data[:,2])
        
print("Total revenue for year 2015 is $", TotalRev15)
print("Total revenue for year 2016 is $", TotalRev16)
print("Total revenue for year 2017 is $", TotalRev17)
print("Total revenue  is $", TotalRev)

#Unique customer calculation each year:
Unique15 = np.unique(Data15[:,1])
Unique16 = np.unique(Data16[:,1])
Unique17 = np.unique(Data17[:,1])

NewCust16 = 0 
NewCust17 = 0

#Revenue for new customers in 2016:
TotalNewCustRev16 = 0
for uq in Unique16:
    if not(uq in Unique15):
        NewCust16 = NewCust16 + 1
        TotalNewCustRev16 = TotalNewCustRev16 + Data16[Data16[:,1]==uq,2]
        
        
#Revenue for new customers in 2017:
TotalNewCustRev17 = 0
for uq in Unique17:
    if not(uq in Unique16):
        NewCust17 = NewCust17 + 1
        TotalNewCustRev17 = TotalNewCustRev17 + Data17[Data17[:,1]==uq,2]
        
print("Total revenue from new customers during year 2016 is $", TotalNewCustRev16)
print("Total revenue from new customers during year 2017 is $", TotalNewCustRev17)

#Customer growth calculation:
ExistingCustGrowth15 = TotalRev15
ExistingCustGrowth16 = TotalRev16-TotalRev15
ExistingCustGrowth17 = TotalRev17-TotalRev16

print("Existing Customer Growth year 2015 (say 2014 total revenue was $0) $", ExistingCustGrowth15)
print("Existing Customer Growth year 2016 $", ExistingCustGrowth16)
print("Existing Customer Growth year 2017 $", ExistingCustGrowth17)

#Revenue lost from attrium calculation:
RevenueLostAtt16 = (TotalRev16-TotalRev15)/TotalRev15 * 100
RevenueLostAtt17 = (TotalRev17-TotalRev16)/TotalRev16 * 100

print("Revenue lost from attrium during year 2016 is ", RevenueLostAtt16, '%')
print("Revenue lost from attrium during year 2017 is ", RevenueLostAtt17, '%')

#Total number of unique customer calculation:
TotalCust15 = len(Data15[:,1])
TotalCust16 = len(Data16[:,1])
TotalCust17 = len(Data17[:,1])

print("Total customers for year 2015 is $", TotalCust15)
print("Total customers for year 2016 is $", TotalCust16)
print("Total customers for year 2017 is $", TotalCust17)

NewCust15 = TotalCust15
LostCust15 = 0
LostCust16 = 0 
LostCust17 = 0

for uq in Unique15:
    if not(uq in Unique16):
        LostCust16 = LostCust16 + 1
        
for uq in Unique16:
    if not(uq in Unique17):
        LostCust17 = LostCust17 + 1
        
print("New customers for year 2015 is $", NewCust15)
print("New customers for year 2016 is $", NewCust16)
print("New customers for year 2017 is $", NewCust17)

print("Lost customers for year 2015 is $", LostCust15)
print("Lost customers for year 2016 is $", LostCust16)
print("Lost customers for year 2017 is $", LostCust17)
        
#Plot net revenue year 2015
plt.scatter(range(len(Data15)),Data15[:,2],color='blue', s=1, alpha=0.05)
plt.xlabel("Customer index")
plt.ylabel("Net revenue")
plt.title("Net revenue by customer during year 2015")
plt.show()

#Plot net revenue year 2016
plt.scatter(range(len(Data16)),Data16[:,2],color='blue', s=1, alpha=0.05)
plt.xlabel("Customer index")
plt.ylabel("Net revenue")
plt.title("Net revenue by customer during year 2016")
plt.show()

#Plot net revenue year 2017
plt.scatter(range(len(Data17)),Data17[:,2],color='blue', s=1, alpha=0.05 )
plt.xlabel("Customer index")
plt.ylabel("Net revenue")
plt.title("Net revenue by customer during year 2017")
plt.show()



        

            