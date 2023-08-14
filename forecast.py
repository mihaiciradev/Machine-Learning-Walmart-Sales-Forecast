# %%
#Mihai Cira - B. 29. Walmart Sales Forecast
#topic: analysis of the influence of holidays on sales

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import statsmodels.api as sm

from sklearn.preprocessing import MinMaxScaler


from sklearn import metrics
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor



# %%


# %%
#load the data
data = pd.read_csv('/kaggle/input/walmart-sales-forecast/train.csv')
stores = pd.read_csv('/kaggle/input/walmart-sales-forecast/stores.csv')
features = pd.read_csv('/kaggle/input/walmart-sales-forecast/features.csv')


# %%
#inspect the loaded data for the DATA table
data.shape
data.head()
data.info()
data.isna().sum()

# %%
#inspect the loaded data for the STORES table
stores.shape
stores.head()
stores.info()

# %%
#inspect the loaded data for the FEATURES table
features.shape
features.head()
features.info()
features.isna().sum()

# %%
# filling missing values
features['CPI'].fillna(features['CPI'].median(),inplace=True)
features['Unemployment'].fillna(features['Unemployment'].median(),inplace=True)

# %%
# Count unique values in the 'MarkDown1' column
features['MarkDown1'].value_counts().unique()

# %%
# Data cleaning for 'MarkDown' columns
from pandas.core.ops import flex_arith_method_FRAME
for i in range(1,6):
  features["MarkDown"+str(i)] = features["MarkDown"+str(i)].apply(lambda x: 0 if x<0 else x )
  features["MarkDown"+str(i)].fillna(value=0,inplace=True)

# %%
features.info()
features.head()

# %%
# Merge the 'data' dataframe ith the 'stores' and 'features'
data = pd.merge(data,stores,on='Store',how='left')
data = pd.merge(data,features,on=['Store','Date'],how='left')

data.head()

# %%
#convert 'date' column to datetime format, sort the dataframe based on 'date', set the 'date' as index
data['Date'] = pd.to_datetime(data['Date'],errors='coerce')
data.sort_values(by=['Date'],inplace=True)
data.set_index(data.Date, inplace=True)
data.head()

# %%
#check whether the column IsHoliday_x and IsHoliday_y are same or not
data['IsHoliday_x'].isin(data['IsHoliday_y']).all()

# %%
#Since this two columns are same so drop any one column and make another column as IsHoliday
data.drop(columns='IsHoliday_x',inplace=True)
data.rename(columns={"IsHoliday_y" : "IsHoliday"}, inplace=True)
data.info()

# %%
#exctracting the date values
data.head()
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Week'] = data['Date'].dt.week
data.head()

# %%
#Outlier Detection and Abnormalities
#group the 'data' dataframe by 'Store' and 'Dept' columns and calculates various statistical measures (maximum, minimum, mean, median, and standard deviation) of the 'Weekly_Sales' column for each group.


agg_data = data.groupby(['Store', 'Dept']).Weekly_Sales.agg(['max', 'min', 'mean', 'median', 'std']).reset_index()
agg_data.head()

# %%
agg_data.isnull().sum()

# %%
#merge 'data' with 'agg_data' on 'Store' and 'Dept' using a left join, resulting in 'store_data'.
store_data = pd.merge(left=data,right=agg_data,on=['Store', 'Dept'],how ='left')
store_data.head(2)

# %%
#drop rows with missing values from 'store_data' and assigns the modified DataFrame back to 'data'
store_data.dropna(inplace=True)
data = store_data.copy()

# %%
#Converts the 'Date' column in the 'data' DataFrame to datetime format and sort the dataframe based on 'date' column
data['Date'] = pd.to_datetime(data['Date'],errors='coerce')
data.sort_values(by=['Date'],inplace=True)
data.set_index(data.Date, inplace=True)
data.head()

# %%
#calculate the total markdown value by summing the individual markdown columns
data['Total_MarkDown'] = data['MarkDown1']+data['MarkDown2']+data['MarkDown3']+data['MarkDown4']+data['MarkDown5']
data.drop(['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5'], axis = 1,inplace=True)
data.head()

# %%
#crate a new dataframe called data_numeric by selecting only the columns specified in the numeric_col list from 'data' dataframe
print(data.shape)
numeric_col = ['Weekly_Sales','Size','Temperature','Fuel_Price','CPI','Unemployment','Total_MarkDown']
data_numeric = data[numeric_col].copy()
print(data_numeric.head())

# %%
#filter the data
data = data[(np.abs(stats.zscore(data_numeric)) < 2.5).all(axis = 1)]
data.shape

# %%
data=data[data['Weekly_Sales']>=0]
data.shape

# %%
data['IsHoliday'] = data['IsHoliday'].astype('int')
data.head()

# %%
#Average Monthly Sales
plt.figure(figsize=(14,8))
sns.barplot(x='Month',y='Weekly_Sales',data=data)
plt.ylabel('Sales',fontsize=14)
plt.xlabel('Months',fontsize=14)
plt.title('Average Monthly Sales',fontsize=16)
plt.savefig('avg_monthly_sales.png')
plt.grid()

# %%
# Monthly Sales for Each Year
data_monthly = pd.crosstab(data["Year"], data["Month"], values=data["Weekly_Sales"],aggfunc='sum')
data_monthly

# %%
#creates a grid of subplots to display the monthly sales for each year. Line graphs are plotted for each month of each year using the 'data_monthly' dataframe.
fig, axes = plt.subplots(3,4,figsize=(16,8))
plt.suptitle('Monthly Sales for each Year', fontsize=18)
k=1
for i in range(3):
    for j in range(4):
      sns.lineplot(ax=axes[i,j],data=data_monthly[k])
      plt.subplots_adjust(wspace=0.4,hspace=0.32)
      plt.ylabel(k,fontsize=12)
      plt.xlabel('Years',fontsize=12)
      k+=1

plt.savefig('monthly_sales_every_year.png')
plt.show()

# %%
# Holiday Distribution
plt.figure(figsize=(8,8))
plt.pie(data['IsHoliday'].value_counts(),labels=['No Holiday','Holiday'],autopct='%0.2f%%')
plt.title("Pie chart distribution",fontsize=14)
plt.legend()
plt.savefig('holiday_distribution.png')
plt.show() 


# %%
#One hot encoding
cat_col = ['Store','Dept','Type']
data_cat = data[cat_col].copy()
data_cat.head()

# %%
data_cat = pd.get_dummies(data_cat,columns=cat_col)
data_cat.head()

# %%
print(data.shape)
data = pd.concat([data, data_cat],axis=1)
print(data.shape)

# %%
data.drop(columns=cat_col,inplace=True)
data.drop(columns=['Date'],inplace=True)
data.shape

# %%
#performs min-max normalization on specified numerical columns of the dataframe
num_col = ['Weekly_Sales','Size','Temperature','Fuel_Price','CPI','Unemployment','Total_MarkDown','max','min','mean','median','std']


minmax_scale = MinMaxScaler(feature_range=(0, 1))
def normalization(df,col):
  for i in col:
    arr = df[i]
    arr = np.array(arr)
    df[i] = minmax_scale.fit_transform(arr.reshape(len(arr),1))
  return df

# %%
print(data.head())
data = normalization(data.copy(),num_col)
data.head()

# %%
# Finding Correlation between features
# generate a heatmap to visualize the correlation matrix between the selected numerical features in the dataset
plt.figure(figsize=(15,8))
corr = data[num_col].corr()
sns.heatmap(corr,vmax=1.0,annot=True)
plt.title('Correlation Matrix',fontsize=16)
plt.savefig('correlation_matrix.png')
plt.show()

# %%
# Feature Elimination
feature_col = data.columns.difference(['Weekly_Sales'])
feature_col

# %%
#train a random forest regressor model with 23 decision trees on the selected features and target variable.
radm_clf = RandomForestRegressor(oob_score=True,n_estimators=23)
radm_clf.fit(data[feature_col], data['Weekly_Sales'])

# %%
#computes the feature importance using a random forest regressor and creates a dataframe to store the rankings
indices = np.argsort(radm_clf.feature_importances_)[::-1]
feature_rank = pd.DataFrame(columns = ['rank', 'feature', 'importance'])

for f in range(data[feature_col].shape[1]):
    feature_rank.loc[f] = [f+1,
                           data[feature_col].columns[indices[f]],
                           radm_clf.feature_importances_[indices[f]]]

feature_rank

# %%
#selects the top 25 features from the feature_rank DataFrame and stores them in a list
x=feature_rank.loc[0:25,['feature']]
x=x['feature'].tolist()
print(x)

# %%
X = data[x]
Y = data['Weekly_Sales']

# %%
data = pd.concat([X,Y],axis=1)
data.head()

# %%
#building the model

#Splitting data into train and test data
X = data.drop(['Weekly_Sales'],axis=1)
Y = data.Weekly_Sales

# %%
#Split the data into training and testing sets
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.20, random_state=50)

# %%
rf = RandomForestRegressor()
rf.fit(X_train, y_train)

# %%
y_pred = rf.predict(X_test)

# %%
print("MAE" , metrics.mean_absolute_error(y_test, y_pred))
print("MSE" , metrics.mean_squared_error(y_test, y_pred))
print("RMSE" , np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R2" , metrics.explained_variance_score(y_test, y_pred))

# %%
rf_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
rf_df.head()

# %%
# generate a plot comparing the predicted values with the actual values for a subset of the test data
plt.figure(figsize=(15,8))
plt.title('Comparison between actual and predicted values',fontsize=16)
plt.plot(rf.predict(X_test[:100]), label="prediction", linewidth=3.0,color='blue')
plt.plot(y_test[:100].values, label="real_values", linewidth=3.0,color='red')
plt.legend(loc="best")
plt.show()



