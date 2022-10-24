#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd 


# In[7]:


import numpy as np


# In[8]:


url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"
df= pd.read_csv(url,header=None) 
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

df.columns=headers


# In[9]:


df


# In[10]:


df.replace('?',np.NaN,inplace=True)


# In[11]:


df.dropna(subset=["price"],axis=0,inplace=True)


# In[12]:


import matplotlib 
from matplotlib import pyplot as plt


# In[13]:


plt.hist(df["price"], bins=3)
plt.xlabel("price")
plt.ylabel("count")
plt.title("price bins")


# In[14]:


df.dtypes


# In[15]:


df["price"].head(20)


# In[16]:


df["price"]= df["price"].astype('int')
df["horsepower"]=df["horsepower"].astype('float')
df["curb-weight"]=df["curb-weight"].astype('float')


# In[17]:


df.dtypes


# In[18]:


bins=np.linspace(min(df["price"]),max(df["price"]),4)
group_names=["low","medium","high"]
df["price-binned"]=pd.cut(df["price"],bins,labels=group_names,include_lowest=True)


# In[19]:


df["price-binned"].value_counts()


# In[20]:


df["drive-wheels"].value_counts().to_frame()


# In[21]:


import seaborn as sns


# In[22]:


sns.boxplot(x="drive-wheels",y="price",data=df)


# plt.title("scatterplot of enginesize vs price")
# plt.xlabel("engine size")
# plt.ylabel("price")

# In[23]:


x=df["engine-size"]
y=df["price"]
plt.title("engine size vs price")
plt.xlabel("engine size")
plt.ylabel("price")
plt.scatter(x,y)


# In[24]:


df_test=df[['drive-wheels','body-style','price']]
df_group=df_test.groupby(['drive-wheels','body-style'],as_index=False).mean()


# In[25]:


df_group


# In[26]:


df_pivot=df_group.pivot(index='drive-wheels',columns='body-style')


# In[27]:


df_pivot


# In[28]:


plt.pcolor(df_pivot,cmap='RdBu')
plt.colorbar()
plt.show()


# In[29]:


sns.regplot(x="engine-size",y="price",data=df)
plt.ylim(0,)


# In[30]:


sns.regplot(x="highway-mpg",y="price",data=df)
plt.ylim(0,)


# In[31]:


sns.regplot(x="peak-rpm",y="price",data=df)
plt.ylim(0,)


# In[ ]:


import scipy


# In[ ]:


from scipy import stats


# In[ ]:


pearson_coeff,p_value=stats.pearsonr(df['horsepower'],df['price'])


# In[ ]:


df.dtypes


# In[ ]:


df.dropna(subset=["horsepower"],axis=0,inplace=True)


# In[ ]:


pearson_coeff


# In[ ]:


p_value


# In[ ]:


cont_table= pd.crosstab(df["fuel-type"],df["aspiration"])


# In[ ]:


import scipy
from scipy import stats


# In[ ]:


scipy.stats.chi2_contingency(cont_table,correction=True)


# In[32]:


cont_table


# In[33]:


import sklearn


# In[34]:


from sklearn.linear_model import LinearRegression
lm=LinearRegression()


# In[35]:


x=df[['highway-mpg']]
y=df['price']
lm.fit(x,y)
Yhat=lm.predict(x)


# lm.intercept_
# lm.coef_

# In[36]:


lm.intercept_


# In[37]:


lm.coef_


# In[38]:


sns.residplot(df['highway-mpg'],df['price'])


# In[39]:


axl=sns.distplot(df['price'], hist=False,color="r",label="actual value")
sns.distplot(Yhat,hist=False,color="b",label="fitted values",ax=axl)


# In[40]:


#r2
x=df[['highway-mpg']]
y=df['price']
lm.fit(x,y)
lm.score(x,y)


# In[41]:


lm.predict(np.array(30.0).reshape(-1,1))


# In[42]:


new_input=np.arange(1,101,1).reshape(-1,1)


# In[43]:


yhat=lm.predict(new_input)


# In[44]:


yhat


# In[45]:


from sklearn.metrics import mean_squared_error


# In[46]:


mse = mean_squared_error(df['price'], Yhat)
print('The mean square error of price and predicted value is: ', mse)


# In[48]:


from sklearn.preprocessing import standardScaler
SCALE= standardscaler()
SCALE.fit(x_data[['horsepower','highway-mpg']])
x_scale=SCALE.transform(x_data[['horsepower','highway-mpg']])


# In[50]:


from sklearn.preprocessing import StandardScaler


# In[53]:


SCALE= StandardScaler()


# In[56]:


SCALE.fit(df[['horsepower','highway-mpg']])


# In[57]:


x_scale=SCALE.transform(df[['horsepower','highway-mpg']])


# In[59]:


x_scale.shape


# In[66]:


from sklearn.preprocessing import PolynomialFeatures


# In[67]:


pr=PolynomialFeatures(degree=2,include_bias=False)


# In[72]:


x_polly=pr.fit_transform(df[['horsepower','compression-ratio']])


# In[71]:


df.dropna(subset=["horsepower"],axis=0,inplace=True)


# In[73]:


x_polly.shape


# In[75]:


y_data = df['price']
x_data=df.drop('price',axis=1)


# In[76]:


from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10, random_state=1)


print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])


# In[84]:


from sklearn.linear_model import LinearRegression
lre = LinearRegression()
lre.fit(x_train[['horsepower']],y_train)


# In[85]:


lre.score(x_test[['horsepower']],y_test)


# In[86]:


lre.score(x_train[['horsepower']], y_train)


# In[87]:


x_train1, x_test1, y_train1, y_test1 = train_test_split(x_data, y_data, test_size=0.4, random_state=0)
lre.fit(x_train1[['horsepower']],y_train1)
lre.score(x_test1[['horsepower']],y_test1)


# In[88]:


lre.score(x_train1[['horsepower']],y_train1)


# In[91]:


from sklearn.model_selection import cross_val_score


# In[92]:


rcross= cross_val_score(lre,x_data[['horsepower']],y_data,cv=3)


# In[93]:


rcross


# In[95]:


print("The mean of the folds are", rcross.mean(), "and the standard deviation is" , rcross.std())


# In[96]:


-1 * cross_val_score(lre,x_data[['horsepower']], y_data,cv=4,scoring='neg_mean_squared_error')


# In[97]:


from sklearn.model_selection import cross_val_predict


# In[98]:


yhat = cross_val_predict(lre,x_data[['horsepower']], y_data,cv=4)


# In[99]:


yhat[0:5]


# In[100]:


lr = LinearRegression()
lr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_train)


# In[101]:


yhat_train = lr.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
yhat_train[0:5]


# In[102]:


yhat_test = lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
yhat_test[0:5]


# In[103]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[113]:


Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'

distributionplot = (y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title)
sns.displot(distributionplot)


# In[117]:


Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
DistributionPlot=(y_test,yhat_test,"Actual Values (Test)","Predicted Values (Test)",Title)
sns.displot(DistributionPlot)


# In[118]:


from sklearn.preprocessing import PolynomialFeatures


# In[119]:


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.45, random_state=0)


# In[120]:


pr = PolynomialFeatures(degree=5)
x_train_pr = pr.fit_transform(x_train[['horsepower']])
x_test_pr = pr.fit_transform(x_test[['horsepower']])
pr


# In[121]:


def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')

    plt.show()
    plt.close()


# In[122]:


Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
DistributionPlot(y_test,yhat_test,"Actual Values (Test)","Predicted Values (Test)",Title)


# In[124]:


Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'

DistributionPlot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title)


# In[125]:


poly = LinearRegression()
poly.fit(x_train_pr, y_train)


# In[126]:


yhat = poly.predict(x_test_pr)
yhat[0:5]


# In[127]:


print("Predicted values:", yhat[0:4])
print("True values:", y_test[0:4].values)


# In[128]:


def PollyPlot(xtrain, xtest, y_train, y_test, lr,poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    
    #training data 
    #testing data 
    # lr:  linear regression object 
    #poly_transform:  polynomial transformation object 
 
    xmax=max([xtrain.values.max(), xtest.values.max()])

    xmin=min([xtrain.values.min(), xtest.values.min()])

    x=np.arange(xmin, xmax, 0.1)


    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()


# In[129]:


PollyPlot(x_train[['horsepower']], x_test[['horsepower']], y_train, y_test, poly,pr)


# In[132]:


Rsqu_test = []
order = [1, 2, 3, 4]
for n in order:
    pr = PolynomialFeatures(degree=n)
    
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    
    x_test_pr = pr.fit_transform(x_test[['horsepower']])    


# In[136]:


lr.fit(x_train_pr, y_train)
    
Rsqu_test.append(lr.score(x_test_pr, y_test))
    
plt.plot(order, Rsqu_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data')
plt.text(3, 0.75, 'Maximum R^2 ')    


# In[137]:


Rsqu_test = []

order = [1, 2, 3, 4]
for n in order:
    pr = PolynomialFeatures(degree=n)
    
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    
    x_test_pr = pr.fit_transform(x_test[['horsepower']])    
    
    lr.fit(x_train_pr, y_train)
    
    Rsqu_test.append(lr.score(x_test_pr, y_test))

plt.plot(order, Rsqu_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data')
plt.text(3, 0.75, 'Maximum R^2 ') 


# In[145]:


df.dropna(subset=["horsepower"],axis=0,inplace=True)


# In[146]:


df.dropna(subset=["curb-weight"],axis=0,inplace=True)


# In[147]:


df.dropna(subset=["engine-size"],axis=0,inplace=True)
df.dropna(subset=["highway-mpg"],axis=0,inplace=True)
df.dropna(subset=["normalized-losses"],axis=0,inplace=True)
df.dropna(subset=["symboling"],axis=0,inplace=True)


# In[149]:


pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[['horsepower', 'highway-mpg']])
x_test_pr=pr.fit_transform(x_test[['horsepower','highway-mpg',]])


# In[151]:


from sklearn.linear_model import Ridge


# In[152]:


RM= Ridge(alpha=1)


# In[153]:


RM.fit(x_train_pr,y_train)


# In[156]:


yhat= RM.predict(x_test_pr)


# In[157]:


print('predicted:',yhat[0:4])
print('test set:', y_test[0:4].values)


# In[159]:


from tqdm import tqdm

Rsqu_test = []
Rsqu_train = []
dummy1 = []
Alpha = 10 * np.array(range(0,1000))
pbar = tqdm(Alpha)

for alpha in pbar:
    RigeModel = Ridge(alpha=alpha) 
    RigeModel.fit(x_train_pr, y_train)
    test_score, train_score = RigeModel.score(x_test_pr, y_test), RigeModel.score(x_train_pr, y_train)
    
    pbar.set_postfix({"Test Score": test_score, "Train Score": train_score})

    Rsqu_test.append(test_score)
    Rsqu_train.append(train_score)


# In[160]:


width = 12
height = 10
plt.figure(figsize=(width, height))

plt.plot(Alpha,Rsqu_test, label='validation data  ')
plt.plot(Alpha,Rsqu_train, 'r', label='training Data ')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend()


# In[161]:


from sklearn.model_selection import GridSearchCV


# In[162]:


parameters1= [{'alpha': [0.001,0.1,1, 10, 100, 1000, 10000, 100000, 100000]}]
parameters1


# In[163]:


RR=Ridge()
RR


# In[164]:


Grid1 = GridSearchCV(RR, parameters1,cv=4)


# In[165]:


Grid1.fit(x_data[['horsepower', 'highway-mpg']], y_data)


# In[166]:


BestRR=Grid1.best_estimator_
BestRR


# In[167]:


BestRR.score(x_test[['horsepower', 'highway-mpg']], y_test)


# In[ ]:




