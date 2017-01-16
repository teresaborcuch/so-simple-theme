
## Problem Statement
A liquor store owner is looking to open a new store in the state of Iowa. Given records of Iowa's liquors sales from 2015, I will assess different locations around the state for market potential. The records include number of bottles sold, product details, pricing for transactions at 1,161 stores, as well as store location. From this data, I can investigate the relationship between store location and total volume of sales to advise the store owner which counties are promising locations to build the next store.

I hypothesize that the best location to build a store will be counties with few stores, but high sales, as this indicates a large demand but low competition.


```python
import pandas as pd
import sklearn
from sklearn import linear_model
from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (8, 8)

## Load the data into a DataFrame and preview
transactions = pd.read_csv('/Users/teresaborcuch/DSI-course-materials/curriculum/04-lessons/week-03/2.4-lab/Iowa_Liquor_sales_sample_10pct.csv')
transactions.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Store Number</th>
      <th>City</th>
      <th>Zip Code</th>
      <th>County Number</th>
      <th>County</th>
      <th>Category</th>
      <th>Category Name</th>
      <th>Vendor Number</th>
      <th>Item Number</th>
      <th>Item Description</th>
      <th>Bottle Volume (ml)</th>
      <th>State Bottle Cost</th>
      <th>State Bottle Retail</th>
      <th>Bottles Sold</th>
      <th>Sale (Dollars)</th>
      <th>Volume Sold (Liters)</th>
      <th>Volume Sold (Gallons)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11/04/2015</td>
      <td>3717</td>
      <td>SUMNER</td>
      <td>50674</td>
      <td>9.0</td>
      <td>Bremer</td>
      <td>1051100.0</td>
      <td>APRICOT BRANDIES</td>
      <td>55</td>
      <td>54436</td>
      <td>Mr. Boston Apricot Brandy</td>
      <td>750</td>
      <td>$4.50</td>
      <td>$6.75</td>
      <td>12</td>
      <td>$81.00</td>
      <td>9.0</td>
      <td>2.38</td>
    </tr>
    <tr>
      <th>1</th>
      <td>03/02/2016</td>
      <td>2614</td>
      <td>DAVENPORT</td>
      <td>52807</td>
      <td>82.0</td>
      <td>Scott</td>
      <td>1011100.0</td>
      <td>BLENDED WHISKIES</td>
      <td>395</td>
      <td>27605</td>
      <td>Tin Cup</td>
      <td>750</td>
      <td>$13.75</td>
      <td>$20.63</td>
      <td>2</td>
      <td>$41.26</td>
      <td>1.5</td>
      <td>0.40</td>
    </tr>
    <tr>
      <th>2</th>
      <td>02/11/2016</td>
      <td>2106</td>
      <td>CEDAR FALLS</td>
      <td>50613</td>
      <td>7.0</td>
      <td>Black Hawk</td>
      <td>1011200.0</td>
      <td>STRAIGHT BOURBON WHISKIES</td>
      <td>65</td>
      <td>19067</td>
      <td>Jim Beam</td>
      <td>1000</td>
      <td>$12.59</td>
      <td>$18.89</td>
      <td>24</td>
      <td>$453.36</td>
      <td>24.0</td>
      <td>6.34</td>
    </tr>
    <tr>
      <th>3</th>
      <td>02/03/2016</td>
      <td>2501</td>
      <td>AMES</td>
      <td>50010</td>
      <td>85.0</td>
      <td>Story</td>
      <td>1071100.0</td>
      <td>AMERICAN COCKTAILS</td>
      <td>395</td>
      <td>59154</td>
      <td>1800 Ultimate Margarita</td>
      <td>1750</td>
      <td>$9.50</td>
      <td>$14.25</td>
      <td>6</td>
      <td>$85.50</td>
      <td>10.5</td>
      <td>2.77</td>
    </tr>
    <tr>
      <th>4</th>
      <td>08/18/2015</td>
      <td>3654</td>
      <td>BELMOND</td>
      <td>50421</td>
      <td>99.0</td>
      <td>Wright</td>
      <td>1031080.0</td>
      <td>VODKA 80 PROOF</td>
      <td>297</td>
      <td>35918</td>
      <td>Five O'clock Vodka</td>
      <td>1750</td>
      <td>$7.20</td>
      <td>$10.80</td>
      <td>12</td>
      <td>$129.60</td>
      <td>21.0</td>
      <td>5.55</td>
    </tr>
  </tbody>
</table>
</div>



### Clean, Refine, and Mine the Data
I'll remove redundant columns, drop rows with incomplete data, and filter to only include stores that were open for all of 2015. Additionally, I'll calculate average price per bottle at each store and profit margin, and count the number of stores in each county


```python
transactions = transactions.drop(['County Number', 'Vendor Number','Item Number','Item Description', 'Volume Sold (Gallons)'], axis = 1)
```


```python
# Remove $
cols = ["State Bottle Cost", "State Bottle Retail", "Sale (Dollars)"]
for col in cols:
    transactions[col] = transactions[col].apply(lambda x: float(x[1:]))
```


```python
# Drop NA's
transactions = transactions.dropna()
```


```python
# Convert dates
transactions["Date"] = pd.to_datetime(transactions["Date"], format="%m/%d/%Y")
```


```python
# Calculate margins, unit prices
transactions["Margin"] = (transactions["State Bottle Retail"] - transactions["State Bottle Cost"]) * transactions["Bottles Sold"]
transactions["Price per Liter"] = transactions["Sale (Dollars)"] / transactions["Volume Sold (Liters)"]
transactions["Price per Bottle"] = transactions["Sale (Dollars)"]/ transactions["Bottles Sold"]
```


```python
# Sales per store, 2015
# Filter by our start and end dates
transactions.sort_values(by=["Store Number", "Date"], inplace=True)
start_date = pd.Timestamp("20150101")
end_date = pd.Timestamp("20151231")
mask = (transactions['Date'] >= start_date) & (transactions['Date'] <= end_date)
sales = transactions[mask]

# Group by store name
sales = sales.groupby(by="Store Number", as_index=False)
# Compute sums, means
sales = sales.agg({"County": lambda x: x.iloc[0],
                   "Sale (Dollars)": [np.sum, np.mean],
                   "Volume Sold (Liters)": [np.sum, np.mean],
                   "Margin": np.sum,
                   "Price per Liter": np.mean,
                   "Zip Code": lambda x: x.iloc[0], # just extract once, should be the same
                   "City": lambda x: x.iloc[0],
                   "Bottles Sold": [np.sum, np.mean],
                  "Price per Bottle": np.mean})
# Collapse the column indices
sales.columns = [' '.join(col).strip() for col in sales.columns.values]
# Rename columns
sales.columns = [u'store_num', u'county', u'city', u'mean_ppb',u'total_bottles',
                 u'mean_bottles', u'total_sales',u'mean_sales', u'total_liters',
                 u'mean_liters', u'zip_code',u'mean_ppl', u'margin']

cols = ['store_num','zip_code','county','city','total_sales','mean_sales','margin','total_bottles','mean_bottles','mean_ppb','total_liters','mean_liters','mean_ppl']
sales = sales[cols]
sales["store_num"] = sales["store_num"].astype('string')
```


```python
# Count number of stores per area
sales['stores_per_county'] = sales.groupby('county')['county'].transform('count')
sales['stores_per_city'] = sales.groupby('city')['city'].transform('count')
sales['stores_per_zip'] = sales.groupby('zip_code')['zip_code'].transform('count')
sales.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_num</th>
      <th>zip_code</th>
      <th>county</th>
      <th>city</th>
      <th>total_sales</th>
      <th>mean_sales</th>
      <th>margin</th>
      <th>total_bottles</th>
      <th>mean_bottles</th>
      <th>mean_ppb</th>
      <th>total_liters</th>
      <th>mean_liters</th>
      <th>mean_ppl</th>
      <th>stores_per_county</th>
      <th>stores_per_city</th>
      <th>stores_per_zip</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2106</td>
      <td>50613</td>
      <td>Black Hawk</td>
      <td>CEDAR FALLS</td>
      <td>146038.70</td>
      <td>277.640114</td>
      <td>48742.20</td>
      <td>10355</td>
      <td>19.686312</td>
      <td>15.459734</td>
      <td>9719.85</td>
      <td>18.478802</td>
      <td>17.844997</td>
      <td>72</td>
      <td>17</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2113</td>
      <td>50543</td>
      <td>Webster</td>
      <td>GOWRIE</td>
      <td>9310.22</td>
      <td>63.334830</td>
      <td>3109.04</td>
      <td>671</td>
      <td>4.564626</td>
      <td>16.315646</td>
      <td>659.85</td>
      <td>4.488776</td>
      <td>18.507700</td>
      <td>20</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2130</td>
      <td>50703</td>
      <td>Black Hawk</td>
      <td>WATERLOO</td>
      <td>111583.91</td>
      <td>285.380844</td>
      <td>37229.32</td>
      <td>7418</td>
      <td>18.971867</td>
      <td>14.740767</td>
      <td>6879.37</td>
      <td>17.594297</td>
      <td>16.817589</td>
      <td>72</td>
      <td>46</td>
      <td>16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2152</td>
      <td>50469</td>
      <td>Cerro Gordo</td>
      <td>ROCKWELL</td>
      <td>7721.08</td>
      <td>54.759433</td>
      <td>2587.53</td>
      <td>573</td>
      <td>4.063830</td>
      <td>12.887660</td>
      <td>633.37</td>
      <td>4.491986</td>
      <td>13.020765</td>
      <td>20</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2178</td>
      <td>52172</td>
      <td>Allamakee</td>
      <td>WAUKON</td>
      <td>24324.18</td>
      <td>102.633671</td>
      <td>8165.70</td>
      <td>1928</td>
      <td>8.135021</td>
      <td>14.558692</td>
      <td>1917.12</td>
      <td>8.089114</td>
      <td>16.053844</td>
      <td>6</td>
      <td>3</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig = plt.figure(figsize = (12,6))
ax1 = fig.add_subplot(1,3,1)
ax1.hist(sales['total_sales'], bins = 100)
ax1.set_title("Histogram of Total Sales")
ax2 = fig.add_subplot(1,3,2)
ax2.hist(sales['total_bottles'], bins = 100)
ax2.set_title("Histogram of Bottles Sold")
ax3 = fig.add_subplot(1,3,3)
ax3.scatter(sales['total_bottles'], sales['total_sales'])
ax3.set_title("# Bottles Sold vs Total Sales")
ax3.set_xlim(0,70000)
ax3.set_ylim(0,1100000)
plt.show()
```


![png](https://github.com/teresaborcuch/teresaborcuch.github.io/blob/master/images/Project_3_Iowa_liquor_sales_files/Project_3_Iowa_liquor_sales_10_0.png?raw=true)



```python
sales['total_sales'].describe()
```




    count      1372.000000
    mean      20757.042259
    std       50787.618666
    min          39.020000
    25%        3202.200000
    50%        7226.150000
    75%       19355.705000
    max      997924.420000
    Name: total_sales, dtype: float64




```python
sales['total_bottles'].describe()
```




    count     1372.000000
    mean      1583.717201
    std       3225.321436
    min          1.000000
    25%        312.000000
    50%        665.500000
    75%       1608.500000
    max      62827.000000
    Name: total_bottles, dtype: float64



Given the extreme right skew for both total sales and number of bottles, I will eliminate the the two outlying stores that sold over 30,000 bottles in a year.


```python
# Remove two most extreme outliers from Polk County
sales = sales[sales['total_bottles'] < 30000]
```


```python
county_pivot = sales.pivot_table(index = ['county'], values = ['total_sales'], aggfunc = np.mean)
county_pivot = county_pivot.sort_values('total_sales', ascending = False)
county_pivot.head(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_sales</th>
    </tr>
    <tr>
      <th>county</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Johnson</th>
      <td>32423.122830</td>
    </tr>
    <tr>
      <th>Scott</th>
      <td>31401.318462</td>
    </tr>
    <tr>
      <th>Dallas</th>
      <td>29456.824118</td>
    </tr>
    <tr>
      <th>Cerro Gordo</th>
      <td>28495.049000</td>
    </tr>
    <tr>
      <th>Howard</th>
      <td>27857.660000</td>
    </tr>
    <tr>
      <th>Dickinson</th>
      <td>27819.681429</td>
    </tr>
    <tr>
      <th>Woodbury</th>
      <td>26324.404211</td>
    </tr>
    <tr>
      <th>Kossuth</th>
      <td>26314.505714</td>
    </tr>
    <tr>
      <th>Jones</th>
      <td>26023.592500</td>
    </tr>
    <tr>
      <th>Linn</th>
      <td>25747.564356</td>
    </tr>
  </tbody>
</table>
</div>




```python
high_sale_counties = ['Johnson','Scott', 'Dallas','Cerro Gordo', 'Howard']
fig = plt.figure(figsize = (14,5))
n = 1
for i in high_sale_counties:
    ax = fig.add_subplot(1,5,n)
    n +=1
    mask = (sales['county']== i)
    ax.scatter(sales[mask]['total_bottles'], sales[mask]['total_sales'])
    ax.set_title(i)
    ax.set_ylim(0,500000)
    ax.set_xlim(0,30000)
plt.show()
```


![png](https://github.com/teresaborcuch/teresaborcuch.github.io/blob/master/images/Project_3_Iowa_liquor_sales_files/Project_3_Iowa_liquor_sales_16_0.png?raw=true)



```python
# Look for correlations between # of stores in area and total average sales
sales.corr()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>total_sales</th>
      <th>mean_sales</th>
      <th>margin</th>
      <th>total_bottles</th>
      <th>mean_bottles</th>
      <th>mean_ppb</th>
      <th>total_liters</th>
      <th>mean_liters</th>
      <th>mean_ppl</th>
      <th>stores_per_county</th>
      <th>stores_per_city</th>
      <th>stores_per_zip</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>total_sales</th>
      <td>1.000000</td>
      <td>0.252291</td>
      <td>0.999995</td>
      <td>0.976454</td>
      <td>0.200997</td>
      <td>0.297484</td>
      <td>0.988715</td>
      <td>0.349756</td>
      <td>0.126936</td>
      <td>0.080028</td>
      <td>0.110614</td>
      <td>0.165568</td>
    </tr>
    <tr>
      <th>mean_sales</th>
      <td>0.252291</td>
      <td>1.000000</td>
      <td>0.252501</td>
      <td>0.198056</td>
      <td>0.906910</td>
      <td>0.235610</td>
      <td>0.236028</td>
      <td>0.922074</td>
      <td>0.129487</td>
      <td>-0.003878</td>
      <td>-0.008826</td>
      <td>0.069060</td>
    </tr>
    <tr>
      <th>margin</th>
      <td>0.999995</td>
      <td>0.252501</td>
      <td>1.000000</td>
      <td>0.976406</td>
      <td>0.201150</td>
      <td>0.297517</td>
      <td>0.988915</td>
      <td>0.350342</td>
      <td>0.126551</td>
      <td>0.079800</td>
      <td>0.110344</td>
      <td>0.165389</td>
    </tr>
    <tr>
      <th>total_bottles</th>
      <td>0.976454</td>
      <td>0.198056</td>
      <td>0.976406</td>
      <td>1.000000</td>
      <td>0.189055</td>
      <td>0.228083</td>
      <td>0.972417</td>
      <td>0.283639</td>
      <td>0.130003</td>
      <td>0.107248</td>
      <td>0.168284</td>
      <td>0.193570</td>
    </tr>
    <tr>
      <th>mean_bottles</th>
      <td>0.200997</td>
      <td>0.906910</td>
      <td>0.201150</td>
      <td>0.189055</td>
      <td>1.000000</td>
      <td>0.004154</td>
      <td>0.190225</td>
      <td>0.875130</td>
      <td>0.079237</td>
      <td>0.014377</td>
      <td>0.059516</td>
      <td>0.105248</td>
    </tr>
    <tr>
      <th>mean_ppb</th>
      <td>0.297484</td>
      <td>0.235610</td>
      <td>0.297517</td>
      <td>0.228083</td>
      <td>0.004154</td>
      <td>1.000000</td>
      <td>0.278476</td>
      <td>0.208188</td>
      <td>0.523065</td>
      <td>0.004774</td>
      <td>-0.041993</td>
      <td>-0.020655</td>
    </tr>
    <tr>
      <th>total_liters</th>
      <td>0.988715</td>
      <td>0.236028</td>
      <td>0.988915</td>
      <td>0.972417</td>
      <td>0.190225</td>
      <td>0.278476</td>
      <td>1.000000</td>
      <td>0.357992</td>
      <td>0.073221</td>
      <td>0.063173</td>
      <td>0.096576</td>
      <td>0.155723</td>
    </tr>
    <tr>
      <th>mean_liters</th>
      <td>0.349756</td>
      <td>0.922074</td>
      <td>0.350342</td>
      <td>0.283639</td>
      <td>0.875130</td>
      <td>0.208188</td>
      <td>0.357992</td>
      <td>1.000000</td>
      <td>-0.034591</td>
      <td>-0.042692</td>
      <td>-0.043877</td>
      <td>0.040112</td>
    </tr>
    <tr>
      <th>mean_ppl</th>
      <td>0.126936</td>
      <td>0.129487</td>
      <td>0.126551</td>
      <td>0.130003</td>
      <td>0.079237</td>
      <td>0.523065</td>
      <td>0.073221</td>
      <td>-0.034591</td>
      <td>1.000000</td>
      <td>0.180935</td>
      <td>0.225422</td>
      <td>0.168533</td>
    </tr>
    <tr>
      <th>stores_per_county</th>
      <td>0.080028</td>
      <td>-0.003878</td>
      <td>0.079800</td>
      <td>0.107248</td>
      <td>0.014377</td>
      <td>0.004774</td>
      <td>0.063173</td>
      <td>-0.042692</td>
      <td>0.180935</td>
      <td>1.000000</td>
      <td>0.628172</td>
      <td>0.385426</td>
    </tr>
    <tr>
      <th>stores_per_city</th>
      <td>0.110614</td>
      <td>-0.008826</td>
      <td>0.110344</td>
      <td>0.168284</td>
      <td>0.059516</td>
      <td>-0.041993</td>
      <td>0.096576</td>
      <td>-0.043877</td>
      <td>0.225422</td>
      <td>0.628172</td>
      <td>1.000000</td>
      <td>0.574656</td>
    </tr>
    <tr>
      <th>stores_per_zip</th>
      <td>0.165568</td>
      <td>0.069060</td>
      <td>0.165389</td>
      <td>0.193570</td>
      <td>0.105248</td>
      <td>-0.020655</td>
      <td>0.155723</td>
      <td>0.040112</td>
      <td>0.168533</td>
      <td>0.385426</td>
      <td>0.574656</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Total sales is most strongly correlated with the volume of liquor sold (either in bottles or liters), and there is a weak correlation with the price per bottle. However, the number of stores in each city or county does not seem to have a significant relationship with sales, so I won't include this parameter in the models.

## Making and Fitting Models
The goal is to construct linear regression models that capture the relationship between location of a store, its total bottle sales, its price per bottle, and its total sales. I'll start by looking at the relationship between total bottles sold and total sales.


```python
# total sale ~ total bottles
X = (sales['total_bottles']).values.reshape(-1,1)
y = sales['total_sales']

lm1 = linear_model.LinearRegression(normalize = True)
model1 = lm1.fit(X,y)
pred_y = model1.predict(X)

plt.scatter(X,y, color = 'b', marker = 'o')
plt.plot(X,pred_y)
plt.xlim(0,30000)
plt.ylim(0,450000)
plt.xlabel("Total Bottles Sold Per Store")
plt.ylabel("Total Sales Per Store")
plt.show()
print "R_squared: ", sklearn.metrics.r2_score(y, pred_y)
```


![png](https://github.com/teresaborcuch/teresaborcuch.github.io/blob/master/images/Project_3_Iowa_liquor_sales_files/Project_3_Iowa_liquor_sales_20_0.png?raw=true)


    R_squared:  0.953462200558


The total number of bottles is a good predictor of total sales.

### Cross-validated Lasso Regression
Now, I'll include the county of each store as a catagorical variable, as well as the price per bottle at each store. I'll use 3-fold cross-validation to increase predictive power.


```python
# make dummy variables for county
dummies = pd.get_dummies(sales["county"], prefix="county")
county_df = pd.concat([sales, dummies], axis=1)
```


```python
# total sales ~ total bottles + county + price per bottle
county_df = county_df.sort_values('total_bottles')
X = county_df[['total_bottles','mean_ppb','county_Adair', 'county_Adams', 'county_Allamakee',
       'county_Appanoose', 'county_Audubon', 'county_Benton',
       'county_Black Hawk', 'county_Boone', 'county_Bremer',
       'county_Buchanan', 'county_Buena Vista', 'county_Butler',
       'county_Calhoun', 'county_Carroll', 'county_Cass', 'county_Cedar',
       'county_Cerro Gordo', 'county_Cherokee', 'county_Chickasaw',
       'county_Clarke', 'county_Clay', 'county_Clayton', 'county_Clinton',
       'county_Crawford', 'county_Dallas', 'county_Davis',
       'county_Decatur', 'county_Delaware', 'county_Des Moines',
       'county_Dickinson', 'county_Dubuque', 'county_Emmet',
       'county_Fayette', 'county_Floyd', 'county_Franklin',
       'county_Fremont', 'county_Greene', 'county_Grundy',
       'county_Guthrie', 'county_Hamilton', 'county_Hancock',
       'county_Hardin', 'county_Harrison', 'county_Henry', 'county_Howard',
       'county_Humboldt', 'county_Ida', 'county_Iowa', 'county_Jackson',
       'county_Jasper', 'county_Jefferson', 'county_Johnson',
       'county_Jones', 'county_Keokuk', 'county_Kossuth', 'county_Lee',
       'county_Linn', 'county_Louisa', 'county_Lucas', 'county_Lyon',
       'county_Madison', 'county_Mahaska', 'county_Marion',
       'county_Marshall', 'county_Mills', 'county_Mitchell',
       'county_Monona', 'county_Monroe', 'county_Montgomery',
       'county_Muscatine', "county_O'Brien", 'county_Osceola',
       'county_Page', 'county_Palo Alto', 'county_Plymouth',
       'county_Pocahontas', 'county_Polk', 'county_Pottawattamie',
       'county_Poweshiek', 'county_Ringgold', 'county_Sac', 'county_Scott',
       'county_Shelby', 'county_Sioux', 'county_Story', 'county_Tama',
       'county_Taylor', 'county_Union', 'county_Van Buren',
       'county_Wapello', 'county_Warren', 'county_Washington',
       'county_Wayne', 'county_Webster', 'county_Winnebago',
       'county_Winneshiek', 'county_Woodbury', 'county_Worth',
       'county_Wright']]
y = county_df['total_sales']

lm = linear_model.LassoCV(alphas = np.arange(0.1,10,0.1), normalize = True, cv = 3)
model = lm.fit(X,y)
predictions = model.predict(X)

my_counties = ['Dallas','Fremont']
mask = (county_df['county'].isin(my_counties))
X_mask = county_df[mask][['total_bottles','mean_ppb','county_Adair', 'county_Adams', 'county_Allamakee',
       'county_Appanoose', 'county_Audubon', 'county_Benton',
       'county_Black Hawk', 'county_Boone', 'county_Bremer',
       'county_Buchanan', 'county_Buena Vista', 'county_Butler',
       'county_Calhoun', 'county_Carroll', 'county_Cass', 'county_Cedar',
       'county_Cerro Gordo', 'county_Cherokee', 'county_Chickasaw',
       'county_Clarke', 'county_Clay', 'county_Clayton', 'county_Clinton',
       'county_Crawford', 'county_Dallas', 'county_Davis',
       'county_Decatur', 'county_Delaware', 'county_Des Moines',
       'county_Dickinson', 'county_Dubuque', 'county_Emmet',
       'county_Fayette', 'county_Floyd', 'county_Franklin',
       'county_Fremont', 'county_Greene', 'county_Grundy',
       'county_Guthrie', 'county_Hamilton', 'county_Hancock',
       'county_Hardin', 'county_Harrison', 'county_Henry', 'county_Howard',
       'county_Humboldt', 'county_Ida', 'county_Iowa', 'county_Jackson',
       'county_Jasper', 'county_Jefferson', 'county_Johnson',
       'county_Jones', 'county_Keokuk', 'county_Kossuth', 'county_Lee',
       'county_Linn', 'county_Louisa', 'county_Lucas', 'county_Lyon',
       'county_Madison', 'county_Mahaska', 'county_Marion',
       'county_Marshall', 'county_Mills', 'county_Mitchell',
       'county_Monona', 'county_Monroe', 'county_Montgomery',
       'county_Muscatine', "county_O'Brien", 'county_Osceola',
       'county_Page', 'county_Palo Alto', 'county_Plymouth',
       'county_Pocahontas', 'county_Polk', 'county_Pottawattamie',
       'county_Poweshiek', 'county_Ringgold', 'county_Sac', 'county_Scott',
       'county_Shelby', 'county_Sioux', 'county_Story', 'county_Tama',
       'county_Taylor', 'county_Union', 'county_Van Buren',
       'county_Wapello', 'county_Warren', 'county_Washington',
       'county_Wayne', 'county_Webster', 'county_Winnebago',
       'county_Winneshiek', 'county_Woodbury', 'county_Worth',
       'county_Wright']]
y_mask = county_df[mask]['total_sales']
predict_mask = model.predict(X_mask)

# Plot the fit
plt.scatter(X['total_bottles'], y, color = "b", marker = "o")
plt.plot(X['total_bottles'],predictions, linewidth = 1, color = "w")
plt.scatter(X_mask['total_bottles'],predict_mask, color = 'r')
plt.xlabel("Total Bottles Sold Per Store")
plt.ylabel("Total Sales Per Store")
plt.xlim(0,30000)
plt.ylim(0,450000)
plt.show()
print "Alpha: ", model.alpha_
print "R-squared: ", model.score(X,y)
print "MAE: ", sklearn.metrics.mean_absolute_error(y,predictions)
```


![png](https://github.com/teresaborcuch/teresaborcuch.github.io/blob/master/images/Project_3_Iowa_liquor_sales_files/Project_3_Iowa_liquor_sales_24_0.png?raw=true)


    Alpha:  0.1
    R-squared:  0.962005861656
    MAE:  3842.92955834



```python
# Get coefficients
coef = model.coef_
var = []
coefs = []
for i in range(0,101):
    var.append(X.columns[i])
    coefs.append(coef[i])
model_summary = pd.DataFrame(columns = ['variable','coefficient'])
model_summary['variable'] =var
model_summary['coefficient'] = coefs
model_summary = model_summary.sort_values('coefficient', ascending = False)
model_summary.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variable</th>
      <th>coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>26</th>
      <td>county_Dallas</td>
      <td>6799.553635</td>
    </tr>
    <tr>
      <th>37</th>
      <td>county_Fremont</td>
      <td>3052.469705</td>
    </tr>
    <tr>
      <th>15</th>
      <td>county_Carroll</td>
      <td>2327.995402</td>
    </tr>
    <tr>
      <th>14</th>
      <td>county_Calhoun</td>
      <td>2216.967505</td>
    </tr>
    <tr>
      <th>13</th>
      <td>county_Butler</td>
      <td>1977.720222</td>
    </tr>
  </tbody>
</table>
</div>



According to the model, having a store in Dallas or Fremont counties is the best predictor of high sales. (Stores in those counties have higher sales than stores in other counties selling the same number of bottles.)

### Compare Model Predictions for Dallas County and Polk County


```python
mask = (sales['county']=='Dallas')
sales = sales.sort_values('total_bottles')
X = sales[mask][['total_bottles','mean_ppb']]
y = sales[mask]['total_sales']
lm = linear_model.LassoCV(normalize = True)
dallas_model = lm.fit(X,y)
predictions = dallas_model.predict(X)

plt.scatter(X['total_bottles'],y)
plt.plot(X['total_bottles'],predictions)
plt.xlabel("Total Bottles Sold")
plt.ylabel("Total Sales")
plt.title("Bottles Sold vs Total Sales for Dallas County")
plt.xlim(0,12000)
plt.ylim(0,300000)
plt.show()

print "R-squared: ", dallas_model.score(X,y)
```

    /Users/teresaborcuch/anaconda2/lib/python2.7/site-packages/ipykernel/__main__.py:3: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      app.launch_new_instance()
    /Users/teresaborcuch/anaconda2/lib/python2.7/site-packages/ipykernel/__main__.py:4: UserWarning: Boolean Series key will be reindexed to match DataFrame index.



![png](https://github.com/teresaborcuch/teresaborcuch.github.io/blob/master/images/Project_3_Iowa_liquor_sales_files/Project_3_Iowa_liquor_sales_28_1.png?raw=true)


    R-squared:  0.966167634188


In Dallas County, selling 6000 bottles in a year is predicted to yield over \$100,000 in total sales. Compare the predictions for Polk County, which had a lower average sales per store, though more stores overall:


```python
mask = sales['county']=='Polk'
sales = sales.sort_values('total_bottles')
X = sales[mask][['total_bottles','mean_ppb']]
y = sales[mask]['total_sales']
lm = linear_model.LassoCV(normalize = True)
model = lm.fit(X,y)
predictions = model.predict(X)

plt.scatter(X['total_bottles'],y)
plt.plot(X['total_bottles'],predictions)
plt.xlabel("Total Bottles Sold")
plt.ylabel("Total Sales")
plt.title("Bottles Sold vs Total Sales for Polk County")
plt.xlim(0,12000)
plt.ylim(0,300000)
plt.show()

print "R-squared: ", model.score(X,y)
print model.coef_
```

    /Users/teresaborcuch/anaconda2/lib/python2.7/site-packages/ipykernel/__main__.py:3: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      app.launch_new_instance()
    /Users/teresaborcuch/anaconda2/lib/python2.7/site-packages/ipykernel/__main__.py:4: UserWarning: Boolean Series key will be reindexed to match DataFrame index.



![png](https://github.com/teresaborcuch/teresaborcuch.github.io/blob/master/images/Project_3_Iowa_liquor_sales_files/Project_3_Iowa_liquor_sales_30_1.png?raw=true)


    R-squared:  0.943191759741
    [   14.0459037   1145.15412331]


In Polk County, selling 6,000 bottles yields closer to \$80,000 in total sales.

## Conclusions
After modeling the effects of county on liquor stores' total sales, I recommend that the liquor store owner open a new store in Dallas County. Excluding the extreme outliers in Polk County, stores in Dallas County are in the top five for mean sales, despite Dallas County ranking 10th in population. I will summarize the total sales for the stores with the highest total sales in 2015 in Dallas County, as well as the model's predictions for these stores' total sales in the future. The liquor store owner can use these predictions to estimate his potential total sales for the new store based on predicted bottle sales.


```python
# Summarize the best performing stores in Dallas County
top_sellers = sales[sales['county']=='Dallas'].sort_values('total_sales', ascending = False)
top_sellers = top_sellers[['store_num','zip_code','county','city','total_sales','margin','total_bottles', 'mean_ppb']]
```


```python
# Summarize the model's predictions for these stores in the future
X = top_sellers[['total_bottles','mean_ppb']]
y = top_sellers['total_sales']
predictions = dallas_model.predict(X)
top_sellers['Predicted Total Sales'] = predictions
top_sellers.head(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_num</th>
      <th>zip_code</th>
      <th>county</th>
      <th>city</th>
      <th>total_sales</th>
      <th>margin</th>
      <th>total_bottles</th>
      <th>mean_ppb</th>
      <th>Predicted Total Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>323</th>
      <td>3814</td>
      <td>50266</td>
      <td>Dallas</td>
      <td>WEST DES MOINES</td>
      <td>247417.42</td>
      <td>83004.48</td>
      <td>10472</td>
      <td>26.315333</td>
      <td>223268.666230</td>
    </tr>
    <tr>
      <th>148</th>
      <td>2665</td>
      <td>50263</td>
      <td>Dallas</td>
      <td>WAUKEE</td>
      <td>126114.56</td>
      <td>42181.47</td>
      <td>8428</td>
      <td>15.550958</td>
      <td>149223.054821</td>
    </tr>
    <tr>
      <th>894</th>
      <td>4678</td>
      <td>50003</td>
      <td>Dallas</td>
      <td>ADEL</td>
      <td>30260.71</td>
      <td>10114.41</td>
      <td>2040</td>
      <td>12.985000</td>
      <td>35866.922733</td>
    </tr>
    <tr>
      <th>108</th>
      <td>2612</td>
      <td>50220</td>
      <td>Dallas</td>
      <td>PERRY</td>
      <td>22949.94</td>
      <td>7664.81</td>
      <td>1854</td>
      <td>13.235988</td>
      <td>33803.651583</td>
    </tr>
    <tr>
      <th>1288</th>
      <td>5123</td>
      <td>50325</td>
      <td>Dallas</td>
      <td>CLIVE</td>
      <td>18253.13</td>
      <td>6105.85</td>
      <td>1239</td>
      <td>15.253723</td>
      <td>31494.184088</td>
    </tr>
    <tr>
      <th>474</th>
      <td>4137</td>
      <td>50263</td>
      <td>Dallas</td>
      <td>WAUKEE</td>
      <td>15861.67</td>
      <td>5316.17</td>
      <td>1148</td>
      <td>14.275374</td>
      <td>26301.515869</td>
    </tr>
    <tr>
      <th>641</th>
      <td>4384</td>
      <td>50003</td>
      <td>Dallas</td>
      <td>ADEL</td>
      <td>7306.21</td>
      <td>2443.86</td>
      <td>794</td>
      <td>11.168800</td>
      <td>8758.124125</td>
    </tr>
    <tr>
      <th>660</th>
      <td>4411</td>
      <td>50069</td>
      <td>Dallas</td>
      <td>DE SOTO</td>
      <td>6347.84</td>
      <td>2128.80</td>
      <td>519</td>
      <td>13.188265</td>
      <td>11969.751095</td>
    </tr>
    <tr>
      <th>1062</th>
      <td>4868</td>
      <td>50263</td>
      <td>Dallas</td>
      <td>WAUKEE</td>
      <td>5807.01</td>
      <td>1939.58</td>
      <td>450</td>
      <td>12.871452</td>
      <td>9647.059939</td>
    </tr>
    <tr>
      <th>637</th>
      <td>4378</td>
      <td>50263</td>
      <td>Dallas</td>
      <td>WAUKEE</td>
      <td>5089.54</td>
      <td>1697.68</td>
      <td>508</td>
      <td>11.592195</td>
      <td>5727.906964</td>
    </tr>
  </tbody>
</table>
</div>
