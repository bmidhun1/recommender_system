
# coding: utf-8

# In[1]:


# @hidden_cell
# The project token is an authorization token that is used to access project resources like data sources, connections, and used by platform APIs.
from project_lib import Project
project = Project(project_id='bfe99726-6cd1-4f12-b8e8-3e10f2ce7d52', project_access_token='p-2dda694dc92cbb2f1bc19052d6e747ec77d2059e')
pc = project.project_context


# In[2]:


get_ipython().system(u'pip install turicreate')


# In[3]:


get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')

import pandas as pd
import numpy as np
import time
import turicreate as tc
from sklearn.model_selection import train_test_split

import sys
sys.path.append("..")


# In[4]:



import types
import pandas as pd
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share your notebook.
client_83b43f68eb9348f8a7795f9fdefa8efd = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='evIZhEfbEBT4IAdtnXwujwwzLamRg7WUezb3jT9wRVXT',
    ibm_auth_endpoint="https://iam.bluemix.net/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3-api.us-geo.objectstorage.service.networklayer.com')

body = client_83b43f68eb9348f8a7795f9fdefa8efd.get_object(Bucket='recommendationsystemforpurchaseda-donotdelete-pr-l0bytgn7jogfsi',Key='train_5UKooLv.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df_data_1 = pd.read_csv(body)
df_data_1.head()
transactions = df_data_1[['CustomerID','StockCode']]
transactions = transactions.rename(columns={'CustomerID': 'customerId', 'StockCode':'products'})
transactions.head()



# In[5]:



body = client_83b43f68eb9348f8a7795f9fdefa8efd.get_object(Bucket='recommendationsystemforpurchaseda-donotdelete-pr-l0bytgn7jogfsi',Key='test_J1hm2KQ.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df_data_2 = pd.read_csv(body)
df_data_2.head()
customers=df_data_2[['CustomerID']]
customers=customers.drop_duplicates('CustomerID')
customers = customers.rename(columns={'CustomerID': 'customerId'})
customers.head()


# In[6]:


print(customers.shape)
customers.head()


# In[7]:


print(transactions.shape)
transactions.head()


# ## 3. Data preparation
# * Our goal here is to break down each list of items in the `products` column into rows and count the number of products bought by a user

# In[8]:


# example 1: split product items
transactions['products'] = transactions['products'].apply(lambda x: [i for i in x.split('|')])
transactions.head(10).set_index('customerId')['products'].apply(pd.Series).reset_index()


# In[9]:


# example 2: organize a given table into a dataframe with customerId, single productId, and purchase count
pd.melt(transactions.head(2).set_index('customerId')['products'].apply(pd.Series).reset_index(), 
             id_vars=['customerId'],
             value_name='products') \
    .dropna().drop(['variable'], axis=1) \
    .groupby(['customerId', 'products']) \
    .agg({'products': 'count'}) \
    .rename(columns={'products': 'purchase_count'}) \
    .reset_index() \
    .rename(columns={'products': 'productId'})


# ### 3.1. Create data with user, item, and target field
# * This table will be an input for our modeling later
#     * In this case, our user is `customerId`, `productId`, and `purchase_count`

# In[10]:


s=time.time()

data = pd.melt(transactions.set_index('customerId')['products'].apply(pd.Series).reset_index(), 
             id_vars=['customerId'],
             value_name='products') \
    .dropna().drop(['variable'], axis=1) \
    .groupby(['customerId', 'products']) \
    .agg({'products': 'count'}) \
    .rename(columns={'products': 'purchase_count'}) \
    .reset_index() \
    .rename(columns={'products': 'productId'})
# data['productId'] = data['productId'].astype(np.int64)

print("Execution time:", round((time.time()-s)/60,2), "minutes")


# In[11]:


print(data.shape)
data.head(40)


# ### 3.2. Create dummy
# * Dummy for marking whether a customer bought that item or not.
# * If one buys an item, then `purchase_dummy` are marked as 1
# * Why create a dummy instead of normalizing it, you ask?
#     * Normalizing the purchase count, say by each user, would not work because customers may have different buying frequency don't have the same taste
#     * However, we can normalize items by purchase frequency across all users, which is done in section 3.3. below.

# In[12]:


def create_data_dummy(data):
    data_dummy = data.copy()
    data_dummy['purchase_dummy'] = 1
    return data_dummy


# In[13]:


data_dummy = create_data_dummy(data)


# ### 3.3. Normalize item values across users
# * To do this, we normalize purchase frequency of each item across users by first creating a user-item matrix as follows

# In[14]:


df_matrix = pd.pivot_table(data, values='purchase_count', index='customerId', columns='productId')
df_matrix.head()


# In[15]:


(df_matrix.shape)


# In[16]:


df_matrix_norm = (df_matrix-df_matrix.min())/(df_matrix.max()-df_matrix.min())
print(df_matrix_norm.shape)
df_matrix_norm.head()


# In[17]:


# create a table for input to the modeling

d = df_matrix_norm.reset_index()
d.index.names = ['scaled_purchase_freq']
data_norm = pd.melt(d, id_vars=['customerId'], value_name='scaled_purchase_freq').dropna()
print(data_norm.shape)
data_norm.head()


# #### Define a function for normalizing data

# In[18]:


def normalize_data(data):
    df_matrix = pd.pivot_table(data, values='purchase_count', index='customerId', columns='productId')
    df_matrix_norm = (df_matrix-df_matrix.min())/(df_matrix.max()-df_matrix.min())
    d = df_matrix_norm.reset_index()
    d.index.names = ['scaled_purchase_freq']
    return pd.melt(d, id_vars=['customerId'], value_name='scaled_purchase_freq').dropna()


# * We can normalize the their purchase history, from 0-1 (with 1 being the most number of purchase for an item and 0 being 0 purchase count for that item).

# ## 4. Split train and test set
# * Splitting the data into training and testing sets is an important part of evaluating predictive modeling, in this case a collaborative filtering model. Typically, we use a larger portion of the data for training and a smaller portion for testing. 
# * We use 80:20 ratio for our train-test set size.
# * Our training portion will be used to develop a predictive model, while the other to evaluate the model's performance.
# * Now that we have three datasets with purchase counts, purchase dummy, and scaled purchase counts, we would like to split each.

# In[19]:


train, test = train_test_split(data, test_size = .2)
print(train.shape, test.shape)


# In[20]:


# Using turicreate library, we convert dataframe to SFrame - this will be useful in the modeling part

train_data = tc.SFrame(train)
test_data = tc.SFrame(test)


# In[21]:


train_data


# In[22]:


test_data


# #### Define a `split_data` function for splitting data to training and test set

# In[23]:


# We can define a function for this step as follows

def split_data(data):
    '''
    Splits dataset into training and test set.
    
    Args:
        data (pandas.DataFrame)
        
    Returns
        train_data (tc.SFrame)
        test_data (tc.SFrame)
    '''
    train, test = train_test_split(data, test_size = .2)
    train_data = tc.SFrame(train)
    test_data = tc.SFrame(test)
    return train_data, test_data


# In[24]:


# # lets try with both dummy table and scaled/normalized purchase table

train_data_dummy, test_data_dummy = split_data(data_dummy)
train_data_norm, test_data_norm = split_data(data_norm)


# ## 5. Baseline Model
# Before running a more complicated approach such as collaborative filtering, we would like to use a baseline model to compare and evaluate models. Since baseline typically uses a very simple approach, techniques used beyond this approach should be chosen if they show relatively better accuracy and complexity.
# 
# ### 5.1. Using a Popularity model as a baseline
# * The popularity model takes the most popular items for recommendation. These items are products with the highest number of sells across customers.
# * We use `turicreate` library for running and evaluating both baseline and collaborative filtering models below
# * Training data is used for model selection
# 
# #### Using purchase counts

# In[25]:


# variables to define field names
user_id = 'customerId'
item_id = 'productId'
target = 'purchase_count'
users_to_recommend = list(transactions[user_id])
n_rec = 10 # number of items to recommend
n_display = 30


# In[26]:


popularity_model = tc.popularity_recommender.create(train_data, 
                                                    user_id=user_id, 
                                                    item_id=item_id, 
                                                    target=target)


# In[27]:


# Get recommendations for a list of users to recommend (from customers file)
# Printed below is head / top 30 rows for first 3 customers with 10 recommendations each

popularity_recomm = popularity_model.recommend(users=users_to_recommend, k=n_rec)
popularity_recomm.print_rows(n_display)


# #### Define a `model` function for model selection

# In[28]:


# Since turicreate is very accessible library, we can define a model selection function as below

def model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display):
    if name == 'popularity':
        model = tc.popularity_recommender.create(train_data, 
                                                    user_id=user_id, 
                                                    item_id=item_id, 
                                                    target=target)
    elif name == 'cosine':
        model = tc.item_similarity_recommender.create(train_data, 
                                                    user_id=user_id, 
                                                    item_id=item_id, 
                                                    target=target, 
                                                    similarity_type='cosine')
    elif name == 'pearson':
        model = tc.item_similarity_recommender.create(train_data, 
                                                    user_id=user_id, 
                                                    item_id=item_id, 
                                                    target=target, 
                                                    similarity_type='pearson')
        
    recom = model.recommend(users=users_to_recommend, k=n_rec)
    recom.print_rows(n_display)
    return model


# In[29]:


# variables to define field names
# constant variables include:
user_id = 'customerId'
item_id = 'productId'
users_to_recommend = list(customers[user_id])
n_rec = 10 # number of items to recommend
n_display = 30 # to print the head / first few rows in a defined dataset


# #### Using purchase dummy

# In[30]:


# these variables will change accordingly
name = 'popularity'
target = 'purchase_dummy'
pop_dummy = model(train_data_dummy, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)


# #### Using normalized purchase count

# In[31]:


name = 'popularity'
target = 'scaled_purchase_freq'
pop_norm = model(train_data_norm, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)


# #### Notes
# * Once we created the model, we predicted the recommendation items using scores by popularity. As you can tell for each model results above, the rows show the first 30 records from 1000 users with 10 recommendations. These 30 records include 3 users and their recommended items, along with score and descending ranks. 
# * In the result, although different models have different recommendation list, each user is recommended the same list of 10 items. This is because popularity is calculated by taking the most popular items across all users.
# * If a grouping example below, products 132, 248, 37, and 34 are the most popular (best-selling) across customers. Using their purchase counts divided by the number of customers, we see that these products are at least bought 3 times on average in the training set of transactions (same as the first popularity measure on `purchase_count` variable)

# In[32]:


train.groupby(by=item_id)['purchase_count'].mean().sort_values(ascending=False).head(20)


# ## 6. Collaborative Filtering Model
# 
# * In collaborative filtering, we would recommend items based on how similar users purchase items. For instance, if customer 1 and customer 2 bought similar items, e.g. 1 bought X, Y, Z and 2 bought X, Y, we would recommend an item Z to customer 2.
# 
# * To define similarity across users, we use the following steps:
#     1. Create a user-item matrix, where index values represent unique customer IDs and column values represent unique product IDs
#     
#     2. Create an item-to-item similarity matrix. The idea is to calculate how similar a product is to another product. There are a number of ways of calculating this. In steps 6.1 and 6.2, we use cosine and pearson similarity measure, respectively.  
#     
#         * To calculate similarity between products X and Y, look at all customers who have rated both these items. For example, both X and Y have been rated by customers 1 and 2. 
#         * We then create two item-vectors, v1 for item X and v2 for item Y, in the user-space of (1, 2) and then find the `cosine` or `pearson` angle/distance between these vectors. A zero angle or overlapping vectors with cosine value of 1 means total similarity (or per user, across all items, there is same rating) and an angle of 90 degree would mean cosine of 0 or no similarity.
#         
#     3. For each customer, we then predict his likelihood to buy a product (or his purchase counts) for products that he had not bought. 
#     
#         * For our example, we will calculate rating for user 2 in the case of item Z (target item). To calculate this we weigh the just-calculated similarity-measure between the target item and other items that customer has already bought. The weighing factor is the purchase counts given by the user to items already bought by him. 
#         * We then scale this weighted sum with the sum of similarity-measures so that the calculated rating remains within a predefined limits. Thus, the predicted rating for item Z for user 2 would be calculated using similarity measures.
# 
# * While I wrote python scripts for all the process including finding similarity using python scripts (which can be found in `scripts` folder, we can use `turicreate` library for now to capture different measures like using `cosine` and `pearson` distance, and evaluate the best model.

# ### 6.1. `Cosine` similarity
# * Similarity is the cosine of the angle between the 2 vectors of the item vectors of A and B
# * It is defined by the following formula
# ![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTnRHSAx1c084UXF2wIHYwaHJLmq2qKtNk_YIv3RjHUO00xwlkt)
# * Closer the vectors, smaller will be the angle and larger the cosine

# #### Using purchase count

# In[33]:


# these variables will change accordingly
name = 'cosine'
target = 'purchase_count'
cos = model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)


# #### Using purchase dummy

# In[34]:


# these variables will change accordingly
name = 'cosine'
target = 'purchase_dummy'
cos_dummy = model(train_data_dummy, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)


# #### Using normalized purchase count

# In[35]:


name = 'cosine'
target = 'scaled_purchase_freq'
cos_norm = model(train_data_norm, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)


# ### 6.2. `Pearson` similarity
# * Similarity is the pearson coefficient between the two vectors.
# * It is defined by the following formula
# ![](http://critical-numbers.group.shef.ac.uk/glossary/images/correlationKT1.png)

# #### Using purchase count

# In[36]:


# these variables will change accordingly
name = 'pearson'
target = 'purchase_count'
pear = model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)


# #### Using purchase dummy

# In[37]:


# these variables will change accordingly
name = 'pearson'
target = 'purchase_dummy'
pear_dummy = model(train_data_dummy, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)


# #### Using normalized purchase count

# In[38]:


name = 'pearson'
target = 'scaled_purchase_freq'
pear_norm = model(train_data_norm, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)


# #### Note
# * In collaborative filtering above, we used two approaches: cosine and pearson distance. We also got to apply them to three training datasets with normal counts, dummy, or normalized counts of items purchase.
# * We can see that the recommendations are different for each user. This suggests that personalization does exist. 
# * But how good is this model compared to the baseline, and to each other? We need some means of evaluating a recommendation engine. Lets focus on that in the next section.

# ## 7. Model Evaluation
# For evaluating recommendation engines, we can use the concept of precision-recall.
# 
# * RMSE (Root Mean Squared Errors)
#     * Measures the error of predicted values
#     * Lesser the RMSE value, better the recommendations
# * Recall
#     * What percentage of products that a user buys are actually recommended?
#     * If a customer buys 5 products and the recommendation decided to show 3 of them, then the recall is 0.6
# * Precision
#     * Out of all the recommended items, how many the user actually liked?
#     * If 5 products were recommended to the customer out of which he buys 4 of them, then precision is 0.8
#     
# * Why are both recall and precision important?
#     * Consider a case where we recommend all products, so our customers will surely cover the items that they liked and bought. In this case, we have 100% recall! Does this mean our model is good?
#     * We have to consider precision. If we recommend 300 items but user likes and buys only 3 of them, then precision is 0.1%! This very low precision indicates that the model is not great, despite their excellent recall.
#     * So our aim has to be optimizing both recall and precision (to be close to 1 as possible).
# 
# Lets compare all the models we have built based on precision-recall characteristics:

# In[39]:


# create initial callable variables

models_w_counts = [popularity_model, cos, pear]
models_w_dummy = [pop_dummy, cos_dummy, pear_dummy]
models_w_norm = [pop_norm, cos_norm, pear_norm]

names_w_counts = ['Popularity Model on Purchase Counts', 'Cosine Similarity on Purchase Counts', 'Pearson Similarity on Purchase Counts']
names_w_dummy = ['Popularity Model on Purchase Dummy', 'Cosine Similarity on Purchase Dummy', 'Pearson Similarity on Purchase Dummy']
names_w_norm = ['Popularity Model on Scaled Purchase Counts', 'Cosine Similarity on Scaled Purchase Counts', 'Pearson Similarity on Scaled Purchase Counts']


# #### Models on purchase counts

# In[40]:


eval_counts = tc.recommender.util.compare_models(test_data, models_w_counts, model_names=names_w_counts)


# #### Models on purchase dummy

# In[41]:


eval_dummy = tc.recommender.util.compare_models(test_data_dummy, models_w_dummy, model_names=names_w_dummy)


# #### Models on normalized purchase frequency

# In[42]:


eval_norm = tc.recommender.util.compare_models(test_data_norm, models_w_norm, model_names=names_w_norm)


# ## 8. Model Selection
# ### 8.1. Evaluation summary
# * Based on RMSE
# 
# 
#     1. Popularity on purchase counts: 1.1111750034210488
#     2. Cosine similarity on purchase counts: 1.9230643981653215
#     3. Pearson similarity on purchase counts: 1.9231102838192284
#     
#     4. Popularity on purchase dummy: 0.9697374361161925
#     5. Cosine similarity on purchase dummy: 0.9697509978436404
#     6. Pearson similarity on purchase dummy: 0.9697745320187097
#     
#     7. Popularity on scaled purchase counts: 0.16230660626840343
#     8. Cosine similarity on scaled purchase counts: 0.16229800354111104
#     9. Pearson similarity on scaled purchase counts: 0.1622982668334026
#     
# * Based on Precision and Recall
# ![](../images/model_comparisons.png)
# 

# #### Notes
# 
# * Popularity v. Collaborative Filtering: We can see that the collaborative filtering algorithms work better than popularity model for purchase counts. Indeed, popularity model doesn’t give any personalizations as it only gives the same list of recommended items to every user.
# * Precision and recall: Looking at the summary above, we see that the precision and recall for Purchase Counts > Purchase Dummy > Normalized Purchase Counts. However, because the recommendation scores for the normalized purchase data is zero and constant, we choose the dummy. In fact, the RMSE isn’t much different between models on the dummy and those on the normalized data.
# * RMSE: Since RMSE is higher using pearson distance thancosine, we would choose model the smaller mean squared errors, which in this case would be cosine.
# Therefore, we select the Cosine similarity on Purchase Dummy approach as our final model.

# ## 8. Final Output
# * In this step, we would like to manipulate format for recommendation output to one we can export to csv, and also a function that will return recommendation list given a customer ID.
# * We need to first rerun the model using the whole dataset, as we came to a final model using train data and evaluated with test set.

# In[43]:


users_to_recommend = list(customers[user_id])

final_model = tc.item_similarity_recommender.create(tc.SFrame(data), 
                                            user_id=user_id, 
                                            item_id=item_id, 
                                            target='purchase_count', 
                                            similarity_type='cosine')

recom = final_model.recommend(users=users_to_recommend, k=n_rec)
recom.print_rows(n_display)


# ### 8.1. CSV output file

# In[44]:


df_rec = recom.to_dataframe()
print(df_rec.shape)
df_rec.head(10)


# In[52]:


# df_rec1 = df_rec.to_dataframe()
import json
df2_unique = df_rec.drop_duplicates(subset='customerId')
final_recom = df2_unique.filter(['customerId',str('productId')])
final_recom = final_recom.rename(columns={'customerId': 'CustomerID', 'productId':'Items'})
final_recom['Items'] = final_recom['Items'].apply(lambda x: "'" + str(x) + "'")
final_recom['Items'] = '[' + final_recom['Items'].astype(str) + ']'


# In[58]:


project.save_data(data=final_recom.to_csv(index=False,sep=','),file_name='recommended_purchase.csv',overwrite=True)


# In[54]:


print(final_recom.shape)
final_recom.head(20)


# In[46]:


df_rec['recommendedProducts'] = df_rec.groupby([user_id])[item_id].transform(lambda x: '|'.join(x.astype(str)))
df_output = df_rec[['customerId', 'recommendedProducts']].drop_duplicates().sort_values('customerId').set_index('customerId')


# #### Define a function to create a desired output

# In[47]:


def create_output(model, users_to_recommend, n_rec, print_csv=True):
    recomendation = model.recommend(users=users_to_recommend, k=n_rec)
    df_rec = recomendation.to_dataframe()
    df_rec['recommendedProducts'] = df_rec.groupby([user_id])[item_id]         .transform(lambda x: '|'.join(x.astype(str)))
    df_output = df_rec[['customerId', 'recommendedProducts']].drop_duplicates()         .sort_values('customerId').set_index('customerId')
    if print_csv:
        df_output.to_csv('recommendation_output.csv')
    return df_output


# In[48]:


df_output = create_output(pear_norm, users_to_recommend, n_rec, print_csv=True)
rec_output=df_output.drop_duplicates()
project.save_data(data=rec_output.to_csv(index=False),file_name='recommendation_output.csv',overwrite=True)
print(df_output.shape)
df_output.head(12)


# ### 8.2. Customer recommendation function

# In[49]:


def customer_recomendation(customer_id):
    if customer_id not in df_output.index:
        print('Customer not found.')
        return customer_id
    return df_output.loc[customer_id]


# In[50]:


customer_recomendation(19980)


# In[51]:


customer_recomendation(15390)


# ## Summary
# In this exercise, we were able to traverse a step-by-step process for making recommendations to customers. We used Collaborative Filtering approaches with `cosine` and `pearson` measure and compare the models with our baseline popularity model. We also prepared three sets of data that include regular buying count, buying dummy, as well as normalized purchase frequency as our target variable. Using RMSE, precision and recall, we evaluated our models and observed the impact of personalization. Finally, we selected the Cosine approach in dummy purchase data. 
