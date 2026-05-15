# Original file is located at https://colab.research.google.com/drive/17GTa-8J_iuTWo2GDtrW5nFVWd1A0v7tE


import pandas as pd
#!gdown 1yxgr0Qj3TiXRehYa0PED1t4zIga9gdY5 
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

import matplotlib.pyplot as plt
import seaborn as sns
import math

import warnings
warnings.filterwarnings('ignore')

"""### **1. Data Loading:**"""

data = pd.read_excel("/content/churn_prediction.xlsx")
data.shape

data.info()

data.head(5)

"""### **2. Data Preprocessing:**"""

numeric_cols = data.select_dtypes(include=['float','int64']).columns
numeric_cols

data.columns = data.columns.str.lower()

# check imbalance
imb_data = data.groupby('churn')['customerid'].count().reset_index()
imb_data['%'] = imb_data['customerid']/sum(imb_data['customerid'])
imb_data

"""#### **1.1 Missing values:**"""

import missingno as msno
msno.matrix(data)

list_null = data.columns[data.isnull().any()].tolist()
data[list_null].dtypes

#check % dữ liệu bị thiếu ở cấp row, có nên drop row hay không?
print(data.isna().any(axis=1).mean() * 100)

# >30% không nên drop
# <5-10% có thể drop an toàn

# 32% nên sẽ check tiếp distribution to decide we will replace with median or 0
for i in list_null:
    fig, ax = plt.subplots(figsize=(5, 2))
    plot_df = data.groupby(i)[['customerid']].count().reset_index()
    sns.barplot(data = plot_df,x=plot_df[i],y= plot_df['customerid'],ax=ax, color='salmon')
    plt.show()

"""-> All will be replace by median, except **hourspendonap** and **couponused** by 0"""

def replace_func(list_columns):
    for i in list_columns:
        if i in ['tenure','warehousetohome','orderamounthikefromlastyear','ordercount','daysincelastorder']:
            median = data[i].median()
            data[i].fillna(median, inplace=True)
        else:
            data[i].fillna(0, inplace=True)

replace_func(list_null)

msno.matrix(data)

"""#### **1.2 Duplicated values:**"""

data.duplicated().any()

"""#### **1.3 Check distribution to dectect same meaning values:**"""

list_obj = data.loc[:, data.dtypes == object].columns.tolist()

for col in list_obj:
    dist = data[col].value_counts(normalize=True) * 100

    print(f"\n{col}")
    for k, v in dist.items():
        print(f"{k:<20} ~{v:.1f}%")

#Replace the same meaning values:
data['preferredlogindevice'] = data['preferredlogindevice'].replace({'Mobile Phone':'Phone'})
data['preferredpaymentmode'] = data['preferredpaymentmode'].replace({'CC':'Credit Card','COD':'Cash on Delivery'})
data['preferedordercat'] = data['preferedordercat'].replace({'Mobile Phone':'Phone'})

for j in list_obj:
  print(f"Unique values of {j}: {data[j].unique()}")

"""Label Column (Target column): Churn (0 = No Churn, 1 = Churn)

Other Column Categories

1. Customer Info:
CustomerID, Gender, MaritalStatus, CityTier

2. Behavioral Features:
Tenure, HourSpendOnApp, OrderCount, CouponUsed, DaySinceLastOrder

3. Transactional Features:
CashbackAmount, OrderAmountHikeFromlastYear, PreferedOrderCat

4. Service/Experience Features:
SatisfactionScore, Complain, PreferredPaymentMode, PreferredLoginDevice

### **3. Features Tranforming - Encoding:**
"""

data = data.drop('customerid', axis=1)

data_encoded = pd.get_dummies(data, drop_first=True)
data_encoded

"""### *Question 1:*

Các hành vi của những người dùng đã rời bỏ là gì? Bạn có đề xuất gì cho công ty để giảm số lượng người dùng rời bỏ?
"""

from sklearn.ensemble import RandomForestClassifier

# X, y
X = data_encoded.drop('churn', axis=1)
y = data_encoded['churn']

# model
rf = RandomForestClassifier(n_estimators=100,random_state=42)

# fit trên TOÀN BỘ data (EDA purpose)
rf.fit(X, y)

# lấy importance
importances = rf.feature_importances_

# convert thành DataFrame cho dễ nhìn
import pandas as pd

feat_imp = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
}).sort_values(by='importance', ascending=False)

print(feat_imp)

feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(X.columns, rf.feature_importances_):
    feats[feature] = importance #add the name/value pair

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
importances = importances.sort_values(by='Gini-importance', ascending=True)

importances = importances.reset_index()

# Create bar chart
plt.figure(figsize=(10, 10))
plt.barh(importances.tail(20)['index'][:20], importances.tail(20)['Gini-importance'], color='salmon')

plt.title('Feature Important')

# Show plot
plt.show()

"""CHOOSE TOP 4 FEATURES TO ANALYSE -> EDA"""

df = data_encoded

#Show Distribution of Tenure, CashbackAmount, WarehousetoHome, Complain

features = ['tenure', 'cashbackamount', 'warehousetohome','complain']

plt.figure(figsize=(12,10))

for i, col in enumerate(features, 1):
    plt.subplot(3, 2, i)
    sns.boxplot(data=df, x='churn', y=col, palette=['#2ecc71','#e74c3c'])
    plt.title(col)

plt.tight_layout()
plt.show()

df.groupby('churn')[features].mean()

"""-> không có sự khác biệt nhiều giữa churn và not churn đối với warehousetohome
-> loại bỏ yếu tố này ra khỏi mô hình train để tránh bias.

SUMMARY
1. Tenure: Khách mới rất dễ churn
-> Người dùng chưa gắn bó -> dễ rời bỏ

Hành động:
* Onboarding tốt hơn
* Ưu đãi cho khách mới
2. Complain: Người đã complain có khả năng churn gấp ~2 lần
-> Trải nghiệm xấu -> churn ngay

Hành động:
* Ưu tiên xử lý complain
* Follow-up khách hàng
3. CashbackAmount: Cashback thấp -> dễ churn hơn
-> Incentive ảnh hưởng retention

Hành động:
* Tăng cashback cho nhóm risk churn

### **4. Model Training and Evaluation:**

### *Question 2:* *Supervised Learning*
Xây dựng mô hình Machine Learning để dự đoán người dùng rời bỏ.
*   Remove warehousetohome column
*   Apply Random Forest model and fine tuning
"""

data_model = data.drop(columns = 'warehousetohome')

data_model.head(5)

cate_columns = data_model.loc[:, data_model.dtypes == object].columns.tolist()
encoded_df = pd.get_dummies(data_model, columns = cate_columns,drop_first=True)
encoded_df.shape

encoded_df.head(3)

# Split Train/Test
X = encoded_df.drop('churn', axis=1)
y = encoded_df['churn']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Base Model
models = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "DecisionTree": DecisionTreeClassifier(max_depth=5),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
}

from sklearn.model_selection import cross_val_score

results = []

for name, model in models.items():
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring='balanced_accuracy'
    )

    results.append({
        "model": name,
        "balanced_accuracy_mean": scores.mean(),
        "balanced_accuracy_std": scores.std()
    })

results_df = pd.DataFrame(results).sort_values("balanced_accuracy_mean", ascending=False)
print(results_df)

"""-> RandomForest là lựa chọn tốt nhất trong số này.

IMPROVE MODEL -> HYPERPARAMETER TUNING
"""

# Enhance Random Forest model:

from sklearn.model_selection import GridSearchCV

# Define the parameter grid

clf_rand = RandomForestClassifier(random_state=0)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

# Use GridSearchCV to find the best parameters
grid_search = GridSearchCV(clf_rand, param_grid, cv=5, scoring='balanced_accuracy')

# Fit the model
grid_search.fit(X_train, y_train)

# Print the best parameters
print("Best Parameters: ", grid_search.best_params_)

# Evaluate the best model on the test set
best_clf = grid_search.best_estimator_
accuracy = best_clf.score(X_test, y_test)
print("Test set accuracy: ", accuracy)

# Re-apply model with new parameters:
from sklearn.metrics import balanced_accuracy_score
# 1. Scale dữ liệu
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(X_train)   # fit trên train
x_test_scaled  = scaler.transform(X_test)       # transform trên test

# 2. Re-apply model với best_params
best_params = grid_search.best_params_
clf_rand_after = RandomForestClassifier(**best_params, random_state=0)

# 3. Fit model trên train_scaled
clf_rand_after.fit(x_train_scaled, y_train)

# 4. Dự đoán
y_ranf_aft_train = clf_rand_after.predict(x_train_scaled)
y_ranf_aft_test  = clf_rand_after.predict(x_test_scaled)

# 5. Tính balanced accuracy
train_bal_acc = balanced_accuracy_score(y_train, y_ranf_aft_train)
test_bal_acc  = balanced_accuracy_score(y_test, y_ranf_aft_test)

print(f"Train balanced accuracy: {train_bal_acc:.4f}")
print(f"Test balanced accuracy:  {test_bal_acc:.4f}")

"""Trước tuning: RandomForest ≈ 0.893

Sau tuning: Test balanced accuracy ≈ 0.955
→ Tuning thực sự cải thiện mô hình, nâng accuracy ~6% trên test set.

Overfitting nhẹ trên train là bình thường với RandomForest. Test accuracy vẫn cao → model đáng tin.

-> use this model as final model

### *Question 3:* *Unsupervised Learning*
Dựa trên hành vi của những người dùng rời bỏ, công ty muốn đưa ra một số khuyến mãi đặc biệt cho họ.
 → Hãy phân khúc (segment) nhóm người dùng rời bỏ này thành các nhóm khác nhau. Sự khác biệt giữa các nhóm là gì?
(Gợi ý: Sử dụng KMeans để clustering trên tập dữ liệu có churn ==1)

* Use K-Means to clustering churn-users groups.
* Find the insight between the groups
"""

#Filter churned users only:
df_churn = data[data['churn'] == 1].drop(columns='churn')

#Transform data:
cate_columns = df_churn.loc[:, df_churn.dtypes == object].columns.tolist()
encoded_df = pd.get_dummies(df_churn, columns = cate_columns,drop_first=True)

# Apply normalization cho toàn bộ dữ liệu:

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
model=scaler.fit(encoded_df)
scaled_data=model.transform(encoded_df)
scaled_df = pd.DataFrame(scaled_data, columns = encoded_df.columns.tolist())

scaled_df.info()

# Dimension reduction:
pca = PCA(n_components=3)
pca.fit(scaled_df)
PCA_ds = pd.DataFrame(pca.transform(scaled_df), columns=(["PC1","PC2", "PC3"]))

explain_variance = []

for i in range(1, 10):
    pca_temp = PCA(n_components=i)
    pca_temp.fit(scaled_df)
    explain_variance.append(sum(pca_temp.explained_variance_ratio_))

PCA_ds.head()

pca.explained_variance_ratio_

sum(pca.explained_variance_ratio_)

sum(pca_temp.explained_variance_ratio_)

"""APPLY K-MEANS MODEL"""

# Choosing K

ss = []
max_clusters = 10
for i in range(1, max_clusters+1):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(PCA_ds)
    # Inertia method returns WCSS for that model
    ss.append(kmeans.inertia_)

# Plot the Elbow method
plt.figure(figsize=(10,5))
plt.plot(range(1, max_clusters+1), ss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

import seaborn as sns

# Visualize clusters in PCA space
temp_clusters = KMeans(n_clusters=4, random_state=42).fit_predict(PCA_ds)

sns.scatterplot(x='PC1', y='PC2', hue=temp_clusters, data=PCA_ds, palette='Set2')
plt.title('PCA 2D visualization of churn clusters')
plt.show()

# Apply K-Means (final)

kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(PCA_ds)

"""EVALUATING MODEL"""

#Silhouette Score:

from sklearn.metrics import silhouette_score

sil_score=silhouette_score(PCA_ds, clusters)
print(sil_score)

"""-> The silhouette score of 0.425 indicates a moderate cluster structure. While some overlap exists between clusters, the segmentation is still meaningful and suitable for further analysis.

"""

# Sau khi đã chạy KMeans
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(PCA_ds)

# Gắn vào data gốc đã encode
df_clustered = encoded_df.copy()
df_clustered['cluster'] = clusters

cluster_summary = df_clustered.groupby('cluster').mean()
cluster_summary

df_clustered['cluster'].value_counts()

"""SUMMARY:

The churned customers can be segmented into 4 distinct groups:

Cluster 0: Loyal but silently disengaging users

Cluster 1: New, deal-sensitive customers

Cluster 2: Low-engagement mobile users

Cluster 3: High-value but demanding customers

Each group churns for different reasons, requiring targeted retention strategies.

PHÂN TÍCH TỪNG CLUSTER:

1. Cluster 0 – “Moderate users, stable behavior”

Đặc điểm:
* Tenure: cao nhất (8.5) → dùng lâu
* Coupon used: cao (4.2)
* Order hike: thấp hơn cluster khác
* Complain: thấp nhất (0.42)
* Married nhiều (62%)

Insight:
Đây là nhóm khách hàng trung thành, ít phàn nàn, nhưng vẫn churn → dấu hiệu “rời bỏ thầm lặng”

Vấn đề:
Không phải vì giá hay trải nghiệm tệ. Có thể do thiếu engagement / đối thủ hấp dẫn hơn

Action:
Loyalty program. Personalized recommendation.
Giữ chân bằng trải nghiệm, không phải giảm giá



---


2. Cluster 1 – “New, deal-sensitive, phone buyers”

Đặc điểm:
* Tenure: rất thấp (2.5) → khách mới
* Coupon used: thấp (1.6)
* Prefer Phone: rất cao (0.89)
* Single nhiều (57%)

Insight:
Khách mới, chủ yếu mua điện thoại, không gắn bó → churn sớm

Vấn đề:
Không thấy đủ giá trị để quay lại. Có thể chỉ mua 1 lần

Action:
First-time user promotion. Bundle deal (phone + accessory). Retargeting ads


---



3. Cluster 2 – “Mobile-heavy, low engagement users”

Đặc điểm:
* Hours on app: thấp (~1.85)
* Devices: thấp (~3.38)
* Coupon used: rất thấp (~0.67)
* Prefer Mobile: rất cao (~0.82)

Insight:
Người dùng mobile nhưng ít tương tác → churn vì không engage

Vấn đề:
UX/app chưa hấp dẫn. Không có động lực quay lại

Action:
Push notification. App UX improvement. Gamification / reward



---


4. Cluster 3 – “High-value but demanding customers”

Đặc điểm:
* Warehouse to home: cao nhất (20) → delivery xa
* City tier: cao (2.8)
* Prefer Laptop: rất cao (0.61)
* E-wallet: cao (0.48)
* Complain: cao (0.56)

Insight:
Khách hàng giá trị cao nhưng khó tính / kỳ vọng cao

Vấn đề:
Delivery chậm / trải nghiệm chưa tốt

Dễ churn nếu không hài lòng

Action:
Faster delivery. Premium support. VIP service
"""

from sklearn.ensemble import RandomForestClassifier
import pandas as pd

X = df_clustered.drop(columns='cluster')
y = df_clustered['cluster']

clf = RandomForestClassifier(random_state=42)
clf.fit(X, y)

# Feature importance
feature_importances = pd.Series(
    clf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(feature_importances.head(10))

"""RandomForest cho thấy yếu tố phân biệt các cluster:

Top 5 feature:
1. cashbackamount → khách nào nhạy khuyến mãi
2. preferedordercat_Phone → loại sản phẩm khách mua
3. preferedordercat_Mobile → mobile engagement
4. ordercount → số đơn đã đặt
5. citytier → khu vực sinh sống

FINAL CONCLUSION:
* Cluster 1 (Phone-heavy, mới) → có feature importance là preferedordercat_Phone cao, xác nhận đúng với phân tích clustering.
* Cluster 2 (Mobile-heavy) → preferedordercat_Mobile top feature, cũng đúng với pattern cluster.
* Cluster 3 (High-value) → cashbackamount và ordercount cao, đúng với nhận định là “nhóm giá trị nhưng khó tính”.
* Cluster 0 (loyal) → tenure cao, ordercount trung bình → phù hợp với feature tenure và days since last order.

**These findings suggest that churn is not driven by a single factor, but rather by a combination of pricing sensitivity, product preference, and customer engagement.**

**The new and deal-sensitive group should be targeted with attractive promotions, the loyal group should be retained through enhanced experiences, the mobile-heavy group requires increased engagement, and the high-value group needs VIP services and personalized care.**

**This analysis enables the company to develop personalized promotion strategies for each segment, improving the retention of churned customers.**
"""