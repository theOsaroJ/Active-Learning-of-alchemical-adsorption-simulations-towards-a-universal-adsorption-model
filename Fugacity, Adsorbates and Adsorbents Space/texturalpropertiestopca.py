import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

data = pd.read_csv('database.csv')
X= data.drop(data.columns[0:21], axis=1)
X= X.drop(X.columns[6:],axis=1)

# create the feature matrix
# X = np.array([LPD, PLD, SA_grav, VF, PSSD, den])

scaler= StandardScaler()
X_std = scaler.fit_transform(X)

pca = PCA()
pca.fit(X_std)

# compute the cumulative explained variance
var_exp = pca.explained_variance_ratio_

# plot the cumulative explained variance using a line plot 
plt.figure(figsize=(8, 6))
plt.plot(range(1, 7), var_exp.cumsum(), marker='o', linestyle='--')
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')

# plot a line at n_components = 2 to estimate the 'elbow' in the plot and determine the explained variance
plt.axvline(x=2, color='black', linestyle='--')
plt.axhline(y=0.9125, color='black', linestyle='--')


# print the cummulative explained variance for all components
for i in range(1, 7):
    print('Explained variance of component %d: %.4f' %(i, var_exp.cumsum()[i-1]))


scores_pca =pca.transform(X_std)

wcss = []
for i in range(1, 7):
    kmeans_pca = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans_pca.fit(scores_pca)
    wcss.append(kmeans_pca.inertia_)


kmeans_pca = KMeans(n_clusters=3, init= 'k-means++', random_state=42)
kmeans_pca.fit(scores_pca)

df_pca_kmeans = pd.concat([X.reset_index(drop=True), pd.DataFrame(scores_pca)], axis = 1)
df_pca_kmeans.columns.values[-6:-4 ] = ['PC1', 'PC2']
df_pca_kmeans['Segment K-means PCA'] = kmeans_pca.labels_

df_pca_kmeans['Segment'] = df_pca_kmeans['Segment K-means PCA'].map({0:'first', 1:'second', 2:'third', 3:'fourth'
                                                                    ,4:'fifth', 5:'sixth', 6:'seventh', 7:'eight'})
x_axis= df_pca_kmeans['PC1']
y_axis = df_pca_kmeans['PC2']
df_pca_kmeans

plt.figure(figsize =(10,8))
sns.scatterplot(x=x_axis, y=y_axis, hue=df_pca_kmeans['Segment'], palette = ['g', 'r', 'c', 'm','b','k','purple'])

# add the scores to the original data
data['PC1'] = scores_pca[:, 0]
data['PC2'] = scores_pca[:, 1]

data.to_csv('database_pca.csv', index='None')
