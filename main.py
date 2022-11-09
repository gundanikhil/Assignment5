from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA


dataset = pd.read_csv('D:/datasets/CC.csv')
x = dataset.iloc[:,[1,2,3,4]]
y = dataset.iloc[:,-1]
print(x.shape, y.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x)

x_scaler = scaler.transform(x)
pca = PCA(2)
x_pca = pca.fit_transform(x_scaler)
df2 = pd.DataFrame(data=x_pca, columns=['principal component 1', 'principal component 2'])
finaldf = pd.concat([df2,dataset[['CUST_ID']]],axis=1)
print(finaldf.head())
print('First a part finished')


print(dataset["CUST_ID"].value_counts())


from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
scaler.fit(x)
X_scaled_array = scaler.transform(x)
X_scaled = pd.DataFrame(X_scaled_array, columns = x.columns)

from sklearn.cluster import KMeans
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.34,random_state=0)
nclusters = 3 # this is the k in kmeans
km = KMeans(n_clusters=nclusters)
km.fit(x)

y_cluster_kmeans = km.predict(x)
from sklearn.metrics import classification_report

print(classification_report(y, y_cluster_kmeans, zero_division=1))
print(confusion_matrix(y, y_cluster_kmeans))


train_accuracy = accuracy_score(y, y_cluster_kmeans)
print("\nAccuracy for our Training dataset with PCA:", train_accuracy)

from sklearn import metrics
score = metrics.silhouette_score(x, y_cluster_kmeans)
print('The Silhouette score is: ',score)

# predict the cluster for each testing data point
y_clus_test = km.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_clus_test, zero_division=1))
print(confusion_matrix(y_test, y_clus_test))

train_accuracy = accuracy_score(y_test, y_clus_test)
print("\nAccuracy for our Training dataset with PCA:", train_accuracy)

#Calculate sihouette Score
score = metrics.silhouette_score(X_test, y_clus_test)
print("Sihouette Score: ",score)


#3rd question
x = dataset.iloc[:,1:-1]
y = dataset.iloc[:,-1]
print(x.shape,y.shape)

print(dataset.head())

print(dataset.shape)

print(dataset['CUST_ID'].value_counts())

print(dataset.isnull().any())

print(dataset.fillna(dataset['CREDIT_LIMIT'].mean(), inplace = True))
print(dataset.fillna(dataset['PAYMENTS'].mean(), inplace = True))
print(dataset.isnull().any())

X = dataset.drop('CUST_ID',axis=1).values
y = dataset['CUST_ID'].values



scaler = StandardScaler()
X_Scale = scaler.fit_transform(X)

pca2 = PCA(n_components=2)
principalComponents = pca2.fit_transform(X_Scale)

principalDf = pd.DataFrame(data = principalComponents, columns = ['Principal component 1', 'Principal component 2'])

finalDf = pd.concat([principalDf, dataset[['CUST_ID']]], axis = 1)
print(finalDf.head())

X = finalDf.drop('CUST_ID',axis=1).values
y = finalDf['CUST_ID'].values
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,random_state=0)

logisticRegr = LogisticRegression()
logisticRegr.fit(X_train,y_train)

y_train_hat =logisticRegr.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_hat)*100
print('"Accuracy for our Training dataset with PCA is: %.4f %%' % train_accuracy)

y_test_hat=logisticRegr.predict(X_test)
test_accuracy=accuracy_score(y_test,y_test_hat)*100
test_accuracy
print("Accuracy for our Testing dataset with tuning is : {:.3f}%".format(test_accuracy))

#2nd Main Question

df= pd.read_csv("D:/datasets/pd_speech_features.csv")

print(df.head())

print(df.shape)

print(df['class'].value_counts())

X = df.drop('class',axis=1).values
y = df['class'].values

scaler = StandardScaler()
X_Scale = scaler.fit_transform(X)

print('Scalling has been done')

pca2 = PCA(n_components=3)
principalComponents = pca2.fit_transform(X_Scale)

principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2', 'principal component 3'])

finalDf = pd.concat([principalDf, df[['class']]], axis = 1)
print(finalDf)

X = finalDf.drop('class',axis=1).values
y = finalDf['class'].values

print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,random_state=0)

logisticRegr = LogisticRegression()
logisticRegr.fit(X_train,y_train)

y_train_hat =logisticRegr.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_hat)*100
print('"Accuracy for our Training dataset with PCA is: %.4f %%' % train_accuracy)

y_test_hat=logisticRegr.predict(X_test)
test_accuracy=accuracy_score(y_test,y_test_hat)*100
test_accuracy
print("Accuracy for our Testing dataset with tuning is : {:.3f}%".format(test_accuracy) )

from sklearn.svm import SVC

classifier = SVC()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))

#3rd Main question

d1 = pd.read_csv('D:/datasets/iris.csv')
X = d1.iloc[:, :-1].values
y = d1.iloc[:, -1].values

print(X.shape,y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)
print(X_train.shape,X_test.shape)
