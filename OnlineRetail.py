# Load the dataset
file_path = 'path_to_your_dataset/OnlineRetail (1) (1).xlsx'
data = pd.read_csv(file_path, encoding='ISO-8859-1')
# Display the first few rows of the dataset
data.head()
# Check for missing values
data.isnull().sum()
# Drop rows with missing Customer ID
data = data.dropna(subset=['CustomerID'])
# Convert data types
data['CustomerID'] = data['CustomerID'].astype(str)
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
# Remove negative quantities
data = data[data['Quantity'] > 0]
# Remove duplicates
data = data.drop_duplicates()
# Display the cleaned dataset
data.info()
import matplotlib.pyplot as plt
import seaborn as sns
# Plot the distribution of transaction quantities
plt.figure(figsize=(10, 6))
sns.histplot(data['Quantity'], bins=50, kde=True)
plt.title('Distribution of Transaction Quantities')
plt.show()
# Top 10 most popular products
top_products = data['Description'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_products.values, y=top_products.index)
plt.title('Top 10 Most Popular Products')
plt.show()
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
# Create the user-item matrix
user_item_matrix = data.pivot(index='CustomerID', columns='StockCode',
values='Quantity').fillna(0)
user_item_sparse_matrix = csr_matrix(user_item_matrix)
# Apply SVD
svd = TruncatedSVD(n_components=50)
matrix_svd = svd.fit_transform(user_item_sparse_matrix)
# Compute cosine similarity
user_similarity = cosine_similarity(matrix_svd)
# Make recommendations for a given user
def recommend_products(user_id, user_item_matrix, user_similarity, top_n=10):
user_index = user_item_matrix.index.get_loc(user_id)
similar_users = list(enumerate(user_similarity[user_index]))
similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)
similar_users = [user for user in similar_users if user[0] != user_index]
recommended_products = {}
for user in similar_users[:top_n]:
similar_user_id = user_item_matrix.index[user[0]]
similar_user_purchases = user_item_matrix.loc[similar_user_id]
for product, quantity in similar_user_purchases.items():
if quantity > 0 and product not in recommended_products:
recommended_products[product] = quantity
recommended_products = sorted(recommended_products.items(), key=lambda x: x[1],
reverse=True)
return recommended_products[:top_n]
# Example: Recommend products for a specific user
user_id = '12346'
recommendations = recommend_products(user_id, user_item_matrix, user_similarity)
recommendations
from sklearn.feature_extraction.text import TfidfVectorizer
# Vectorize product descriptions
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['Description'])
# Compute cosine similarity between products
product_similarity = cosine_similarity(tfidf_matrix)
# Make recommendations based on product similarity
def recommend_similar_products(product_id, data, product_similarity, top_n=10):
product_index = data[data['StockCode'] == product_id].index[0]
similar_products = list(enumerate(product_similarity[product_index]))
similar_products = sorted(similar_products, key=lambda x: x[1], reverse=True)
similar_products = [product for product in similar_products if product[0] != product_index]
recommended_products = [(data.iloc[product[0]]['StockCode'], product[1]) for product in
similar_products[:top_n]]
return recommended_products
# Example: Recommend similar products for a specific product
product_id = '85123A'
similar_products = recommend_similar_products(product_id, data, product_similarity)
similar_products
