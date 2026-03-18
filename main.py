import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Đọc dữ liệu và chuẩn hóa cột
users = pd.read_csv('Users.csv', sep=';', encoding='latin-1', on_bad_lines='skip', dtype={'User-ID': str}, low_memory=False)
books = pd.read_csv('Books.csv', sep=';', encoding='latin-1', on_bad_lines='skip', low_memory=False)
ratings = pd.read_csv('Ratings.csv', sep=';', encoding='latin-1', on_bad_lines='skip', dtype={'User-ID': str, 'ISBN': str}, low_memory=False)

books.columns = ['ISBN', 'Title', 'Author', 'Year', 'Publisher']
users.columns = ['User-ID', 'Age']
ratings.columns = ['User-ID', 'ISBN', 'Rating']

books = books[['ISBN', 'Author', 'Year', 'Publisher']]
users = users[['User-ID', 'Age']]
ratings = ratings[['User-ID', 'ISBN', 'Rating']]

# Chuyển kiểu dữ liệu
books['Year'] = pd.to_numeric(books['Year'], errors='coerce')
users['Age'] = pd.to_numeric(users['Age'], errors='coerce')

# Xóa dữ liệu lỗi
books.dropna(inplace=True)
users.dropna(inplace=True)

# Tính Avg-Rating
avg_rating = ratings.groupby('ISBN')['Rating'].mean().reset_index()
avg_rating.columns = ['ISBN', 'Avg-Rating']

# Merge dữ liệu
df = ratings.merge(users, on='User-ID')
df = df.merge(books, on='ISBN')
df = df.merge(avg_rating, on='ISBN')

# Lọc dữ liệu
df = df[df['Rating'] > 0]
df = df[(df['Age'] > 5) & (df['Age'] < 90)]

# Sample trước để tránh dữ liệu quá lớn
df = df.sample(5000, random_state=42)

# OneHotEncode Author + Publisher
df = pd.get_dummies(df, columns=['Author', 'Publisher'])

# Feature + Target
X = df.drop(columns=['User-ID', 'ISBN', 'Rating'])
y = df['Rating']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_pred = np.clip(y_pred, 0, 10)

# Đánh giá
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("MAE:", mae)
print("RMSE:", rmse)

# Test 5 mẫu ngẫu nhiên
samples = X_test.sample(5)
predictions = model.predict(samples)
actuals = y_test.loc[samples.index]

for i in range(5):
    print("SAMPLE", i+1)
    print("Predicted:", predictions[i])
    print("Actual:", actuals.values[i])
    print("------")

# Biểu đồ
plt.scatter(y_test, y_pred, s=10)
plt.plot([0, 10], [0, 10], color='red')
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("Actual vs Predicted Rating")
plt.show()