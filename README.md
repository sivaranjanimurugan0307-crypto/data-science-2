import pandas as pd
from sklearn.linear_model import LinearRegression
data = {
    'Advertising': [50, 60, 70, 80, 90],
    'Sales': [200, 220, 240, 260, 280]
}
df = pd.DataFrame(data)
X = df[['Advertising']]
y = df['Sales']
model = LinearRegression()
model.fit(X, y)
new_ad = [[100]]
prediction = model.predict(new_ad)
print("Predicted Sales:", prediction[0])