import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("hw1.csv")

types_dict = {'Arrive or Depart': str, 'Schedule': str, 'Airline': str, 'Flight Number': str, 'Gate': str,
              'Terminal': float}
for col, col_type in types_dict.items():
    df[col] = df[col].astype(col_type)

df.info()

df.describe()

df.corr()

le = preprocessing.LabelEncoder()

arriveOrDepart_enc = le.fit_transform(df["Arrive or Depart"].values)
print(arriveOrDepart_enc)


airline_enc = le.fit_transform(df["Airline"].values)
print(airline_enc)

gate_enc = le.fit_transform(df["Gate"].values)
print(gate_enc)

terminal_enc = le.fit_transform(df["Terminal"].values)
print(terminal_enc)

features = list(zip(gate_enc, terminal_enc, airline_enc))
print(features)

kNN = KNeighborsClassifier(n_neighbors=100)

kNN.fit(features, arriveOrDepart_enc)

predicted = kNN.predict([(60, 3, 1), (78, 3, 1)])
print(predicted)
