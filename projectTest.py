import pandas as pd
from displayfunction import display
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("hw1.csv")

types_dict = {'Arrive or Depart': str, 'Schedule': str, 'Airline': str, 'Flight Number': str, 'Gate': str,
              'Terminal': float, 'Arrive or Depart Prediction': str}
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

y = arriveOrDepart_enc

X = list(zip(gate_enc, terminal_enc, airline_enc))

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.20,
                                                    random_state=22)

# print("X_train", X_train)
# print("X_test", X_test)
# print("y_train", y_train)
# print("y_test", y_test)

rfc = RandomForestClassifier()

rfc.fit(X_train, y_train)

print(rfc.score(X_train, y_train))
print(rfc.score(X_test, y_test))
df["Arrive or Depart Prediction"] = rfc.predict(X)
print(df["Arrive or Depart Prediction"])
# df.sample(15, random_state=22)

rfc_imp = pd.DataFrame(rfc.feature_importances_, columns=['Importance'])
rfc_imp["Importance"] = rfc_imp["Importance"]*100
rfc_imp= rfc_imp.set_index([df.columns.drop(["Arrive or Depart", "Schedule", "Flight Number", "Arrive or Depart Prediction"])])
display(rfc_imp.sort_values(by="Importance", ascending=False))