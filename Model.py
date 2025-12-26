
import pandas as pd
# import os
# print(os.getcwd())
df = pd.read_csv("nba_2008-2025.csv")

# Convert date column
df["date"] = pd.to_datetime(df["date"])

print(df.columns)
print(df.head())



df["home_win"] = (df["score_home"] > df["score_away"]).astype(int)




train_df = df[
    (df["date"] >= "2018-01-01") &
    (df["date"] <  "2023-01-01")
]

test_df = df[df["date"] >= "2023-01-01"]

print("Train games:", len(train_df))
print("Test games:", len(test_df))



features = ["moneyline_home", "moneyline_away"]

# Drop rows with missing odds
train_df = train_df.dropna(subset=features)
test_df  = test_df.dropna(subset=features)

X_train = train_df[features]
y_train = train_df["home_win"]

X_test = test_df[features]
y_test = test_df["home_win"]


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)



print("Train accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))
