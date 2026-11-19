import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. IMPORTATION DIRECTE DES DONNÉES (Rouges + Blancs)
url_base = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"
red = pd.read_csv(url_base + "winequality-red.csv", sep=";")
white = pd.read_csv(url_base + "winequality-white.csv", sep=";")

red["is_red"] = 1
white["is_red"] = 0
df = pd.concat([red, white], ignore_index=True)

# 2. FILTRAGE DES EXTRÊMES (Très mauvais <= 4 vs Très bons >= 7)
df_extremes = df[(df["quality"] <= 4) | (df["quality"] >= 7)].copy()
df_extremes["target"] = (df_extremes["quality"] >= 7).astype(int)

print(f"Nombre total de vins sélectionnés : {len(df_extremes)}")
print("Répartition (0 = Mauvais, 1 = Bon) :\n", df_extremes["target"].value_counts(), "\n")

# 3. DÉCOUPAGE ENTRAÎNEMENT / TEST
X = df_extremes.drop(columns=["quality", "target"])
y = df_extremes["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. MODÈLE 1 : RÉGRESSION LOGISTIQUE (Baseline)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

baseline = LogisticRegression()
baseline.fit(X_train_scaled, y_train)
pred_base = baseline.predict(X_test_scaled)
print(f"Précision Régression Logistique (Baseline) : {accuracy_score(y_test, pred_base):.2%}")

# 5. MODÈLE 2 : RANDOM FOREST
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
print(f"Précision Random Forest : {accuracy_score(y_test, pred_rf):.2%}\n")

# 6. IMPORTANCE DES VARIABLES CHIMIQUES
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Top 5 des composants chimiques les plus déterminants :")
print(importances.head(5))