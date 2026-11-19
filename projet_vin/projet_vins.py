import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. IMPORTATION DIRECTE DES DONNÉES (Rouges + Blancs)
url_base = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"
red = pd.read_csv(url_base + "winequality-red.csv", sep=";")
white = pd.read_csv(url_base + "winequality-white.csv", sep=";")

red["is_red"] = 1
white["is_red"] = 0
df = pd.concat([red, white], ignore_index=True)

# 2. FILTRAGE DES EXTRÊMES (Très mauvais <= 4 vs Très bons >= 7)
NB_MOYENS = 500   # nombre de vins moyens à ajouter (à modifier)

extremes = df[(df["quality"] <= 4) | (df["quality"] >= 7)]
moyens = df[(df["quality"] == 5) | (df["quality"] == 6)].sample(n=NB_MOYENS, random_state=42)

df_extremes = pd.concat([extremes, moyens], ignore_index=True)
df_extremes["target"] = pd.cut(df_extremes["quality"], bins=[0, 4, 6, 10], labels=[0, 1, 2]).astype(int)

print(f"Nombre total de vins sélectionnés : {len(df_extremes)}")
print("Répartition (0 = Mauvais, 1 = Moyen, 2 = Bon) :\n", df_extremes["target"].value_counts(), "\n")

proportions = df_extremes["target"].value_counts(normalize=True)
print("Proportions :\n", proportions.round(3), "\n")
print(f"Exactitude d'un modèle qui répond toujours la classe la plus fréquente : {proportions.max():.2%}")

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

baseline = LogisticRegression(class_weight="balanced" , max_iter=1000)
baseline.fit(X_train_scaled, y_train)
pred_base = baseline.predict(X_test_scaled)
print(f"Précision Régression Logistique (Baseline) : {accuracy_score(y_test, pred_base):.2%}")
print(confusion_matrix(y_test, pred_base))
print(classification_report(y_test, pred_base, target_names=["Mauvais (3-4)","moyen (5-6)", "Bon (7-9)"]))

# 5. MODÈLE 2 : RANDOM FOREST
rf = RandomForestClassifier(n_estimators=100, class_weight ="balanced", random_state=42)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
print(f"Précision Random Forest : {accuracy_score(y_test, pred_rf):.2%}\n")
print(confusion_matrix(y_test, pred_rf))
print(classification_report(y_test, pred_rf, target_names=["Mauvais (3-4)","moyen (5-6)", "Bon (7-9)"]))

# 6. IMPORTANCE DES VARIABLES CHIMIQUES
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Top 5 des composants chimiques les plus déterminants :")
print(importances.head(5))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Créer une figure avec 2 graphiques côte à côte
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Graphique 1 : Matrice de confusion
cm = confusion_matrix(y_test, pred_rf)
sns.heatmap(
    cm,
    cbar = True,
    annot=True,
    fmt="d",
    cmap="Purples",
    xticklabels=["Mauvais (3-4)", "moyen (5-6)", "Bon (7-9)"],
    yticklabels=["Mauvais (3-4)", "moyen (5-6)", "Bon (7-9)"],
    ax=axes[0],
)
axes[0].set_title("Matrice de confusion (Random Forest)")
axes[0].set_xlabel("Prédiction du modèle")
axes[0].set_ylabel("Vraie qualité du vin")

# Graphique 2 : Importance des variables chimiques
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
importances.plot(kind="barh", color="teal", ax=axes[1])
axes[1].set_title("Importance des caractéristiques du vin")
axes[1].set_xlabel("Poids dans la décision du modèle")

# Sauvegarder et afficher
plt.tight_layout()
plt.savefig("projet_vin/resultats_modele.png", dpi=300)
plt.show()
