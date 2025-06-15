import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Passo 1: Carregar os dados do CSV
try:
    csv_path = 'DATASET-balanced.csv'
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"ERRO: Arquivo não encontrado em '{csv_path}'")
    exit()

print("Dataset carregado com sucesso")
print("Primeiras linhas do dataset:")
print(df.head())
print(f"\nDimensões do dataset: {df.shape}")
print(f"Distribuição das classes:\n{df['LABEL'].value_counts()}")


X = df.drop('LABEL', axis=1)
y = df['LABEL']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nDimensões dos dados de treino: {X_train.shape}")
print(f"Dimensões dos dados de teste: {X_test.shape}")


# --- Modelo 1: K-Nearest Neighbors (KNN) ---
print("\n--- Treinando e Avaliando o K-Nearest Neighbors (KNN) ---")

pipeline_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])

pipeline_knn.fit(X_train, y_train)
y_pred_knn = pipeline_knn.predict(X_test)

print("\nResultados do KNN:")
print(f"Acurácia: {accuracy_score(y_test, y_pred_knn):.4f}")
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred_knn))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred_knn))


print("\n\n--- Treinando e Avaliando o Random Forest ---")

pipeline_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])

pipeline_rf.fit(X_train, y_train)
y_pred_rf = pipeline_rf.predict(X_test)

print("\nResultados do Random Forest:")
print(f"Acurácia: {accuracy_score(y_test, y_pred_rf):.4f}")
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred_rf))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred_rf))