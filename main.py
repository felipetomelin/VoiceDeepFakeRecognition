import os

import pandas as pd
from sklearn.model_selection import train_test_split

from AudioProcessor import extract_csv_compatible_features
from DeepFakeVoiceDetector import DeepfakeDetector

if __name__ == '__main__':
    CSV_PATH = 'DATASET-balanced.csv'

    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"ERRO: Arquivo CSV não encontrado em '{CSV_PATH}'")
        exit()

    print("Dataset CSV carregado com sucesso!")

    # Prepara os dados para o treinamento
    X = df.drop('LABEL', axis=1)
    y_text = df['LABEL']
    y = y_text.map({'REAL': 0, 'FAKE': 1})

    input_dim = X.shape[1]
    print(f"Número de características do CSV: {input_dim}")

    # Divisão dos dados em Treino, Validação e Teste
    # 70% treino, 15% validação, 15% teste
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print(f"\nDimensões dos conjuntos de dados:")
    print(f"Treino: {X_train.shape}, Validação: {X_val.shape}, Teste: {X_test.shape}")

    detector = DeepfakeDetector(input_dim=input_dim)
    detector.train(X_train.values, y_train.values, X_val.values, y_val.values,
                   epochs=100)

    print("\n--- Avaliando o modelo treinado com os dados de teste do CSV ---")
    detector.evaluate(X_test.values, y_test.values)

    detector.save(model_path='detector_from_csv.h5', scaler_path='scaler_from_csv.pkl')

    # --- PARTE 2: TESTE COM UM ARQUIVO DE ÁUDIO REAL ---

    print("\n\n--- Testando o modelo com um novo arquivo de áudio ---")

    # Carrega o detector recém-salvo
    loaded_detector = DeepfakeDetector.load(model_path='detector_from_csv.h5', scaler_path='scaler_from_csv.pkl')

    PATH_AUDIO_PARA_TESTE = r"trump-original.wav"

    if not os.path.exists(PATH_AUDIO_PARA_TESTE):
        print(f"AVISO: Arquivo de áudio de teste não encontrado em '{PATH_AUDIO_PARA_TESTE}'. Pulando predição.")
    else:
        new_audio_features = extract_csv_compatible_features(PATH_AUDIO_PARA_TESTE)

        if new_audio_features is not None:
            new_audio_features = new_audio_features.reshape(1, -1)
            new_audio_features_scaled = loaded_detector.scaler.transform(new_audio_features)

            prediction_proba = loaded_detector.model.predict(new_audio_features_scaled)[0][0]
            prediction_class = 'FAKE' if prediction_proba > 0.5 else 'REAL'

            print(f"Predição para o áudio '{os.path.basename(PATH_AUDIO_PARA_TESTE)}':")
            print(f"Probabilidade: {prediction_proba:.4f} -> Classe: {prediction_class}")