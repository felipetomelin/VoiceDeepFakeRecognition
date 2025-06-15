# BLOCO DE EXECUÇÃO PRINCIPAL
import os
import kagglehub
from sklearn.model_selection import train_test_split

from AudioProcessor import AudioProcessor
from DeepFakeVoiceDetector import DeepfakeDetector

# Download latest version
path = kagglehub.dataset_download("birdy654/deep-voice-deepfake-voice-recognition")

print("Path to dataset files:", path)

if __name__ == '__main__':
    DATASET_PATH = path

    # Verifica se o diretório do dataset existe
    if not os.path.exists(DATASET_PATH):
        print(f"ERRO: O diretório do dataset '{DATASET_PATH}' não foi encontrado.")
        print("Por favor, baixe o dataset de https://www.kaggle.com/datasets/birdy654/deep-voice-deepfake-voice-recognition")
    else:
        # Processamento dos Dados
        processor = AudioProcessor()
        X, y = processor.load_dataset(DATASET_PATH)

        # Divisão dos Dados em Treino, Validação e Teste
        # 70% treino, 15% validação, 15% teste
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

        print(f"\nDimensões dos conjuntos de dados:")
        print(f"Treino: {X_train.shape}, Validação: {X_val.shape}, Teste: {X_test.shape}")

        # Criação e Treinamento do Modelo
        detector = DeepfakeDetector(input_dim=processor.feature_dimension)
        detector.train(X_train, y_train, X_val, y_val, epochs=100)

        # Avaliação do Modelo
        detector.evaluate(X_test, y_test)

        # Salvando o modelo treinado
        detector.save()

        # Trocar por algum arquivo que não seja do dataset
        sample_real_audio = os.path.join(DATASET_PATH, 'real', os.listdir(os.path.join(DATASET_PATH, 'real'))[0])
        sample_fake_audio = os.path.join(DATASET_PATH, 'fake', os.listdir(os.path.join(DATASET_PATH, 'fake'))[0])

        # Carrega o modelo salvo
        loaded_detector = DeepfakeDetector.load()

        # Extrai features do áudio de teste
        real_features = processor.extract_features(sample_real_audio).reshape(1, -1)
        fake_features = processor.extract_features(sample_fake_audio).reshape(1, -1)

        # Normaliza as features com o scaler carregado
        real_features_scaled = loaded_detector.scaler.transform(real_features)
        fake_features_scaled = loaded_detector.scaler.transform(fake_features)

        # Faz a predição
        pred_real = loaded_detector.model.predict(real_features_scaled)[0][0]
        pred_fake = loaded_detector.model.predict(fake_features_scaled)[0][0]

        print(
            f"Predição para áudio REAL ('{os.path.basename(sample_real_audio)}'): {pred_real:.4f} -> {'FAKE' if pred_real > 0.5 else 'REAL'}")
        print(
            f"Predição para áudio FAKE ('{os.path.basename(sample_fake_audio)}'): {pred_fake:.4f} -> {'FAKE' if pred_fake > 0.5 else 'REAL'}")
