import pickle

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
# Sklearn para pré-processamento e avaliação
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
# TensorFlow e Keras para o modelo de Deep Learning
from tensorflow.keras.models import Sequential, load_model


# ==============================================================================
# 3. CLASSE DO MODELO DE DETECÇÃO (MLP)
# ==============================================================================
class DeepfakeDetector:
    """
    Classe que encapsula a criação, treinamento, avaliação e predição
    usando um modelo Perceptron Multicamadas (MLP).
    """

    def __init__(self, input_dim):
        """
        Inicializa o detector.

        Args:
            input_dim (int): Dimensão do vetor de entrada (número de features).
        """
        self.input_dim = input_dim
        self.model = self._create_model()
        self.scaler = StandardScaler()

    def _create_model(self):
        """
        Cria a arquitetura do modelo MLP com Keras.

        Returns:
            tf.keras.models.Sequential: Modelo Keras compilado.
        """
        model = Sequential(name="Deepfake_Voice_Detector_MLP")
        model.add(Input(shape=(self.input_dim,)))

        # Camada 1
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        # Camada 2
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        # Camada 3
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        # Camada 4
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))

        # Camada de Saída
        model.add(Dense(1, activation='sigmoid'))

        # Compila o modelo
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        Treina o modelo com os dados fornecidos.

        Args:
            X_train, y_train: Dados de treinamento.
            X_val, y_val: Dados de validação.
            epochs (int): Número de épocas para o treinamento.
            batch_size (int): Tamanho do lote.
        """
        # Normaliza os dados de treino e aplica a mesma transformação na validação
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Callbacks para otimizar o treinamento
        checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

        print("\nIniciando o treinamento do modelo...")
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint, early_stopping, reduce_lr]
        )
        print("Treinamento concluído.")
        return history

    def evaluate(self, X_test, y_test):
        """
        Avalia o modelo treinado no conjunto de teste.

        Args:
            X_test, y_test: Dados de teste.
        """
        # Normaliza os dados de teste
        X_test_scaled = self.scaler.transform(X_test)

        # Faz predições
        y_pred_proba = self.model.predict(X_test_scaled).ravel()
        y_pred_class = (y_pred_proba > 0.5).astype(int)

        # Calcula e imprime as métricas
        print("\n--- Resultados da Avaliação no Conjunto de Teste ---")
        print(f"Acurácia: {accuracy_score(y_test, y_pred_class):.4f}")
        print(f"Precisão: {precision_score(y_test, y_pred_class):.4f}")
        print(f"Recall: {recall_score(y_test, y_pred_class):.4f}")
        print(f"F1-Score: {f1_score(y_test, y_pred_class):.4f}")
        print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")
        print("\nMatriz de Confusão:")
        print(confusion_matrix(y_test, y_pred_class))
        print("---------------------------------------------------\n")

    def save(self, model_path='deepfake_detector.h5', scaler_path='scaler.pkl'):
        """ Salva o modelo treinado e o normalizador. """
        self.model.save(model_path)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Modelo salvo em {model_path} e normalizador em {scaler_path}")

    @classmethod
    def load(cls, model_path='deepfake_detector.h5', scaler_path='scaler.pkl'):
        """ Carrega um modelo pré-treinado e o normalizador. """
        loaded_model = load_model(model_path)
        with open(scaler_path, 'rb') as f:
            loaded_scaler = pickle.load(f)

        input_dim = loaded_model.input_shape[1]
        detector = cls(input_dim)
        detector.model = loaded_model
        detector.scaler = loaded_scaler
        print(f"Modelo e normalizador carregados de {model_path} e {scaler_path}")
        return detector
