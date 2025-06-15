# CLASSE DE PRÉ-PROCESSAMENTO DE DADOS
import os
import librosa
import numpy as np
import tqdm

class AudioProcessor:
    """
    Classe responsável por carregar o dataset, extrair características (features)
    dos arquivos de áudio e prepará-los para o modelo.
    """

    def __init__(self, n_mfcc=40, max_pad_len=44):
        """
        Inicializa o processador de áudio.

        Args:
            n_mfcc (int): Número de coeficientes MFCC a serem extraídos.
            max_pad_len (int): Comprimento máximo para o padding das features.
                               O vetor de features terá dimensão n_mfcc * max_pad_len.
        """
        self.n_mfcc = n_mfcc
        self.max_pad_len = max_pad_len
        self.feature_dimension = n_mfcc * max_pad_len  # 40 * 44 = 1760

    def extract_features(self, file_path):
        """
        Extrai MFCCs de um único arquivo de áudio.

        Args:
            file_path (str): Caminho para o arquivo de áudio.

        Returns:
            np.ndarray: Vetor de características (features) normalizado e com padding.
        """
        try:
            audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=self.n_mfcc)

            # Aplica padding ou truncamento para garantir um tamanho fixo
            if mfccs.shape[1] > self.max_pad_len:
                mfccs = mfccs[:, :self.max_pad_len]
            else:
                pad_width = self.max_pad_len - mfccs.shape[1]
                mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

            return mfccs.flatten()
        except Exception as e:
            print(f"Erro ao processar o arquivo {file_path}: {e}")
            return None

    def load_dataset(self, dataset_path):
        """
        Carrega o dataset completo a partir do caminho fornecido.
        Espera uma estrutura com subdiretórios 'real' e 'fake'.

        Args:
            dataset_path (str): Caminho para o diretório raiz do dataset.

        Returns:
            tuple: Uma tupla contendo o array de features (X) e o array de labels (y).
        """
        features = []
        labels = []

        # Processa áudios reais (label 0)
        real_path = os.path.join(dataset_path, 'real')
        for filename in tqdm(os.listdir(real_path), desc="Processando áudios REAIS"):
            file_path = os.path.join(real_path, filename)
            data = self.extract_features(file_path)
            if data is not None:
                features.append(data)
                labels.append(0)

        # Processa áudios falsos (label 1)
        fake_path = os.path.join(dataset_path, 'fake')
        for filename in tqdm(os.listdir(fake_path), desc="Processando áudios FAKE"):
            file_path = os.path.join(fake_path, filename)
            data = self.extract_features(file_path)
            if data is not None:
                features.append(data)
                labels.append(1)

        return np.array(features), np.array(labels)