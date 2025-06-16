# CLASSE DE PRÉ-PROCESSAMENTO DE DADOS
import os
import librosa
import numpy as np
from tqdm import tqdm

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
        Espera uma estrutura com subdiretórios 'REAL' e 'FAKE'.

        Args:
            dataset_path (str): Caminho para o diretório raiz do dataset.

        Returns:
            tuple: Uma tupla contendo o array de features (X) e o array de labels (y).
        """
        features = []
        labels = []

        # Processa áudios reais (label 0)
        real_path = os.path.join(dataset_path, 'REAL')
        for filename in tqdm(os.listdir(real_path), desc="Processando áudios REAIS"):
            file_path = os.path.join(real_path, filename)
            data = self.extract_features(file_path)
            if data is not None:
                features.append(data)
                labels.append(0)

        # Processa áudios falsos (label 1)
        fake_path = os.path.join(dataset_path, 'FAKE')
        for filename in tqdm(os.listdir(fake_path), desc="Processando áudios FAKE"):
            file_path = os.path.join(fake_path, filename)
            data = self.extract_features(file_path)
            if data is not None:
                features.append(data)
                labels.append(1)

        return np.array(features), np.array(labels)


def extract_csv_compatible_features(file_path):
    try:
        # Carrega o áudio
        y, sr = librosa.load(file_path, sr=None, mono=True)

        # Extrai as características, calculando a média para ter um valor por arquivo
        chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
        rms = np.mean(librosa.feature.rms(y=y))
        spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))

        # Extrai os MFCCs (20, como no CSV)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfccs_mean = np.mean(mfccs, axis=1)  # Média de cada coeficiente

        # Concatena todas as características em um único vetor
        features = np.array([
            chroma_stft,
            rms,
            spec_cent,
            spec_bw,
            rolloff,
            zcr
        ])
        features = np.concatenate((features, mfccs_mean))

        return features

    except Exception as e:
        print(f"Erro ao processar o arquivo {file_path}: {e}")
        return None