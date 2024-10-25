import cv2
import numpy as np
from tensorflow import keras

# Carregar o modelo treinado
modelo = keras.models.load_model('modelo_gatos_cachorros.h5')

# Caminho para a nova imagem
caminho_imagem = 'dataset/test/WhatsApp Image 2024-10-24 at 18.09.25.jpeg'

# Carregar e pré-processar a imagem
nova_imagem = cv2.imread(caminho_imagem)
nova_imagem = cv2.cvtColor(nova_imagem, cv2.COLOR_BGR2RGB)
nova_imagem = cv2.resize(nova_imagem, (150, 150))
nova_imagem = nova_imagem / 255.0
nova_imagem = np.expand_dims(nova_imagem, axis=0)

# Classificar a imagem
previsao = modelo.predict(nova_imagem)

# Exibir a previsão
if previsao > 0.5:
    print('Previsão: Cachorro')
else:
    print('Previsão: Gato')