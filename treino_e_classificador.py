import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt
import cv2
import os

def carregar_imagens(diretorio, tamanho_imagem=(150, 150)):
    imagens = []
    rotulos = []
    for classe in os.listdir(diretorio):
        caminho_classe = os.path.join(diretorio, classe)
        if os.path.isdir(caminho_classe):
            for imagem_nome in os.listdir(caminho_classe):
                caminho_imagem = os.path.join(caminho_classe, imagem_nome)
                imagem = cv2.imread(caminho_imagem)
                imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)  # Converter para RGB
                imagem = cv2.resize(imagem, tamanho_imagem)
                imagem = imagem / 255.0  # Normalização
                imagens.append(imagem)
                rotulos.append(classe)
    return np.array(imagens), np.array(rotulos)

# Exemplo de uso
diretorio_imagens = 'dataset'  # Substitua pelo caminho do seu dataset
imagens, rotulos = carregar_imagens(diretorio_imagens)

# Converter rótulos de texto para numéricos (ex: 'gato' -> 0, 'cachorro' -> 1)
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
rotulos_numericos = encoder.fit_transform(rotulos)

# Dividir o dataset
from sklearn.model_selection import train_test_split
X_treino, X_teste, y_treino, y_teste = train_test_split(
    imagens, rotulos_numericos, test_size=0.2, random_state=42
)
X_treino, X_val, y_treino, y_val = train_test_split(
    X_treino, y_treino, test_size=0.1, random_state=42
)

modelo = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')  # Saída binária (gato ou cachorro)
])

modelo.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

historico = modelo.fit(X_treino, y_treino, epochs=10, batch_size=32, validation_data=(X_val, y_val))

perda, acuracia = modelo.evaluate(X_teste, y_teste)
print('Acurácia no conjunto de teste:', acuracia)

plt.plot(historico.history['accuracy'], label='acurácia_treino')
plt.plot(historico.history['val_accuracy'], label = 'acurácia_validação')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.savefig('grafico_acuracia.png')

modelo.save('modelo_gatos_cachorros.h5')  # Salve o modelo em um arquivo .h5

nova_imagem = cv2.imread('dataset/test/WhatsApp Image 2024-10-24 at 18.19.00.jpeg')
nova_imagem = cv2.cvtColor(nova_imagem, cv2.COLOR_BGR2RGB)
nova_imagem = cv2.resize(nova_imagem, (150, 150))
nova_imagem = nova_imagem / 255.0
nova_imagem = np.expand_dims(nova_imagem, axis=0)  # Adicionar dimensão do batch

previsao = modelo.predict(nova_imagem)
if previsao > 0.5:
    print('Previsão: Cachorro')
else:
    print('Previsão: Gato')