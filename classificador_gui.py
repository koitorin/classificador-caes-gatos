import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
from tensorflow import keras

# Carregar o modelo treinado
modelo = keras.models.load_model('modelo_gatos_cachorros.h5')

def carregar_imagem():
    global imagem_path  # Variável global para armazenar o caminho da imagem
    imagem_path = filedialog.askopenfilename()
    if imagem_path:
        # Abrir a imagem com Pillow
        imagem = Image.open(imagem_path)
        imagem = imagem.resize((300, 300))  # Redimensionar para exibir na interface
        imagem_tk = ImageTk.PhotoImage(imagem)
        label_imagem.config(image=imagem_tk)
        label_imagem.image = imagem_tk  # Manter uma referência para evitar que a imagem seja descartada pelo garbage collector

def classificar_imagem():
    if imagem_path:
        try:
            # Pré-processar a imagem com OpenCV
            imagem = cv2.imread(imagem_path)
            imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
            imagem = cv2.resize(imagem, (150, 150))
            imagem = imagem / 255.0
            imagem = np.expand_dims(imagem, axis=0)

            # Classificar a imagem
            previsao = modelo.predict(imagem)
            if previsao > 0.5:
                resultado = 'Cachorro'
            else:
                resultado = 'Gato'

            # Exibir o resultado
            label_resultado.config(text=f"Resultado: {resultado}")
        except Exception as e:
            label_resultado.config(text=f"Erro: {e}")
    else:
        label_resultado.config(text="Carregue uma imagem primeiro!")

# Criar a janela principal
janela = tk.Tk()
janela.title("Classificador de Gatos e Cachorros")

# Botão para carregar a imagem
botao_carregar = tk.Button(janela, text="Carregar Imagem", command=carregar_imagem)
botao_carregar.pack(pady=10)

# Label para exibir a imagem
label_imagem = tk.Label(janela)
label_imagem.pack()

# Botão para classificar a imagem
botao_classificar = tk.Button(janela, text="Classificar", command=classificar_imagem)
botao_classificar.pack(pady=10)

# Label para exibir o resultado
label_resultado = tk.Label(janela, text="")
label_resultado.pack()

# Iniciar o loop da interface gráfica
janela.mainloop()