from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.filechooser import FileChooserListView
from kivy.clock import Clock
import cv2
import numpy as np
from tensorflow import keras

# Carregar o modelo treinado
modelo = keras.models.load_model('modelo_gatos_cachorros.h5')

class ClassificadorApp(App):
    def build(self):
        self.imagem_path = None
        layout = BoxLayout(orientation='vertical')

        # Seletor de arquivos
        self.filechooser = FileChooserListView()
        layout.add_widget(self.filechooser)

        # Botão para carregar a imagem
        botao_carregar = Button(text="Carregar Imagem")
        botao_carregar.bind(on_press=self.carregar_imagem)
        layout.add_widget(botao_carregar)

        # Imagem
        self.imagem = Image()
        layout.add_widget(self.imagem)

        # Botão para classificar
        botao_classificar = Button(text="Classificar")
        botao_classificar.bind(on_press=self.classificar_imagem)
        layout.add_widget(botao_classificar)

        # Resultado
        self.resultado = Label(text="")
        layout.add_widget(self.resultado)

        return layout

    def carregar_imagem(self, instance):
        try:
            self.imagem_path = self.filechooser.selection[0]
            self.imagem.source = self.imagem_path
            self.imagem.reload()
        except Exception as e:
            self.resultado.text = f"Erro: {e}"

    def classificar_imagem(self, instance):
        if self.imagem_path:
            try:
                # Pré-processar a imagem com OpenCV
                imagem = cv2.imread(self.imagem_path)
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
                self.resultado.text = f"Resultado: {resultado}"
            except Exception as e:
                self.resultado.text = f"Erro: {e}"
        else:
            self.resultado.text = "Carregue uma imagem primeiro!"

if __name__ == '__main__':
    ClassificadorApp().run()
