import easyocr as ocr  #OCR

import streamlit as st  #Web App
from PIL import Image #Image Processing
import numpy as np #Image Processing 

from pysentimiento import create_analyzer

# transformer
from spanlp.palabrota import Palabrota



#title
st.title("Prueba de funcionamiento del modelo de detección")

#subtitle
st.markdown("## Detección y analisís")

st.markdown("## Inserte descripción del modelo")
#st.markdown("")

#image uploader
image = st.file_uploader(label = "Upload your image here",type=['png','jpg','jpeg'])


@st.cache_data 
def load_model(): 
    reader = ocr.Reader(['es'],model_storage_directory='.')
    return reader 

class model_NLP:
    def __init__(self):
        # pysentimiento
        self.pysentimiento_sentimiento = create_analyzer(task="sentiment", lang="es")
        self.pysentimiento_odio = create_analyzer(task="hate_speech", lang="es")
        self.pysentimiento_emociones = create_analyzer(task="emotion", lang="es")
        self.spanlp_palabrota = Palabrota()
        

        self.texto = None
        
    def preprocesamiento(self, list):
        self.texto  = ' '.join(list)

    def palabrotas(self):
        return self.spanlp_palabrota.contains_palabrota(self.texto)

    def analizar_sentimiento(self):
        # Realizar predicción de sentimiento
        resultado = self.pysentimiento_sentimiento.predict(self.texto)
        return resultado
    
    def analizar_odio(self):
        # Realizar predicción de odio
        resultado = self.pysentimiento_odio.predict(self.texto)
        return resultado
    
    def analizar_emociones(self):
        # Realizar predicción de emociones
        resultado = self.pysentimiento_emociones.predict(self.texto)
        return resultado

    def predict(self, list):
        self.preprocesamiento(list)

        return self.palabrotas(), self.analizar_sentimiento(), self.analizar_odio(), self.analizar_emociones()
    
palabras_ = []



reader = load_model() #load model
model = model_NLP()

if image is not None:

    input_image = Image.open(image) #read image
    st.image(input_image) #display image

    with st.spinner("AI is at Work! "):
        result = reader.readtext(np.array(input_image))

        



        result_text = [] #empty list for results


        for text in result:
            result_text.append(text[1])

        st.write(result_text)

        palabrotas, sentido, odio, emocion = model.predict(result_text)

        st.markdown("### Resultados")

        st.markdown(f'#### Contiene garabatos {palabrotas}')

        st.markdown(f'#### Sentido')
        for llave in sentido.probas.keys():
            st.markdown(f'Sentido: {llave}, probabilidad: {sentido.probas[llave]:.3f}')

        st.markdown(f'#### Emocion')
        for llave in emocion.probas.keys():
            st.markdown(f'Emoción: {llave}, probabilidad: {emocion.probas[llave]:.3f}')
        
        st.markdown(f'#### Odio')
        for llave in odio.probas.keys():
            st.markdown(f'Discurso de Odio: {llave}, probabilidad: {odio.probas[llave]:.3f}')

        

        
    #st.success("Here you go!")
    st.balloons()
    st.snow()

else:
    st.write("Upload an Image")

st.caption("Made by NLP-BP")