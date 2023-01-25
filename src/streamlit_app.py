import streamlit as st
import numpy as np

st.set_page_config(
    page_title="Perceptrón simple",
    initial_sidebar_state="collapsed",
    layout="wide",
)

image = 'src/img/neurona.jpg'
st.image(image, caption='Neurona')


st.title("Simulador de una neurona")

neuronas = st.slider('Elige el número de entradas/pesos que tendrá la neurona', 1, 10)

x = []
st.subheader("Entradas")
cols = st.columns(neuronas)
for i in range(neuronas):
    with cols[i]:
        x.append(st.number_input(f'$x${i}', key=f'x{i}'))

weights = []
st.subheader("Pesos")
cols = st.columns(neuronas)
for i in range(neuronas):
    with cols[i]:
        weights.append(st.number_input(f'$w${i}', key=f'w{i}'))

col1, col2 = st.columns(2)
with col1:
    bias = st.number_input("introduce el valor del sesgo")


with col2:
    func = st.selectbox("elige la funcion de activacion", ["Sigmoide", "ReLu", "Tanh"])

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))
 
def tanh(x):
    return np.tanh(x)

def relu(x):
  return np.maximum(0, x)

class Neuron:
 
    def __init__(self, weights, bias, func):
        self.func = func
        self.weights = weights
        self.bias = bias

    def run(self, input_data):
      x = np.array(input_data)
      y = np.dot(x, self.weights) + (self.bias)
      if func == 'Sigmoide':
          self.func = sigmoid
      elif func == 'Tanh':
          self.func = tanh
      elif func == 'ReLu':
          self.func = relu
      return self.func(y)



if st.button("Calcular la salida"):

    n1 = Neuron(weights, bias, func)
    out=n1.run(input_data=x)

    st.text(f"This instance is a {out}")

