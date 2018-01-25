from caffe2.python import workspace, model_helper, utils
import numpy as np
from caffe2.proto import caffe2_pb2
import matplotlib.pyplot as plt

#Datos espiral
def generarDatosEspiral():
	N = 100 # number of points per class
	D = 2 # dimensionality
	K = 3 # number of classes
	data = np.zeros((N*K,D)).astype(np.float32) # data matrix (each row = single example)
	label = np.zeros(N*K).astype(np.int32) # class labels
	for j in range(K):
		ix = range(N*j,N*(j+1))
		r = np.linspace(0.0,1,N) # radius
		t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
		data[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
		label[ix] = j

	return data, label

def plotEspiral(X, y):
	# lets visualize the data:
	plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
	plt.show()

def generarDatosRandom():
	data = np.random.rand(16, 100).astype(np.float32)
	label = (np.random.rand(16) * 10).astype(np.int32)

	return data, label

def plotLossFunction(loss_vector):
	# Grafica
	plt.plot(loss_vector)
	plt.text(0,loss_vector[0], "I0",fontsize=10)
	plt.text(len(loss_vector),loss_vector[len(loss_vector)-1], "I1",fontsize=10)
	plt.plot([0, len(loss_vector)],[loss_vector[0],loss_vector[len(loss_vector)-1]], label="insatisfaccion", color="red")
	plt.legend(loc="upper left")
	plt.ylabel('Loss')
	plt.show()

data, label = generarDatosRandom()

workspace.FeedBlob("data", data)
workspace.FeedBlob("label", label)

# Create model using a model helper
m = model_helper.ModelHelper(name="mi red")

#XavierFill -> inicializar pesos de tal forma que la varianza sea igual para la salida y la entrada
#With each passing layer, we want the variance to remain the same. This helps us keep the signal from exploding to a high value or vanishing to zero. In other words, we need to initialize the weights in such a way that the variance remains the same for x and y.
weight = m.param_init_net.XavierFill([], 'fc_w', shape=[10, 100])
#ConstantFill -> inicializar bias a constante, en concreto a 0
bias = m.param_init_net.ConstantFill([], 'fc_b', shape=[10, ])

#Descomentar para datos con espiral
#weight = m.param_init_net.XavierFill([], 'fc_w', shape=[3, 2])
#bias = m.param_init_net.ConstantFill([], 'fc_b', shape=[3, ])

#m.param_init_net.* -> devuelve un BlobReference: un wrapper sobre un string
# Podemos poner la siguiente linea asi:
# fc_1 = m.net.FC(["data", 'fc_w', 'fc_b'], "fc1")

#full-connected net
fc_1 = m.net.FC(["data", weight, bias], "fc1")

#funcion activacion
pred = m.net.Sigmoid(fc_1, "pred")
#loss y el softmax -> se almacena el BlobReference en estas variables
softmax, loss = m.net.SoftmaxWithLoss([pred, "label"], ["softmax", "loss"])

#workspace contiene los parametros (weights and bias), pero los introduce una vez llamado RunNetOnce
#print(workspace.HasBlob('fc_w'))

m.AddGradientOperators([loss])

#Inicializar weights and bias. Solo una vez
workspace.RunNetOnce(m.param_init_net)
#print(workspace.HasBlob('fc_w'))

#Crear la red
workspace.CreateNet(m.net)

#print(workspace.HasBlob('fc_w'))

loss_vector = []
for i in range(100):
	data, label = generarDatosRandom()

	workspace.FeedBlob("data", data)
	workspace.FeedBlob("label", label)

	#epoca de 10: se entrena a la red con los mismos datos 10 veces
	workspace.RunNet(m.name, 10)   # run for 10 times
	#loss-> nivel de insatisfaccion de las etiquetas predecidas con las reales -> hay que minimizar el loss
	loss_vector.append(workspace.FetchBlob(loss))



#print("Data: ", workspace.FetchBlob("data"))
#print("Data shape: ", workspace.FetchBlob("data").shape)

#plotEspiral(workspace.FetchBlob("data"), workspace.FetchBlob("label"))
#plotLossFunction(loss_vector)
print(m.net.Proto())

'''
#Softmax: para cada dato nos da la probabilidad de que se corresponda con esa etiqueta
#Cada fila de softmax se corresponde con las probabilidades de las 0,1,2...9 etiquetas con los datos
#fila 0 -> probabilidades para cada clase (0,9) del primer dato : dato[0][0]
#fila 1 -> probababilidades para cada clase (0,9) del segundo dato : dato[0][1]
#...
#fila i -> probababilidades para cada clase (0,9) del segundo dato : dato[0][i]

print("Softmax: ", workspace.FetchBlob(softmax))
for x in range(len(loss_vector)):
	print("Loss: "+str(x), workspace.FetchBlob(loss))
print("Loss Final: ", workspace.FetchBlob(loss))
'''


#Este ejemplo no aprende, genera distintos loss porque dentro del bucle for se generan distintos numeros aleatorios
#Para que aprendiese tenemos que meter mas operadores
#Solamente ilustra algunos de los conceptos como modelHelper o workspace y como afecta al proto meter los distintos operadores
#ejecutar sin m.AddGradientOperators([loss]) y con m.AddGradientOperators([loss])
#y ver el resultaado del print m.net.Proto()