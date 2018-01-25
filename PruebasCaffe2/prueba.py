from caffe2.python import workspace, model_helper
import numpy as np
import matplotlib.pyplot as plt

# Create the input data
data = np.random.rand(16, 100).astype(np.float32)
# Create labels for the data as integers [0, 9].
label = (np.random.rand(16) * 10).astype(np.int32)

workspace.FeedBlob("data", data)
workspace.FeedBlob("label", label)

# Create model using a model helper
m = model_helper.ModelHelper(name="mi red")

#XavierFill -> inicializar pesos de tal forma que la varianza sea igual para la salida y la entrada
#With each passing layer, we want the variance to remain the same. This helps us keep the signal from exploding to a high value or vanishing to zero. In other words, we need to initialize the weights in such a way that the variance remains the same for x and y.
weight = m.param_init_net.XavierFill([], 'fc_w', shape=[10, 100])
#ConstantFill -> inicializar bias a constante, en concreto a 0
bias = m.param_init_net.ConstantFill([], 'fc_b', shape=[10, ])

#m.param_init_net.* -> devuelve un BlobReference: un wrapper sobre un string
# Podemos poner la siguiente linea asi:
# fc_1 = m.net.FC(["data", 'fc_w', 'fc_b'], "fc1")

#full-connected net
fc_1 = m.net.FC(["data", weight, bias], "fc1")

#funcion activacion
pred = m.net.Sigmoid(fc_1, "pred")
#loss y el softmax -> se almacena el BlobReference en esas variables
softmax, loss = m.net.SoftmaxWithLoss([pred, "label"], ["softmax", "loss"])

#workspace contiene los parametros (weights and bias), pero los introduce una vez llamado RunNetOnce
#print(workspace.HasBlob('fc_w'))

#Inicializar weights and bias. Solo una vez
workspace.RunNetOnce(m.param_init_net)
#print(workspace.HasBlob('fc_w'))

#Crear la red
workspace.CreateNet(m.net)

#print(workspace.HasBlob('fc_w'))

loss_vector = []
for i in range(100):
	data = np.random.rand(16, 100).astype(np.float32)
	label = (np.random.rand(16) * 10).astype(np.int32)

	workspace.FeedBlob("data", data)
	workspace.FeedBlob("label", label)

	#epoca de 10: se entrena a la red con los mismos datos 10 veces
	workspace.RunNet(m.name, 10)   # run for 10 times
	#loss-> nivel de insatisfaccion de las etiquetas predecidas con las reales -> hay que minimizar el loss
	loss_vector.append(workspace.FetchBlob(loss))



print("Data: ", workspace.FetchBlob("data"))
print("Softmax: ", workspace.FetchBlob(softmax))
for x in range(len(loss_vector)):
	print("Loss: "+str(x), workspace.FetchBlob(loss))

# TFG
plt.plot(loss_vector)
plt.text(0,loss_vector[0], "I0",fontsize=10)
plt.text(len(loss_vector),loss_vector[len(loss_vector)-1], "I1",fontsize=10)
plt.plot([0, len(loss_vector)],[loss_vector[0],loss_vector[len(loss_vector)-1]], label="insatisfaccion", color="red")
plt.legend(loc="upper left")
plt.ylabel('Loss')
plt.show()
# FIN TFG

print("Loss Final: ", workspace.FetchBlob(loss))

