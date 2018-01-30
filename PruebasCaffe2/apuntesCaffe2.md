# Conceptos generales

*ModelHelper*: Crea el modelo de la red.

*XavierFill*: inicializar pesos de tal forma que la varianza sea igual para la salida y la entrada. *With each passing layer, we want the variance to remain the same. This helps us keep the signal from exploding to a high value or vanishing to zero. In other words, we need to initialize the weights in such a way that the variance remains the same for x and y.*
`weight = m.param_init_net.XavierFill([], 'fc_w', shape=[10, 100])`

*ConstantFill*: inicializar bias a constante, en concreto a 0
`bias = m.param_init_net.ConstantFill([], 'fc_b', shape=[10, ])`

Las dimensiones de los p esos y los bias se deben corresponder con los datos de entrada y las etiquetas:

```
data = np.random.rand(16, 100).astype(np.float32)
label = (np.random.rand(16) * 10).astype(np.int32)
```

`m.param_init_net.*`:  devuelve un `BlobReference`: un wrapper sobre un string, es decir, una clase con un atributo string
Es equivalente:
```
fc_1 = m.net.FC(["data", 'fc_w', 'fc_b'], "fc1")
fc_1 = m.net.FC(["data", weight, bias], "fc1")
```

El workspace contiene los parámetros (weights and bias), pero los introduce una vez llamado `RunNetOnce`. Para saber si el workspace tiene algún Blob: `print(workspace.HasBlob('fc_w'))`

`RunNet`: epoca de 10, se entrena a la red con los mismos datos 10 veces:
`workspace.RunNet(m.name, 10)`

*Softmax*: para cada dato nos da la probabilidad de que se corresponda con esa etiqueta.
Cada fila de softmax se corresponde con las probabilidades de las 0,1,2...9 etiquetas con los datos:
fila 0 -> probabilidades para cada clase (0,9) del primer dato : dato[0][0]
fila 1 -> probababilidades para cada clase (0,9) del segundo dato : dato[0][1]
...
fila i -> probababilidades para cada clase (0,9) del segundo dato : dato[0][i]



# CNN MNIST
*ModelHelper* tiene dos redes: una para la inicialización de parámetros `param_init_net` y otra para la computación `net`.

`model.Print` o `model.Summarize` para ver parametros o resúmenes de los paramétros de la red.  

Formato de datos NCHW or channels_first, NHWC or channels_last.
In this example, we’re using NCHW storage order on the mnist_train dataset.
This is called NCHW for Number, Channels, Height and Width.

## Capa RELU
One way ReLUs improve neural networks is by speeding up training. The gradient computation is very simple (either 0 or 1 depending on the sign of xx). Also, the computational step of a ReLU is easy: any negative elements are set to 0.0 -- no exponentials, no multiplication or division operations.
Nonlinear activation functions are important because the function you are trying to learn is usually nonlinear(los conceptos que vamos a aprender no se describen con funciones lineales, por ejemplo las predicciones). If nonlinear activation functions weren’t used, the net would be a large linear classifier.
After each conv layer, it is convention to apply a nonlinear layer (or activation layer) immediately afterward.The purpose of this layer is to introduce nonlinearity to a system that basically has just been computing linear operations during the conv layers (just element wise multiplications and summations).

Usar una RELU después de cada capa conv.
La desventaja es que tras cierto tiempo de entrenamiento puede causar "neuronas muertas" ya que cuando actualiza el gradiente siempre se lo lleva a 0 (la función es f(x)=max(0,x)).

## LabelCrossEntropy
We need a way to measure the difference between predicted probabilities y and ground-truth probabilities y', and during training we try to tune parameters so that this difference is minimized.

This operator is almost always used after getting a softmax and before computing the model’s loss.

Es una forma de medir el error, lo podríamos hacer con error cuadrático o cualquier otra forma, pero éste suele dar mejores resultados.

