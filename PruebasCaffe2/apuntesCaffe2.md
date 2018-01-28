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