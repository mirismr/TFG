"""Analyze the distribution of classes in ImageNet."""
import codecs
import _pickle as cPickle
from anytree import LevelOrderIter

'''
Hay que pasarle fichero .txt con las URL de todas las imagenes
XXXX_YYYY -> XXXX: WNID, YYYY: imagen
'''
def count_images_per_class(file):
	classes = {}
	images = 0

	with codecs.open(file, "rb", encoding='utf-8', errors='ignore') as f:
		for i, line in enumerate(f):
			label, _ = line.split("\t", 1)
			wnid, _ = label.split("_")
			if wnid in classes:
				classes[wnid] += 1
			else:
				classes[wnid] = 1
			images += 1


	print("Classes: %i" % len(classes))
	print("Images: %i" % images)

	return classes



'''Devuelve el n imagenes de un nodo del arbol (incluido el y sus hijos) REVISAR'''
'''
def get_number_images(nodo):
	total = 0
	if nodo.name in classes:
		total = classes[nodo.name]

	children = nodo.children
	for c in children:
		total = total + get_number_images(c)

	return total
'''
'''elige aquellas clases que tengan al menos param:number_images
	Se incluye las imagenes de sus hijos+propias -> revisar
'''

def print_information(arbol, nodo_buscado):
	classes = cPickle.load(open("../count/count_classes", "rb"))
	raiz = cPickle.load(open(arbol, "rb"))
	buscado = search.find(raiz, lambda node: node.name == nodo_buscado)
	print("PADRE")
	try:
		print(buscado.name, ' -> ', classes[buscado.name])
	except Exception as e:
		print(buscado.name, ' -> ', 0)

	print("HIJOS")
	suma = 0
	for c in buscado.children:
		suma += classes[c.name]
		try:
			print(c.name, ' -> ', classes[c.name])
		except Exception as e:
			print(c.name, ' -> ', 0)

	print("FALTAN: ", 1000-classes[buscado.name])
	print("SUMA TOTAL HIJOS: ", suma)
	print("IMAGENES POR HIJO: ", (1000-classes[buscado.name])/len(buscado.children))

def choose_random_images(dir_src, dir_dest, sysnet, num):
	import os, random
	from shutil import copyfile

	dir_src = dir_src+"/"+sysnet+"/"

	print("Procesando --> "+sysnet)

	if not os.path.exists(dir_dest):
			os.makedirs(dir_dest)

	#eliminar repetidos porque en las descargas me descargo al propio y a los hijos,
	#entonces puede que un hijo se repita luego como padre
	if not os.path.exists(dir_dest+sysnet):
		os.makedirs(dir_dest+sysnet)

		images = os.listdir(dir_src)

		if len(images) >= num:
			random.shuffle(images)

			images_random = images[:num]

			for img in images_random:
				copyfile(dir_src+img,dir_dest+sysnet+"/"+img)
		else:
			print("Not enough images in "+dir_src)
	else:
		print("Sysnet ",sysnet, " repetido. No se ha copiado.")


import os
ruta_choosen = "/home/mirismr/Descargas/choosen/"
ruta_aws = "/home/mirismr/Descargas/aws/"
choosen = os.listdir(ruta_choosen)
for s in choosen:
	choose_random_images(ruta_choosen, ruta_aws, s, 1000)



#classes = count_images_per_class("fall11_urls.txt")
#cPickle.dump(classes, open("count_classes", "wb"))

#raiz = cPickle.load(open("../tree_sysnet_python/tree_n01317541", "rb"))
#print_information("tree_n01317541", "n01317541")

'''
class_counts = [count for _, count in classes.items()]
top10 = sorted(classes.items(), key=lambda n: n[1], reverse=True)[:10]
for class_label, count in top10:
	print("%s:\t%i" % (class_label, count))
'''