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

'''Devuelve el nºimagenes de un nodo del arbol (incluido el y sus hijos)'''
def get_number_images(nodo):
	total = 0
	if nodo.name in classes:
		total = classes[nodo.name]

	children = nodo.children
	for c in children:
		total = total + get_number_images(c)

	return total

'''elige aquellas clases que tengan al menos param:number_images
	Se incluye las imagenes de sus hijos+propias
'''
def choose_classes(number_images, root):
	result = []
	for node in LevelOrderIter(root):
		print(node.name, "\t", get_number_images(node))
		if get_number_images(node) >= number_images and not node.name in result:
			result.append(node.name)

	return result

#classes = count_images_per_class("fall11_urls.txt")
#cPickle.dump(classes, open("count_classes", "wb"))

classes = cPickle.load(open("count_classes", "rb"))
raiz = cPickle.load(open("../tree_sysnet_python/tree_n00015388", "rb"))

chosen_classes = choose_classes(5000, raiz)
print(chosen_classes)
print("total: ",len(chosen_classes))

cPickle.dump(chosen_classes, open("chosen_classes_tree_n00015388", "wb"))

'''
class_counts = [count for _, count in classes.items()]
top10 = sorted(classes.items(), key=lambda n: n[1], reverse=True)[:10]
for class_label, count in top10:
    print("%s:\t%i" % (class_label, count))
'''