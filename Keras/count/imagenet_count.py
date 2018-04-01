"""Analyze the distribution of classes in ImageNet."""
import codecs
import _pickle as cPickle

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
	total = classes[nodo.name]

	children = nodo.children
	for c in children:
		if not c.is_leaf:
			total = total + get_number_images(c)

	return total


#classes = count_images_per_class("fall11_urls.txt")
#cPickle.dump(classes, open("count_classes", "wb"))
#probar sin funciona
classes = cPickle.load(open("count_classes", "rb"))
raiz = cPickle.load(open("../tree_sysnet_python/tree_n12992868", "rb"))
print(get_number_images(raiz))

'''
class_counts = [count for _, count in classes.items()]
top10 = sorted(classes.items(), key=lambda n: n[1], reverse=True)[:10]
for class_label, count in top10:
    print("%s:\t%i" % (class_label, count))
'''