from anytree import Node, RenderTree, AsciiStyle, search
import _pickle as cPickle
from anytree.exporter import JsonExporter

#obtiene las etiquetas asociadas a un synset dado su wnid
def get_words(wnid):
	link = "http://www.image-net.org/api/text/wordnet.synset.getwords?wnid="+wnid
	import urllib.request
	data = urllib.request.urlopen(link)

	words = ''
	for line in data:
		line = line.decode("utf-8")
		words=words+line

	words = words.split('\n')
	words.pop()

	string = ''
	for x in words:
		string=string+x+", "

	string = string[:-2]

	return string

def get_hyponyms(wnid):
	link = "http://www.image-net.org/api/text/wordnet.structure.hyponym?wnid="+wnid
	print(link)
	import urllib.request
	data = urllib.request.urlopen(link)

	hyponyms = ''
	for line in data:
		line = line.decode("utf-8")
		hyponyms=hyponyms+line

	hyponyms = hyponyms.split('\r\n-')
	hyponyms[-1] = hyponyms[-1][:-2]
	hyponyms.remove(wnid)
	
	return hyponyms

def is_leaf(wnid):
	hyponyms = get_hyponyms(wnid)
	
	if hyponyms:
		return False
	else:
		return True

def build_tree(node):
	hijos = get_hyponyms(node.name)
	print(hijos)
	for h in hijos:
		hijo = Node(h, parent=node)
		if not is_leaf(h):
			build_tree(hijo)

def search_in_tree(tree_wnid, wnid):
	file_name = "tree_"+tree_wnid
	root = cPickle.load(open(file_name, "rb"))

	result = search.find_by_attr(root, wnid)
	
	w = Walker()
	walked = [w.walk(root, result)]

	list_nodes = [walked[0][1].name]
	for i in walked[0][2]:
		list_nodes.append(i.name)
	
	return list_nodes


#raiz = cPickle.load(open("tree_n12992868", "rb"))
#exporter = JsonExporter()
#exporter.write(raiz, open("json_n12992868.json", "w"))
#print(RenderTree(raiz))

'''
raiz = Node("n12982915")
build_tree(raiz)
cPickle.dump(raiz, open("arbolprueba", "wb"))
'''