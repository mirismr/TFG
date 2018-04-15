import urllib.request
import _pickle as cPickle
import os, collections
from anytree import search

ruta = "/home/mirismr/Descargas/choosen_n02085374/"

count_classes = cPickle.load(open("count_classes", "rb"))
raiz = cPickle.load(open("../tree_sysnet_python/tree_n01317541", "rb"))
buscado = search.find(raiz, lambda node: node.name == "n02085374")

classes = buscado.children
for wnid in classes:
	if count_classes[wnid.name] >= 1000:
		print("Procesando -> ",wnid.name)
		link = "http://image-net.org/download/synset?wnid="+wnid.name+"&username=mirismr&accesskey=bc84ebb8e94dbbb9fe1af4fcadb13b6368b19c11&release=latest&src=stanford"
		testfile = urllib.request.URLopener()
		testfile.retrieve(link, ruta+wnid.name+".tar")
		print("Descargado -> ",wnid.name)
