import urllib.request
import _pickle as cPickle
import os, collections

ruta = "/home/mirismr/Descargas/tree_n00015388/"


classes = cPickle.load(open("chosen_classes_tree_n00015388", "rb"))

#comprobar que se han descargado todos
'''
os.chdir(ruta)
files = os.listdir(ruta)
files = [x[:-4] for x in files]
print(len(classes))
print(len(files))
print(set(files) == set(classes))
'''

#classes = classes[classes.index("n01674216")+1:]
for wnid in classes:
	print("Procesando -> ",wnid)
	link = "http://image-net.org/download/synset?wnid="+wnid+"&username=mirismr&accesskey=bc84ebb8e94dbbb9fe1af4fcadb13b6368b19c11&release=latest&src=stanford"
	testfile = urllib.request.URLopener()
	testfile.retrieve(link, ruta+wnid+".tar")
	print("Descargado -> ",wnid)
