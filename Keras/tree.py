from anytree import Node, RenderTree

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
	import urllib.request
	data = urllib.request.urlopen(link)

	hyponyms = ''
	for line in data:
		line = line.decode("utf-8")
		hyponyms=hyponyms+line

	hyponyms = hyponyms.split('\r\n-')
	hyponyms[-1] = hyponyms[-1][:-2]
	
	return hyponyms


print(get_hyponyms("nfall11"))

'''
udo = Node("Udo")
marc = Node("Marc", parent=udo)
lian = Node("Lian", parent=marc)
dan = Node("Dan", parent=udo)
jet = Node("Jet", parent=dan)
jan = Node("Jan", parent=dan)
joe = Node("Joe", parent=dan)

for pre, fill, node in RenderTree(udo):
	print("%s%s" % (pre, node.name))
'''