"""Analyze the distribution of classes in ImageNet."""
import codecs
import _pickle as cPickle

classes = {}
images = 0

with codecs.open("fall11_urls.txt", "rb",encoding='utf-8', errors='ignore') as f:
	for i, line in enumerate(f):
		label, _ = line.split("\t", 1)
		wnid, _ = label.split("_")
		if wnid in classes:
			classes[wnid] += 1
		else:
			classes[wnid] = 1
		images += 1

# Output
print("Classes: %i" % len(classes))
print("Images: %i" % images)

print(classes)
cPickle.dump(classes, open("count_classes", "wb"))

'''
class_counts = [count for _, count in classes.items()]
top10 = sorted(classes.items(), key=lambda n: n[1], reverse=True)[:10]
for class_label, count in top10:
    print("%s:\t%i" % (class_label, count))
'''