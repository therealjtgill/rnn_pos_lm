brown = open('brown_tagged.txt')

with open('map1.txt', 'w') as f:	
	for t in map(lambda x: str(x[0:-1]), brown.readlines()):
		f.write(str(t) + "\n")

