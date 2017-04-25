brown = open('brown_tagged.txt')

with open('map1.txt', 'w') as f:	
	for t in brown.readlines():
		f.write(str(t) + "\n")

