from corpus_gen import *

corp = corpus(10000)
vocab = corp.word_vocab

mapped = open('map1.txt', 'r')

mapped_lines = mapped.readlines()
num_items = float(len(mapped_lines))
print(num_items)

with open('reduce1.txt', 'w') as f:
	
	for line in mapped_lines:
		line_split = line.split('!@#')

		if line_split[0] in vocab:
			f.write(line_split[0] + "!@#" + line_split[1])
		else:
			f.write("<unk>!@#" + line_split[1])
