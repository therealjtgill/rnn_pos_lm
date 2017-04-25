from corpus_gen import *

mapped = open('reduce2.txt', 'r')

mapped_lines = mapped.readlines()

corp = corpus(10000)
word_vocab = corp.word_vocab
pos_vocab = corp.pos_vocab

with open('reduce3.txt', 'w') as f:
	current_key = None
	count = 1.

	for line in mapped_lines:
		#if line.count(',') == 1:
		#	line_split = line.split(',')
		#else:
		#	line_split = line.split(',', 2)
		#line_split = line.split('\t')
		line_split = line.split('!@#')
		word = line_split[0]
		pos, prob = line_split[1].split('\t')

		_, word_index = corp.word_to_one_hot(word)
		_, pos_index = corp.pos_to_one_hot(pos)

		f.write(str(word_index) + "," + str(pos_index) + "\t" + prob)
