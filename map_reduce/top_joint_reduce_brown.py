#from corpus_gen import *
import operator

mapped = open('reduce2.txt', 'r')
sorted_ = open('sorted_joint.txt', 'w')

mapped_lines = mapped.readlines()

#corp = corpus(10000)
#word_vocab = corp.word_vocab
#pos_vocab = corp.pos_vocab
all_pairs = {}
k = 25

with open('reduce3.txt', 'w') as f:
	current_key = None
	count = 1.

	for line in mapped_lines:

		line_split = line.split('!@#')
		word = line_split[0]
		pos, prob = line_split[1].split('\t')

		prob = float(prob)
		all_pairs[(word, pos)] = prob

sorted_pairs = sorted(all_pairs.items(), key=operator.itemgetter(1))[::-1]

top_k_prob = 0.
for i, pair in enumerate(sorted_pairs):
	sorted_.write(str(pair[0][0]) + "," + str(pair[0][1]) + "," + str(pair[1]) + "," + '\n')
	if i+1 < k:
		top_k_prob += pair[1]
		print(pair[0], pair[1])

print('The top', k, 'pairs represent', top_k_prob*100, 'percent of all data')

sorted_.close()