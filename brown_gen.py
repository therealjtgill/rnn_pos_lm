import nltk
import numpy as np
from numpy.random import rand
import unicodedata
from nltk.corpus import brown
from nltk.probability import FreqDist

corpus_length = len(brown.tagged_words())
train_size = 0.6
validation_size = 0.2
test_size = 0.2

def get_train_batch(batch_size=64, seq_length=100):

	batch = []
	tagged_words = brown.tagged_words()

	for _ in range(batch_size):
		seq_start = int(rand()*(train_size*corpus_length-seq_length))
		batch.append(tagged_words[seq_start:seq_start+seq_length])

	return batch

def get_validation_batch(batch_size=64, seq_length=100):

	batch = []
	offset = train_size*corpus_length
	tagged_words = brown.tagged_words()

	for _ in range(batch_size):
		seq_start = int(rand()*(validation_size*corpus_length-seq_length))
		seq_start += offset
		batch.append(tagged_words[seq_start:seq_start+seq_length])

	return batch

def get_test_batch(batch_size=64, seq_length=100):

	batch = []
	offset = corpus_length*(train_size + validation_size)
	tagged_words = brown.tagged_words()

	for _ in range(batch_size):
		seq_start = int(rand()*(validation_size*corpus_length-seq_length))
		seq_start += offset
		batch.append(tagged_words[seq_start:seq_start+seq_length])

	return batch

def get_word_vocabulary(num_words=10000):
	return FreqDist(brown.words()).most_common(num_words)

def get_pos_vocabulary():
	tagset = []
	with open('brown_tagset.txt') as f:
		tagset = [line.split('\t')[0] for line in f.readlines()]
	
	return tagset