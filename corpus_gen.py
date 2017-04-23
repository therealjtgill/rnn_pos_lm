import nltk
import numpy as np
from numpy.random import rand
import unicodedata
from nltk.corpus import treebank
from nltk.probability import FreqDist

class corpus(object):
	def __init__(self, num_words=10000):
		self.corpus_length = len(treebank.tagged_words())
		self.train_frac = 0.6
		self.validation_frac = 0.2
		self.test_frac = 0.2
		self.word_vocab_size = num_words

		self.pos_vocab = []
		with open('upenn_tagset.dat') as f:
			self.pos_vocab = [line.split(': ')[0] \
				for line in f.readlines() if ':' in line]
		#self.pos_vocab.append(':')
		self.pos_vocab_size = len(self.pos_vocab)

		self.tagged_words = treebank.tagged_words()
		vocab_dist = FreqDist(treebank.words()).most_common(num_words - 1)
		self.word_vocab = [unicodedata.normalize(
			'NFKD', v[0]).encode('ascii', 'ignore') for v in vocab_dist]
		self.word_vocab.append('<unk>')
		print len(self.word_vocab)

	def get_train_batch(self, batch_size=64, seq_length=100):

		offset = 0
		train_frac = self.train_frac
		word_batch, pos_batch = \
			self.get_offset_batch(batch_size, seq_length+1, offset, train_frac)
		word_batch_np, pos_batch_np = \
			self.batch_seq_to_tensor(word_batch, pos_batch)

		word_input = word_batch_np[:,0:seq_length,:].copy()
		pos_input = pos_batch_np[:,0:seq_length,:].copy()

		word_target = word_batch_np[:,1:seq_length + 1,:].copy()
		pos_target = pos_batch_np[:,1:seq_length + 1,:].copy()

		return word_input, pos_input, word_target, pos_target

	def get_validation_batch(self, batch_size=64, seq_length=100):

		train_frac = self.train_frac
		valid_frac = self.validation_frac
		offset = int(train_frac*self.corpus_length)
		word_batch, pos_batch = \
			self.get_offset_batch(batch_size, seq_length+1, offset, valid_frac)
		word_batch_np, pos_batch_np = \
			self.batch_seq_to_tensor(word_batch, pos_batch)

		word_input = word_batch_np[:,0:seq_length,:].copy()
		pos_input = pos_batch_np[:,0:seq_length,:].copy()

		word_target = word_batch_np[:,1:seq_length + 1,:].copy()
		pos_target = pos_batch_np[:,1:seq_length + 1,:].copy()

		return word_input, pos_input, word_target, pos_target

	def get_test_batch(self, batch_size=64, seq_length=100):

		train_frac = self.train_frac
		valid_frac = self.validation_frac
		test_frac = self.test_frac
		offset = int(self.corpus_length*(train_frac + valid_frac))
		word_batch, pos_batch = \
			self.get_offset_batch(batch_size, seq_length, offset, test_frac)
		word_batch_np, pos_batch_np = \
			self.batch_seq_to_tensor(word_batch, pos_batch)

		return word_batch_np, pos_batch_np
		
	def get_offset_batch(self, batch_size, seq_length, offset, frac):

		word_batch = []
		pos_batch = []
		corpus_length = self.corpus_length
		
		for _ in range(batch_size):
			seq_start = int(rand()*(corpus_length*frac - seq_length))
			seq_start += offset
			batch = self.tagged_words[seq_start:seq_start+seq_length]
			word_batch.append([unicodedata.normalize(
				'NFKD', w[0]).encode('ascii', 'ignore') for w in batch])
			pos_batch.append([unicodedata.normalize(
				'NFKD', w[1]).encode('ascii', 'ignore') for w in batch])

		#print(len(word_batch))
		#print(len(word_batch[0]))

		return word_batch, pos_batch

	def batch_seq_to_tensor(self, word_batch, pos_batch):
		batch_size = len(word_batch)
		seq_length = len(word_batch[0])
		WV = self.word_vocab_size
		PV = self.pos_vocab_size

		#print('retrieved batch size, seq length:', batch_size, seq_length)

		word_batch_np = np.zeros((batch_size, seq_length, WV))
		pos_batch_np = np.zeros((batch_size, seq_length, PV))

		#print(word_batch)

		for b in range(batch_size):
			for s in range(seq_length):
				_, word_index = self.word_to_one_hot(word_batch[b][s])
				word_batch_np[b, s, word_index] = 1.
				_, pos_index = self.pos_to_one_hot(pos_batch[b][s])
				pos_batch_np[b, s, pos_index] = 1.

		return word_batch_np, pos_batch_np

	def get_word_vocabulary(self):
		return self.word_vocab

	def get_pos_vocabulary(self):
		return self.pos_vocab

	def pos_to_one_hot(self, pos):
		index = self.pos_vocab_size - 1		
		one_hot = np.zeros(self.pos_vocab_size)
		if pos in self.pos_vocab:
			index = self.pos_vocab.index(pos)
		else:
			print(pos)
		one_hot[index] = 1.
		return one_hot, index

	def word_to_one_hot(self, word):
		index = self.word_vocab_size - 1
		one_hot = np.zeros(self.word_vocab_size)
		if word in self.word_vocab:
			index = self.word_vocab.index(word)
		one_hot[index] = 1.
		return one_hot, index

	def one_hot_to_word(self, one_hot):
		index = np.where(one_hot > 0.)
		return self.word_vocab[index]