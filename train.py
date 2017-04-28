from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys
from lm import *
from pos_lm import *
from corpus_gen import *
from utils import *

def decrease_lr(loss, threshold, factor, lr):
	if len(loss) <= 1:
		rate = lr

	else:
		dp = (loss[-2] - loss[-1])/loss[-2]
		if dp < threshold:
			rate = lr * factor
		else:
			rate = lr
	return rate

def main(argv):
	save_dir = get_date()
	tb = corpus(10000)
	word_vocab_size = tb.word_vocab_size
	pos_vocab_size = tb.pos_vocab_size
	batch_size = 32
	seq_length = 50
	sample_step = 400
	halving_threshold = 0.003
	max_halvings = 5
	num_halvings = 0
	max_steps = 10000
	num_steps = 0
	learning_rate = 0.01
	
	sess = tf.Session()
	if argv[0] == 'lm':
		lm = WordLM(sess, 'lm', word_vocab_size)
		save_text('word-level language model', save_dir, 'info')

	elif argv[0] == 'pos':
		lm = WordLM(sess, 'lm', word_vocab_size)
		save_text('word-level+pos language model', save_dir, 'info')
	else:
		print('Invalid specification for model. Use \'lm\' or \'pos\'')
		sys.exit(1)
	sess.run(tf.global_variables_initializer())

	test_seq_loss = []
	perplexity = []
	vperp = []
	while num_steps < max_steps and num_halvings < max_halvings:
		train_pieces = tb.get_train_batch(batch_size, seq_length)
		word_in, pos_in, word_target, pos_target = train_pieces
		train_loss = lm.train(word_in, word_target, learning_rate)

		if num_steps % 20 == 0:
			print(num_steps)
			print('training loss:', train_loss)

		if (num_steps % 200 == 0) and num_steps > 0:
			validation_pieces = tb.get_validation_batch(32, 50)
			word_in, pos_in, word_target, pos_target = validation_pieces
			test_seq_loss.append(lm.validate(word_in, word_target))
			perplexity.append(np.exp(test_seq_loss[-1]))

			train_pieces = tb.get_train_batch(batch_size, seq_length)
			word_in, pos_in, word_target, pos_target = train_pieces
			train_perp = np.exp(lm.validate(word_in, word_target))

			print('perplexity of validation sequence:', perplexity[-1])
			print('perplexity of training sequence:', train_perp)
			save_text(str(perplexity[-1]) + "," + str(num_steps),
				save_dir, 'lmperplexity')
			save_text(str(train_perp) + "," + str(num_steps),
				save_dir, 'lmtrainperplexity')
			loss = []

		if (num_steps % sample_step == 0) and num_steps > 0:
			validation_pieces = tb.get_validation_batch(32, 50)
			word_in, pos_in, word_target, pos_target = validation_pieces
			test_seq_loss.append(lm.validate(word_in, word_target))
			vperp.append(np.exp(test_seq_loss[-1]))

			new_lr = decrease_lr(vperp, halving_threshold,
				0.5, learning_rate)
			if new_lr != learning_rate:
				learning_rate = new_lr
				num_halvings +=1
				print('\tlearning rate halved!')
				save_text(str(learning_rate) + "," + str(num_steps),
					save_dir, 'halvings')

			print('\tnumber of learning rate halvings:', num_halvings)
			word_seed, pos_seed = tb.get_test_batch(1, 5)
			generated_text, generated_syntax = \
				tb.one_hots_to_words(lm.run(word_seed, 300))
			print(generated_text)
			if len(generated_syntax) > 0:
				save_text(str(num_steps) + ":", save_dir, 'lmgensyntax')
				save_text(generated_syntax + ":", save_dir, 'lmgensyntax')
			save_text(str(num_steps) + ":", save_dir, 'lmgentext')
			save_text(generated_text, save_dir, 'lmgentext')
			
		num_steps += 1

if __name__ == "__main__":
	main(sys.argv[1:])