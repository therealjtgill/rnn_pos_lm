import numpy as np
import tensorflow as tf
from utils import *
from corpus_gen import *

class word_embedding(object):

	def __init__(self, learning_rate, vocab_size, embedding_size, 
		num_lstm_units, seq_length, session):

		self.session = session
		lr = learning_rate
		V = vocab_size
		E = embedding_size
		S = seq_length
		N = num_lstm_units
		dt = tf.float32

		
		with tf.variable_scope('embedding'):

			self.feed_in = tf.placeholder(dtype=dt, shape=[None, S, V])
			self.feed_out = tf.placeholder(dtype=dt, shape=[None, S, V])

			self.weight_embed = tf.Variable(tf.random_normal([V, E]),
				name='emb_w')
			self.weight_output = tf.Variable(tf.random_normal([N, V]))

			self.bias_output = tf.Variable(tf.random_normal([V,]))

			batch_size = tf.shape(self.feed_in)[0]
			sequence_length = tf.shape(self.feed_in)[1]

			embed_exp = tf.expand_dims(self.weight_embed, axis=0)
			embed_tile = tf.tile(embed_exp, [batch_size, 1, 1])

			layer_embedded = tf.tanh(tf.matmul(self.feed_in, embed_tile))

			lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=N)
			lstm_out_raw, _ = tf.nn.dynamic_rnn(
				cell=lstm_cell, inputs=layer_embedded, dtype=dt)

			feed_out_flat = tf.reshape(self.feed_out, [-1, V])
			lstm_out_raw_flat = tf.reshape(lstm_out_raw, [-1, N])
			logits_flat = tf.matmul(lstm_out_raw_flat, self.weight_output) + \
				self.bias_output

			layer_output = tf.nn.softmax(
				tf.reshape(logits_flat, [batch_size, sequence_length, V]))

			self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
				logits=logits_flat, labels=feed_out_flat))

			self.train_op = tf.train.RMSPropOptimizer(
				learning_rate=lr).minimize(self.loss)

	def train(self, batch_in, batch_out):

		feeds = {
			self.feed_in:batch_in,
			self.feed_out:batch_out,
		}

		fetches = [self.loss, self.train_op]

		cost, _ = self.session.run(fetches, feed_dict=feeds)

		return cost

# Similar to the catastrophic forgetting RNN, we want this to train until
# perplexity asymptotically tails off to zero derivative.
# Except what I'm actually gonna do is just train it for a static number of
# timesteps cuz i'm lzy.

def main():
	top_k = 10000
	sess = tf.Session()
	embedding = word_embedding(0.0001, top_k, 100, 100, 5, sess)
	sess.run(tf.global_variables_initializer())

	tb = corpus(top_k)
	#word_vocab = tb.get_word_vocabulary()
	#pos_vocab = tb.get_pos_vocabulary()
	loss = []

	for num_steps in range(30000):
		batch_x, _, batch_y, _ = tb.get_train_batch(64, 5)
		loss.append(embedding.train(batch_x, batch_y))
		if num_steps % 100 == 0 and num_steps > 0:
			print "perplexity:", np.exp(sum(loss)/100)
			print "step number:", num_steps
			loss = []

	train_vars = tf.trainable_variables()
	save_vars = [i for i in train_vars if '/emb_w' in i.name]

	enc_params = sess.run(save_vars)
	print("enc_params:", enc_params)
	np.savez('wordembedding', enc_params[0])

if __name__ == "__main__":
	main()