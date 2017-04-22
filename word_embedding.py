import numpy as np
import tensorflow as tf

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

		
		with tf.scope('embedding'):

			self.feed_in = tf.placeholder(dtype=dt, [None, S, V])
			self.feed_out = tf.placeholder(dtype=dt, [None, S, V])

			self.weight_embed = tf.Variable(tf.random_normal([V, E]))
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
			logits_flat = tf.matmul(lstm_out_raw, self.weight_output) + \
				self.bias_output

			layer_output = tf.nn.softmax(
				tf.reshape(logits_flat, [batch_size, sequence_length, V]))

			self.loss = tf.nn.softmax_cross_entropy_with_logits(
				logits=logits_flat, labels=feed_out_flat)

			self.train_op = tf.train.RMSPropOptimizer(
				learning_rate=lr).minimize(self.loss)

	def train(self, batch_in, batch_out, lr=0.0001):

		feeds = {
			self.feed_in:batch_in,
			self.feed_out:batch_out,
		}

		fetches = [self.loss, self.train_op]

		cost, _ = self.session.run(fetches, feed_dict=feeds)

		return cost

# Similar to the catastrophic forgetting RNN, we want this to train until
# perplexity asymptotically tails off to zero derivative.