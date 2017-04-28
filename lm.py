from __future__ import print_function
import numpy as np
import tensorflow as tf

from tensorflow.contrib.rnn import BasicLSTMCell, MultiRNNCell, LSTMStateTuple

class WordLM(object):

	def __init__(self, session, scope_name, word_vocab_size,
		num_lstm_layers=2, num_lstm_units=256):
		self.session = session
		self.num_lstm_layers = num_lstm_layers
		self.num_lstm_units = num_lstm_units
		self.word_vocab_size = word_vocab_size
		
		word_embed = np.load('word_embedding.npz')

		VW = word_vocab_size
		L = num_lstm_layers
		N = num_lstm_units
		dt = tf.float32
		#lr = learning_rate

		with tf.variable_scope(scope_name):

			self.feed_in_word = tf.placeholder(dtype=dt,
				shape=[None, None, VW])

			self.feed_out_word = tf.placeholder(dtype=dt,
				shape=[None, None, VW])
			self.learning_rate = tf.placeholder(dtype=dt,
				shape=[])

			lr = self.learning_rate

			self.lstm_states = []
			for i in range(L):
				self.lstm_states.append(
					(tf.placeholder(dtype=dt, shape=(None, N)),
					tf.placeholder(dtype=dt, shape=(None, N))))
			print(len(self.lstm_states), len(self.lstm_states[0]))
				
			rnn_tuple_states = tuple([LSTMStateTuple(r[0], r[1]) \
				for r in self.lstm_states])

			batch_size = tf.shape(self.feed_in_word)[0]
			seq_length = tf.shape(self.feed_in_word)[1]

			weight_word = tf.constant(word_embed['arr_0'])

			weight_word_tile = \
				self.expand_and_tile(weight_word, 0, [batch_size, 1, 1])

			layer_full_embed = self.tanh_layer(self.feed_in_word,
				weight_word_tile)
			#layer_full_embed = tf.concat([word_embed, pos_embed], axis=2)

			lstm_cells = [BasicLSTMCell(num_units=N) for _ in range(L)]

			self.multi_lstm = MultiRNNCell(lstm_cells)
			self.zero_states = self.multi_lstm.zero_state

			self.lstm_out_raw, self.lstm_last_state = \
				tf.nn.dynamic_rnn(cell=self.multi_lstm, inputs=layer_full_embed,
					initial_state=rnn_tuple_states, dtype=dt)

			lstm_out_flat = tf.reshape(self.lstm_out_raw, [-1, N])
			weight_output = tf.Variable(tf.random_normal([N, VW]))
			bias_output = tf.Variable(tf.random_normal([VW,]))

			logits_flat = tf.matmul(lstm_out_flat, weight_output) + bias_output

			feed_out_flat = tf.reshape(self.feed_out_word, [-1, VW])

			#feed_out_flat = tf.concat([feed_out_word_flat, feed_out_pos_flat], 
			#	axis=1)

			#self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
			#	logits=logits_flat, labels=feed_out_flat))
			
			self.word_logits = tf.reshape(logits_flat, 
				[batch_size, seq_length, VW])
			#pos_logits, word_logits = tf.split(logits, [VP, VW], axis=2)
			word_logits_flat = tf.reshape(self.word_logits, [-1, VW])

			self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
				logits=word_logits_flat, labels=feed_out_flat)
			self.loss = tf.reduce_mean(self.cross_entropy)

			#sum_loss = tf.reduce_sum(self.cross_entropy, axis=-1)
			#self.loss = tf.reduce_mean(sum_loss)

			self.train_op = tf.train.RMSPropOptimizer(
				learning_rate=lr).minimize(self.loss)

			self.word_prob = tf.nn.softmax(self.word_logits)

			self.perplexity = tf.exp(self.loss)

	def expand_and_tile(self, tensor, expand_axis, tile_pattern):
		tensor_expanded = tf.expand_dims(tensor, axis=expand_axis)
		tensor_tiled = tf.tile(tensor_expanded, tile_pattern)
		return tensor_tiled

	def tanh_layer(self, in_, weight, bias=None):
		local_field = tf.matmul(in_, weight)
		if bias != None:
			local_field += bias
		layer_out = tf.tanh(local_field)
		return layer_out

	def train(self, word_ins, pos_ins, word_outs, pos_outs, learning_rate):

		dt = tf.float32
		batch_size = word_ins.shape[0]
		seq_length = word_ins.shape[1]
		zero_states = self.session.run(self.zero_states(batch_size, dtype=dt))
		#print(len(zero_states), len(zero_states[0]))

		feeds = {
			self.feed_in_word:word_ins,
			self.feed_out_word:word_outs,
			self.learning_rate:learning_rate
		}

		for i in range(self.num_lstm_layers):
			#print(i)
			feeds[self.lstm_states[i][0]] = zero_states[i][0]
			feeds[self.lstm_states[i][1]] = zero_states[i][1]

		fetches = [self.loss, self.train_op]

		loss, _ = self.session.run(fetches, feed_dict=feeds)
		#print('training loss:', loss)

		return loss

	def validate(self, word_ins, pos_ins, word_outs, pos_outs):

		dt = tf.float32
		batch_size = word_ins.shape[0]
		seq_length = word_ins.shape[1]
		zero_states = self.session.run(self.zero_states(batch_size, dtype=dt))

		feeds = {
			self.feed_in_word:word_ins,
			self.feed_out_word:word_outs,
		}

		for i in range(self.num_lstm_layers):
			feeds[self.lstm_states[i][0]] = zero_states[i][0]
			feeds[self.lstm_states[i][1]] = zero_states[i][1]

		fetches = [self.loss, self.cross_entropy, self.perplexity]

		loss, cross_entropy, perplexity = \
			self.session.run(fetches, feed_dict=feeds)

		#print('cross entropy shape:', cross_entropy.shape)
		#print(cross_entropy)
		#print('sum cross entropy:', np.sum(cross_entropy))
		#print(np.sum(cross_entropy)/(batch_size*seq_length))
		#print('internal perplexity calc:', perplexity)
		#print('reported loss:', loss)

		return loss

	def run(self, word_ins, _, num_steps=100):

		dt = tf.float32
		batch_size = 1
		seq_length = word_ins.shape[0]
		#num_lstm_layers = self.num_lstm_layers

		lstm_next_state = self.session.run(
			self.multi_lstm.zero_state(batch_size, dt))

		word_probs = []
		pos_probs = []
		feeds = {self.feed_in_word:word_ins}
		fetches = [self.word_prob, self.lstm_last_state]

		for i in range(self.num_lstm_layers):
			feeds[self.lstm_states[i][0]] = lstm_next_state[i][0]
			feeds[self.lstm_states[i][1]] = lstm_next_state[i][1]

		word_prob, lstm_next_state = self.session.run(fetches, feed_dict=feeds)

		for i in range(num_steps):
			# vvv need to implement the soft_prob_to_one_hot method
			#word_one_hot = soft_prob_to_one_hot(word_prob[0][-1])
			#pos_one_hot = soft_prob_to_one_hot(pos_prob[0][-1])
			word_one_hot = np.zeros_like(word_prob[0][-1])
			word_index = np.random.choice(range(self.word_vocab_size),
				p=word_prob[0][-1])
			word_one_hot[word_index] = 1.

			word_probs.append(word_one_hot)

			feeds[self.feed_in_word] = [[word_one_hot]]

			for j in range(self.num_lstm_layers):
				feeds[self.lstm_states[j][0]] = lstm_next_state[j][0]
				feeds[self.lstm_states[j][1]] = lstm_next_state[j][1]

			word_prob, lstm_next_state = self.session.run(
				fetches, feed_dict=feeds)

			#print(word_prob.shape, pos_prob.shape)

		return word_probs, pos_probs

	def save(self, filename, save_dir=''):
		save_path = os.path.join(save_dir, filename)
		self.saver.save(self.session, save_path + '.ckpt')

	def set_saver(self, saver):
		self.saver = saver

'''
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

def main():
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
	lm = WordLM(sess, 'lm', word_vocab_size)
	sess.run(tf.global_variables_initializer())

	test_seq_loss = []
	perplexity = []
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
			
			new_lr = decrease_lr(perplexity, halving_threshold,
				0.5, learning_rate)
			if new_lr != learning_rate:
				learning_rate = new_lr
				num_halvings +=1
				print('\tlearning rate halved!')
				save_text(str(learning_rate) + "," + str(num_steps),
					save_dir, 'halvings')

			print('\tnumber of learning rate halvings:', num_halvings)
			word_seed, pos_seed = tb.get_test_batch(1, 5)
			print(tb.one_hots_to_words(lm.run(word_seed)[0]))
			
		num_steps += 1

if __name__ == "__main__":
	main()

'''