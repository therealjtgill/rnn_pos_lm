from corpus_gen import *
import numpy as np

def generate_joint_matrix(filename):
	stuff = corpus(10000)
	joint = np.zeros((stuff.word_vocab_size, stuff.pos_vocab_size))
	with open(filename) as f:
		for line in f:
			vocab_index_str, stuff = line.split(',')
			pos_index_str, prob_str = stuff.split('\t')

			vocab_index = int(vocab_index_str)
			pos_index = int(pos_index_str)
			prob = float(prob_str.strip())
			joint[vocab_index, pos_index] = prob

	return joint

def generate_pw_conditional_matrix(filename):
	joint = generate_joint_matrix(filename)
	pw_conditional = np.zeros_like(joint)

	rows, columns = joint.shape
	w_marginal = np.sum(joint, axis=1)
	for r in range(rows):
		for c in range(columns):
			if w_marginal[r] != 0:
				pw_conditional[r][c] = joint[r][c]/w_marginal[r]

	return pw_conditional, w_marginal

def generate_wp_conditional_matrix(filename):
	joint = generate_joint_matrix(filename)
	wp_conditional = np.zeros_like(joint)

	rows, columns = joint.shape
	p_marginal = np.sum(joint, axis=0)
	for r in range(rows):
		for c in range(columns):
			if p_marginal[c] != 0:
				wp_conditional[r][c] = joint[r][c]/p_marginal[c]

	return wp_conditional, p_marginal

def get_count_probs(filename):
	joint = generate_joint_matrix(filename)
	pw_conditional, w_marginal = generate_pw_conditional_matrix(filename)
	wp_conditional, p_marginal = generate_wp_conditional_matrix(filename)

	save_dict = {
		'joint': joint,
		'pwcond': pw_conditional,
		'wpcond': wp_conditional,
		'wmarg': w_marginal,
		'pmarg': p_marginal
	}

	np.savez('firstordercalcs', save_dict)

get_count_probs('reduce3.txt')