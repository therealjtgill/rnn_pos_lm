mapped = open('reduce1.txt', 'r')

mapped_lines = mapped.readlines()
num_items = float(len(mapped_lines))
print(num_items)

with open('reduce2.txt', 'w') as f:
	current_key = None
	count = 1.

	for line in mapped_lines:
		#if line.count(',') == 1:
		#	line_split = line.split(',')
		#else:
		#	line_split = line.split(',', 2)
		#line_split = line.split('\t')

		if current_key != line:
			if current_key != None:
				f.write(current_key[0:-1] + "\t" + str(count/num_items) + "\n")
			current_key = line
			count = 1.
		else:
			count += 1.

