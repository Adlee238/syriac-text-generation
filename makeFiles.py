'''
File: makeFiles.oy
Description: This program reads through the 'src' directory (the corpus), and creates a training and testing file for the model.
'''
import numpy as np
import os.path

# Create corpus array
all_corpus_lines = []
for i in range(580):
    filename = 'src/' + str(i) + '.txt'
    if os.path.isfile(filename):
        file = open(filename, "r")
        file_lines = file.readlines()
        for line in file_lines:
            all_corpus_lines.append(line)
        file.close()

# shuffle corpus array
np.random.shuffle(all_corpus_lines)

# Determine cut-off point to separate training and testing data
n_corpus_lines = len(all_corpus_lines)
n_training_lines = n_corpus_lines * 9 // 10
n_testing_lines = n_corpus_lines - n_training_lines

# create training file
training_lines = all_corpus_lines[: n_training_lines]
training_file = open("training.txt", "w")
training_file.writelines(training_lines)
training_file.close()

# create testing file
testing_lines = all_corpus_lines[n_training_lines: n_corpus_lines]
testing_file = open("testing.txt", "w")
testing_file.writelines(testing_lines)
testing_file.close()