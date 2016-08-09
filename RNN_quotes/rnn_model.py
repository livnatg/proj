import numpy as np
import sys

from datetime import datetime
import operator


class Simpel_rnn:
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.U = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))
        self.learn = 0

    def forward_propagation(self, x):
        # The total number of time steps
        T = len(x)
        s = np.zeros((T + 1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        o = np.zeros((T, self.word_dim))
        for t in np.arange(T):
            s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t - 1]))
            temp = np.exp(self.V.dot(s[t]) - np.max(self.V.dot(s[t])))
            o[t] = temp / np.sum(temp)
        return [o, s]

    def totalLoss(self, x, y):
        L = 0
        # For each sentence...
        for i in np.arange(len(y)):
            o, s = self.forward_propagation(x[i])
            # We only care about our prediction of the "correct" words
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            # Add to the loss based on how off we were
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L

    def lossCalc(self, x, y):
        # Divide the total loss by the number of training examples
        N = np.sum((len(y_i) for y_i in y))
        return self.totalLoss(x, y) / N


    def bptt(self, x, y):
        T = len(y)
        # Perform forward propagation
        o, s = self.forward_propagation(x)
        # We accumulate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1.
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            for bptt_step in np.arange(max(0, t - self.bptt_truncate), t + 1)[::-1]:
                dLdW += np.outer(delta_t, s[bptt_step - 1])
                dLdU[:, x[bptt_step]] += delta_t
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step - 1] ** 2)
        return [dLdU, dLdV, dLdW]

    def oneSGD(self, x, y, learning_rate):
        # Calculate the gradients
        dLdU, dLdV, dLdW = self.bptt(x, y)
        # Change parameters according to gradients and learning rate
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW

    def totalSGD(self, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
        # We keep track of the losses so we can plot them later
        losses = []
        num_examples_seen = 0
        for epoch in range(nepoch):
            # Optionally evaluate the loss
            if (epoch % evaluate_loss_after == 0):
                loss = self.lossCalc(X_train, y_train)
                losses.append((num_examples_seen, loss))
                if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                    learning_rate = learning_rate * 0.5
                sys.stdout.flush()
            for i in range(len(y_train)):
                self.oneSGD(X_train[i], y_train[i], learning_rate)
                num_examples_seen += 1
        self.learn=learning_rate


    def getSentence(self,word_to_index,index_to_word):
        # We start the sentence with the start token
        new_sentence = [word_to_index["SENTENCE_START"]]
        i = 0;
        while not (new_sentence[-1] == word_to_index["SENTENCE_END"]):
            if i < 20:
                next_word_probs, s = self.forward_propagation(new_sentence)
                sampled_word = word_to_index["UNKNOWN_TOKEN"]
                # We don't want to sample unknown words
                while sampled_word == word_to_index["UNKNOWN_TOKEN"]:
                    samples = np.random.multinomial(1, next_word_probs[0])
                    sampled_word = np.argmax(samples)
                new_sentence.append(sampled_word)
                i += 1
            else:
                new_sentence.append(word_to_index["SENTENCE_END"])
        str = ""
        for x in new_sentence[1:-1]:
            if (index_to_word[x] == "'ve"):
                str = str + " " + "have"
            elif (index_to_word[x] == "'s"):
                str = str + "'s"
            elif (index_to_word[x] == "''") or (index_to_word[x] == "--") or (index_to_word[x] == ";") \
                    or (index_to_word[x] == "``") or (index_to_word[x] == "/") or (index_to_word[x] == ".") \
                    or (index_to_word[x] == ">") or (index_to_word[x] == "<")or (index_to_word[x] == "(")\
                    or (index_to_word[x] == ")"):
                continue
            else:
                str = str + " " + index_to_word[x]
        return str