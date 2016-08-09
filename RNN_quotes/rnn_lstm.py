import numpy as np
import theano as theano
import theano.tensor as T

class LSTM_rnn:

    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=-1):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Initialize the network parameters
        E = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
        U = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (6, hidden_dim, hidden_dim))
        W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (6, hidden_dim, hidden_dim))
        V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (word_dim, hidden_dim))
        b = np.zeros((6, hidden_dim))
        c = np.zeros(word_dim)
        self.mE = theano.shared(name='mE', value=np.zeros(E.shape).astype(theano.config.floatX))
        self.mU = theano.shared(name='mU', value=np.zeros(U.shape).astype(theano.config.floatX))
        self.mV = theano.shared(name='mV', value=np.zeros(V.shape).astype(theano.config.floatX))
        self.mW = theano.shared(name='mW', value=np.zeros(W.shape).astype(theano.config.floatX))
        self.mb = theano.shared(name='mb', value=np.zeros(b.shape).astype(theano.config.floatX))
        self.mc = theano.shared(name='mc', value=np.zeros(c.shape).astype(theano.config.floatX))

        self.E = theano.shared(name='E', value=E.astype(theano.config.floatX))
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.b = theano.shared(name='b', value=b.astype(theano.config.floatX))
        self.c = theano.shared(name='c', value=c.astype(theano.config.floatX))
        self.theano = {}
        self.__theano_build__()

    def __theano_build__(self):
        E, V, U, W, b, c = self.E, self.V, self.U, self.W, self.b, self.c
        x = T.ivector('x')
        y = T.ivector('y')
        learning_rate = T.scalar('learning_rate')

        def forward_prop_step(x_t, s_t1_prev, s_t2_prev):
            x_e = E[:, x_t]
            #L1
            z_t1 = T.nnet.hard_sigmoid(U[0].dot(x_e) + W[0].dot(s_t1_prev) + b[0])
            r_t1 = T.nnet.hard_sigmoid(U[1].dot(x_e) + W[1].dot(s_t1_prev) + b[1])
            c_t1 = T.tanh(U[2].dot(x_e) + W[2].dot(s_t1_prev * r_t1) + b[2])
            s_t1 = (T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev
            # L2
            z_t2 = T.nnet.hard_sigmoid(U[3].dot(s_t1) + W[3].dot(s_t2_prev) + b[3])
            r_t2 = T.nnet.hard_sigmoid(U[4].dot(s_t1) + W[4].dot(s_t2_prev) + b[4])
            c_t2 = T.tanh(U[5].dot(s_t1) + W[5].dot(s_t2_prev * r_t2) + b[5])
            s_t2 = (T.ones_like(z_t2) - z_t2) * c_t2 + z_t2 * s_t2_prev
            o_t = T.nnet.softmax(V.dot(s_t2) + c)[0]
            return [o_t, s_t1, s_t2]

        [o, s, s2], updates = theano.scan(forward_prop_step,sequences=x,truncate_gradient=self.bptt_truncate,
            outputs_info=[None,dict(initial=T.zeros(self.hidden_dim)),dict(initial=T.zeros(self.hidden_dim))])
        prediction = T.argmax(o, axis=1)
        cost = T.sum(T.nnet.categorical_crossentropy(o, y))

        dE = T.grad(cost, E)
        dU = T.grad(cost, U)
        dW = T.grad(cost, W)
        db = T.grad(cost, b)
        dV = T.grad(cost, V)
        dc = T.grad(cost, c)
        self.predict = theano.function([x], o)
        self.predict_class = theano.function([x], prediction)
        self.ce_error = theano.function([x, y], cost)
        self.bptt = theano.function([x, y], [dE, dU, dW, db, dV, dc])
        decay = T.scalar('decay')
        mE = decay * self.mE + (1 - decay) * dE ** 2
        mU = decay * self.mU + (1 - decay) * dU ** 2
        mW = decay * self.mW + (1 - decay) * dW ** 2
        mV = decay * self.mV + (1 - decay) * dV ** 2
        mb = decay * self.mb + (1 - decay) * db ** 2
        mc = decay * self.mc + (1 - decay) * dc ** 2

        self.oneSGD = theano.function([x, y, learning_rate, theano.Param(decay, default=0.9)],[],
            updates=[(E, E - learning_rate * dE / T.sqrt(mE + 1e-6)), (U, U - learning_rate * dU / T.sqrt(mU + 1e-6)),
                     (W, W - learning_rate * dW / T.sqrt(mW + 1e-6)),(V, V - learning_rate * dV / T.sqrt(mV + 1e-6)),
                     (b, b - learning_rate * db / T.sqrt(mb + 1e-6)),(c, c - learning_rate * dc / T.sqrt(mc + 1e-6)),
                     (self.mE, mE),(self.mU, mU),(self.mW, mW),(self.mV, mV),(self.mb, mb),(self.mc, mc)])



    def totalLoss(self, X, Y):
        return np.sum([self.ce_error(x, y) for x, y in zip(X, Y)])

    def lossCalc(self, X, Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.totalLoss(X, Y) / float(num_words)


    def totalSGD(self, X_train, y_train, learning_rate=0.005, nepoch=100, decay=0.9,
        callback_every=10000, callback=None):
        num_examples_seen = 0
        for epoch in range(nepoch):
            for i in np.random.permutation(len(y_train)):
                self.oneSGD(X_train[i], y_train[i], learning_rate, decay)
                num_examples_seen += 1


    def getSentence(self,word_to_index,index_to_word):
        new_sentence = [word_to_index["SENTENCE_START"]]
        index=0
        while not (new_sentence[-1] == word_to_index["SENTENCE_END"]):
            if index<20:
                next_word_probs = self.predict(new_sentence)[-1]
                sampled_word = word_to_index["UNKNOWN_TOKEN"]
                while sampled_word == word_to_index["UNKNOWN_TOKEN"]:
                    samples = np.random.multinomial(1, next_word_probs)
                    sampled_word = np.argmax(samples)
                new_sentence.append(sampled_word)
                index=index+1
            else:
                new_sentence.append( word_to_index["SENTENCE_END"])
        sent= ""
        for x in new_sentence[1:-1]:
            if (index_to_word[x] == "'ve"):
                sent = sent + " " + "have"
            elif (index_to_word[x] == "'s"):
                sent = sent + "'s"
            elif (index_to_word[x] == "''") or (index_to_word[x] == "--") or (index_to_word[x] == ";") \
                    or (index_to_word[x] == "``") or (index_to_word[x] == "/") or (index_to_word[x] == ".") \
                    or (index_to_word[x] == ">") or (index_to_word[x] == "<")or (index_to_word[x] == "(")\
                    or (index_to_word[x] == ")"):
                continue
            else:
                sent = sent + " " + index_to_word[x]
        return sent


