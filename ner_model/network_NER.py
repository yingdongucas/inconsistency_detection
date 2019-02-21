import lasagne
import numpy as np
import theano.tensor as T
import theano
import pdb
import pickle
import matplotlib.pyplot as plt
import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import config
plt.switch_backend('agg')

import logging

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)


CRF_INIT = True
COST = True
COST_CONST = 5.0

PERIOD = 100  # 100, 4000
MIN_PERIOD = 4000

NOISE = False
NOISE_RATE = 0.1

REDUCE = False


def print_arr(y):
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            print(y[i, j], )
        print(' ')


def theano_logsumexp(x, axis=None):
    xmax = x.max(axis=axis, keepdims=True)
    xmax_ = x.max(axis=axis)
    return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))


class CRFLayer(lasagne.layers.Layer):

    def __init__(self, incoming, num_classes, W_sim=lasagne.init.GlorotUniform(), W=lasagne.init.GlorotUniform(), \
                 W_init=lasagne.init.GlorotUniform(), mask_input=None, label_input=None, **kwargs):

        super(CRFLayer, self).__init__(incoming, **kwargs)
        num_inputs = self.input_shape[-1]
        self.num_classes = num_classes
        self.W_sim = self.add_param(W_sim, (num_classes, num_classes))
        self.W = self.add_param(W, (num_inputs, num_classes))
        self.mask_input = mask_input
        self.label_input = label_input
        if CRF_INIT:
            self.W_init = self.add_param(W_init, (1, num_classes))
        else:
            self.W_init = None

    def get_output_shape_for(self, input_shape):
        return (1,)

    def get_output_for(self, input, **kwargs):
        def norm_fn(f, mask, label, previous, W_sim):
            # f: inst * class, mask: inst, previous: inst * class, W_sim: class * class
            next = previous.dimshuffle(0, 1, 'x') + f.dimshuffle(0, 'x', 1) + W_sim.dimshuffle('x', 0, 1)
            if COST:
                next = next + COST_CONST * (1.0 - T.extra_ops.to_one_hot(label, self.num_classes).dimshuffle(0, 'x', 1))
            # next: inst * prev * cur
            next = theano_logsumexp(next, axis=1)
            # next: inst * class
            mask = mask.dimshuffle(0, 'x')
            next = previous * (1.0 - mask) + next * mask
            return next

        f = T.dot(input, self.W)
        # f: inst * time * class

        initial = f[:, 0, :]
        if CRF_INIT:
            initial = initial + self.W_init[0].dimshuffle('x', 0)
        if COST:
            initial = initial + COST_CONST * (1.0 - T.extra_ops.to_one_hot(self.label_input[:, 0], self.num_classes))
        outputs, _ = theano.scan(fn=norm_fn, \
                                 sequences=[f.dimshuffle(1, 0, 2)[1:], self.mask_input.dimshuffle(1, 0)[1:],
                                            self.label_input.dimshuffle(1, 0)[1:]], \
                                 outputs_info=initial, non_sequences=[self.W_sim], strict=True)
        norm = T.sum(theano_logsumexp(outputs[-1], axis=1))

        f_pot = (f.reshape((-1, f.shape[-1]))[
                     T.arange(f.shape[0] * f.shape[1]), self.label_input.flatten()] * self.mask_input.flatten()).sum()
        if CRF_INIT:
            f_pot += self.W_init[0][self.label_input[:, 0]].sum()

        labels = self.label_input
        # labels: inst * time
        shift_labels = T.roll(labels, -1, axis=1)
        mask = self.mask_input
        # mask : inst * time
        shift_mask = T.roll(mask, -1, axis=1)

        g_pot = (self.W_sim[labels.flatten(), shift_labels.flatten()] * mask.flatten() * shift_mask.flatten()).sum()

        return - (f_pot + g_pot - norm) / f.shape[0]


class CRFDecodeLayer(lasagne.layers.Layer):

    def __init__(self, incoming, num_classes, W_sim=lasagne.init.GlorotUniform(), W=lasagne.init.GlorotUniform(), \
                 W_init=lasagne.init.GlorotUniform(), mask_input=None, **kwargs):

        super(CRFDecodeLayer, self).__init__(incoming, **kwargs)
        num_inputs = self.input_shape[-1]
        self.W_sim = self.add_param(W_sim, (num_classes, num_classes))
        self.W = self.add_param(W, (num_inputs, num_classes))
        self.mask_input = mask_input
        if CRF_INIT:
            self.W_init = self.add_param(W_init, (1, num_classes))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1])

    def get_output_for(self, input, **kwargs):
        def max_fn(f, mask, prev_score, prev_back, W_sim):
            next_score = prev_score.dimshuffle(0, 1, 'x') + f.dimshuffle(0, 'x', 1) + W_sim.dimshuffle('x', 0, 1)
            next_back = T.argmax(next_score, axis=1)
            next_score = T.max(next_score, axis=1)
            mask = mask.dimshuffle(0, 'x')
            next_score = next_score * mask + prev_score * (1.0 - mask)
            next_back = next_back * mask + prev_back * (1.0 - mask)
            next_back = T.cast(next_back, 'int32')
            return [next_score, next_back]

        def produce_fn(back, mask, prev_py):
            # back: inst * class, prev_py: inst, mask: inst
            next_py = back[T.arange(prev_py.shape[0]), prev_py]
            next_py = mask * next_py + (1.0 - mask) * prev_py
            next_py = T.cast(next_py, 'int32')
            return next_py

        f = T.dot(input, self.W)

        init_score, init_back = f[:, 0, :], T.zeros_like(f[:, 0, :], dtype='int32')
        if CRF_INIT:
            init_score = init_score + self.W_init[0].dimshuffle('x', 0)
        ([scores, backs], _) = theano.scan(fn=max_fn, \
                                           sequences=[f.dimshuffle(1, 0, 2)[1:], self.mask_input.dimshuffle(1, 0)[1:]], \
                                           outputs_info=[init_score, init_back], non_sequences=[self.W_sim],
                                           strict=True)

        init_py = T.argmax(scores[-1], axis=1)
        init_py = T.cast(init_py, 'int32')
        # init_py: inst, backs: time * inst * class
        pys, _ = theano.scan(fn=produce_fn, \
                             sequences=[backs, self.mask_input.dimshuffle(1, 0)[1:]], outputs_info=[init_py],
                             go_backwards=True)
        # pys: (rev_time - 1) * inst
        pys = pys.dimshuffle(1, 0)[:, :: -1]
        # pys : inst * (time - 1)
        return T.concatenate([pys, init_py.dimshuffle(0, 'x')], axis=1)


class ElemwiseMergeLayer(lasagne.layers.MergeLayer):

    def __init__(self, incomings, merge_function, **kwargs):
        super(ElemwiseMergeLayer, self).__init__(incomings, **kwargs)
        self.merge_function = merge_function

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        output = None
        for input in inputs:
            if output is not None:
                output = self.merge_function(output, input)
            else:
                output = input
        return output


class cnn_rnn:

    def __init__(self, char_cnt, label_cnt, word_cnt, save_path, category, use_crf=False, ind2word=None, lemma_cnt=0, pos_cnt=0,
                 batch_size=50, dropout_rate=0.5, char_hidden_size=80, hidden_size=300,
                 double_layer=True, char_double_layer=True, char_rnn=True, transfer=False):
        lasagne.random.set_rng(np.random)
        np.random.seed(13)  # 13

        logging.info('batch_size: ' + str(batch_size))
        logging.info('dropout_rate: ' + str(dropout_rate))
        logging.info('char_hidden_size: ' + str(char_hidden_size))
        logging.info('hidden_size: ' + str(hidden_size))
        logging.info('double_layer: ' + str(double_layer))
        logging.info('char_double_layer: ' + str(char_double_layer))
        logging.info('char_rnn: ' + str(char_rnn) + '\n\n\n')

        # fine tune

        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.char_hidden_size = char_hidden_size
        self.hidden_size = hidden_size

        self.double_layer = double_layer
        self.char_double_layer = char_double_layer

        self.char_rnn = char_rnn  # true

        # fine tune

        self.char_cnt = char_cnt
        self.label_cnt = label_cnt
        self.embedding_size = 50  # 25, 50
        self.learning_rate = 1e-2  # 1e-2
        self.f_sizes = [3, 4, 5]
        self.filter_num = 200
        self.char_f_sizes = [3, 4, 5]
        self.char_filter_num = 20
        self.word_cnt = word_cnt
        self.w_embedding_size = config.word_emb_dim_ner  # 50, 64, 300 # 100?
        self.epoch = 200
        self.gaze_embedding_size = 7
        self.ind2word = ind2word
        self.test_batch_size = 100
        logging.info('in model, use_crf is ' + str(use_crf))
        self.use_crf = use_crf
        self.dropout = True  # feb27
        self.use_gaze = True  # True

        self.char_cnn = True
        self.char_rnn = True  # true feb27
        self.word_cnn = True
        self.word_embedding = True  # true

        self.use_lemma = False  # true
        self.lemma_cnt = lemma_cnt
        self.lemma_size = 10

        self.use_pos = False  # true
        self.pos_cnt = pos_cnt
        self.pos_size = 10

        self.reduce_size = 15

        self.tanh = True  # False
        self.tanh_size = 150  # 150
        self.joint = False  # False
        self.top_joint = False  # False
        self.very_top_joint = False

        self.min_epoch = 2

        self.save_path = save_path
        self.transfer = transfer
        self.category = category

    def add_data(self, x, y, m, wx, cm, gaze, tx, ty, tm, twx, tcm, tgaze, test_data_dic, lemma=None, tlemma=None,
                 pos=None, tpos=None):
        self.x, self.y, self.m, self.wx, self.tx, self.ty, self.tm, self.twx = x, y, m, wx, tx, ty, tm, twx
        self.cm, self.tcm = cm, tcm
        self.gaze, self.tgaze = gaze, tgaze
        if self.gaze is None:
            self.use_gaze = False

        self.test_data_dic = test_data_dic
        self.lemma, self.tlemma = lemma, tlemma
        if self.lemma is None:
            self.use_lemma = False
        self.pos, self.tpos = pos, tpos
        if self.pos is None:
            self.use_pos = False

    def build(self):
        x_sym = T.itensor3('x')
        y_sym = T.imatrix('y')
        m_sym = T.matrix('mask')
        wx_sym = T.imatrix('wx')
        lemma_sym = T.imatrix('lemma')
        pos_sym = T.imatrix('pos')
        cm_sym = T.tensor3('cmask')
        gaze_sym = T.tensor3('gaze')

        l = lasagne.layers.InputLayer(shape=(None, self.x.shape[1], self.x.shape[2]), input_var=x_sym)
        l = lasagne.layers.EmbeddingLayer(l, self.char_cnt, self.embedding_size)
        l_char = lasagne.layers.ReshapeLayer(l, (-1, [2], [3]))
        l_c_mask = lasagne.layers.InputLayer(shape=(None, self.x.shape[1], self.x.shape[2]), input_var=cm_sym)
        l_c_mask = lasagne.layers.ReshapeLayer(l_c_mask, (-1, [2]))
        l_gru = lasagne.layers.GRULayer(l_char, self.char_hidden_size, mask_input=l_c_mask)
        l_gru_2 = lasagne.layers.GRULayer(l_char, self.char_hidden_size, mask_input=l_c_mask, backwards=True)
        if self.char_double_layer:
            l_char_rnn = lasagne.layers.ConcatLayer([l_gru, l_gru_2], axis=2)
            l_gru = lasagne.layers.GRULayer(l_char_rnn, self.char_hidden_size, mask_input=l_c_mask)
            l_gru_2 = lasagne.layers.GRULayer(l_char_rnn, self.char_hidden_size, mask_input=l_c_mask, backwards=True)
        l_gru = lasagne.layers.ReshapeLayer(l_gru, (-1, self.x.shape[1], [1], [2]))
        l_gru = lasagne.layers.SliceLayer(l_gru, -1, axis=2)
        l_gru_2 = lasagne.layers.ReshapeLayer(l_gru_2, (-1, self.x.shape[1], [1], [2]))
        l_gru_2 = lasagne.layers.SliceLayer(l_gru_2, 0, axis=2)
        l = lasagne.layers.ReshapeLayer(l, (-1, [3], [2]))
        ls = []
        l_c_mask = lasagne.layers.DimshuffleLayer(l_c_mask, (0, 'x', 1))
        l_filter = ElemwiseMergeLayer([l, l_c_mask], T.mul)
        for f_size in self.char_f_sizes:
            ls.append(lasagne.layers.Conv1DLayer(l_filter, self.char_filter_num, f_size))
        for i in range(len(ls)):
            ls[i] = lasagne.layers.GlobalPoolLayer(ls[i], T.max)
        l_c = lasagne.layers.ConcatLayer(ls)
        l_c = lasagne.layers.ReshapeLayer(l_c, (-1, self.x.shape[1], [1]))
        l_word = lasagne.layers.InputLayer(shape=(None, self.x.shape[1]), input_var=wx_sym)
        # l_word = lasagne.layers.EmbeddingLayer(l_word, self.word_cnt, self.w_embedding_size, W = lasagne.init.Normal(std = 1e-3))
        l_word = lasagne.layers.EmbeddingLayer(l_word, self.word_cnt, self.w_embedding_size,
                                               W=lasagne.init.GlorotUniform('relu'))
        # l_word = lasagne.layers.EmbeddingLayer(l_word, self.word_cnt, self.w_embedding_size, W = lasagne.init.HeNormal('relu'))

        self.embedding = l_word.W
        if REDUCE:
            l_word = lasagne.layers.ReshapeLayer(l_word, (-1, [2]))
            l_word = lasagne.layers.EmbeddingLayer(l_word, self.word_cnt, self.w_embedding_size,
                                                   W=lasagne.init.Normal(std=1e-3))

            self.trans = l_word.W
            l_word = lasagne.layers.ReshapeLayer(l_word, (-1, self.x.shape[1], [1]))
        layer_list = []
        if self.word_embedding:
            layer_list.append(l_word)
        if self.use_lemma:
            l_lemma = lasagne.layers.InputLayer(shape=(None, self.x.shape[1]), input_var=lemma_sym)
            l_lemma = lasagne.layers.EmbeddingLayer(l_lemma, self.lemma_cnt, self.lemma_size,
                                                    W=lasagne.init.Normal(std=1e-3))
            layer_list.append(l_lemma)
        if self.use_pos:
            print(self.pos_cnt)
            l_pos = lasagne.layers.InputLayer(shape=(None, self.x.shape[1]), input_var=pos_sym)
            l_pos = lasagne.layers.EmbeddingLayer(l_pos, self.pos_cnt, self.pos_size, W=lasagne.init.Normal(std=1e-3))
            layer_list.append(l_pos)
        if self.char_cnn:
            layer_list.append(l_c)
        if self.char_rnn:
            if self.tanh:
                l_grus = lasagne.layers.ConcatLayer([l_gru, l_gru_2], axis=2)
                l_grus = lasagne.layers.ReshapeLayer(l_grus, (-1, [2]))
                l_gru = lasagne.layers.DenseLayer(l_grus, self.tanh_size, nonlinearity=lasagne.nonlinearities.tanh)
                l_gru = lasagne.layers.ReshapeLayer(l_gru, (-1, self.x.shape[1], [1]))
                self.char_layer = l_gru
                layer_list.append(l_gru)
            elif self.joint:
                l_grus = lasagne.layers.ConcatLayer([l_gru, l_gru_2], axis=2)
                self.char_layer = l_grus
                layer_list.append(l_grus)
                char_output = lasagne.layers.get_output(self.char_layer)
                self.char_fn = theano.function([x_sym, cm_sym], char_output, on_unused_input='ignore')
            else:
                layer_list += [l_gru, l_gru_2]

        if len(layer_list) > 1:
            l = lasagne.layers.ConcatLayer(layer_list, axis=2)
        else:
            l = layer_list[0]
        if not self.joint:
            self.char_layer = l

        l_cnn = lasagne.layers.DimshuffleLayer(l, (0, 2, 1))
        l_mask = lasagne.layers.InputLayer(shape=(None, self.x.shape[1]), input_var=m_sym)
        l_cnn_mask = lasagne.layers.DimshuffleLayer(l_mask, (0, 'x', 1))
        l_cnn = ElemwiseMergeLayer([l_cnn, l_cnn_mask], T.mul)
        ls = []

        l_1 = lasagne.layers.GRULayer(l, self.hidden_size, mask_input=l_mask)
        l = lasagne.layers.GRULayer(l, self.hidden_size, mask_input=l_mask, backwards=True)
        if self.dropout:
            l_1 = lasagne.layers.DropoutLayer(l_1, self.dropout_rate)
            l = lasagne.layers.DropoutLayer(l, self.dropout_rate)
        if self.double_layer:
            l = lasagne.layers.ConcatLayer([l_1, l], axis=2)
            l_1 = lasagne.layers.GRULayer(l, self.hidden_size, mask_input=l_mask)
            l = lasagne.layers.GRULayer(l, self.hidden_size, mask_input=l_mask, backwards=True)
            if self.dropout:
                l_1 = lasagne.layers.DropoutLayer(l_1, self.dropout_rate)
                l = lasagne.layers.DropoutLayer(l, self.dropout_rate)
        layer_list = [l_1, l]
        if self.word_cnn:
            layer_list = layer_list + ls
        if self.use_gaze:
            l_gaze = lasagne.layers.InputLayer(shape=(None, self.x.shape[1], self.label_cnt), input_var=gaze_sym)
            layer_list.append(l_gaze)
        l = lasagne.layers.ConcatLayer(layer_list, axis=2)

        if self.top_joint:
            self.char_layer = l

        if self.use_crf:
            l_train = CRFLayer(l, self.label_cnt, mask_input=m_sym, label_input=y_sym)
            l_test = CRFDecodeLayer(l, self.label_cnt, mask_input=m_sym, W=l_train.W, W_sim=l_train.W_sim, \
                                    W_init=l_train.W_init)
            self.l = l_train
            if self.very_top_joint:
                self.char_layer = self.l

            loss = lasagne.layers.get_output(l_train)
            params = lasagne.layers.get_all_params(self.l, trainable=True)
            updates = lasagne.updates.adam(loss, params, learning_rate=self.learning_rate)
            self.train_fn = theano.function([x_sym, y_sym, m_sym, wx_sym, cm_sym, gaze_sym, lemma_sym, pos_sym], loss,
                                            updates=updates, on_unused_input='ignore')

            py = lasagne.layers.get_output(l_test, deterministic=True)
            self.test_fn = theano.function([x_sym, m_sym, wx_sym, cm_sym, gaze_sym, lemma_sym, pos_sym], py,
                                           on_unused_input='ignore')
        else:
            l = lasagne.layers.ReshapeLayer(l, (-1, [2]))
            l = lasagne.layers.DenseLayer(l, self.label_cnt, nonlinearity=lasagne.nonlinearities.softmax)
            self.l = l
            py = lasagne.layers.get_output(l, deterministic=True)
            loss = T.dot(lasagne.objectives.categorical_crossentropy(py, y_sym.flatten()), m_sym.flatten())

            params = lasagne.layers.get_all_params(l, trainable=True)
            updates = lasagne.updates.adam(loss, params, learning_rate=self.learning_rate)
            self.train_fn = theano.function([x_sym, y_sym, m_sym, wx_sym, cm_sym, gaze_sym, lemma_sym, pos_sym], loss,
                                            updates=updates, on_unused_input='ignore')
            self.test_fn = theano.function([x_sym, m_sym, wx_sym, cm_sym, gaze_sym, lemma_sym, pos_sym], py,
                                           on_unused_input='ignore')

        if self.transfer:
            layer_cnt = len(lasagne.layers.get_all_layers(self.l))
            idx = 0
            for layer in lasagne.layers.get_all_layers(self.l):
                idx += 1
                if layer_cnt - idx > 0:
                    logging.info('freeze layer idx: ' + str(idx))
                    for param in layer.params:
                        layer.params[param].discard('trainable')
                else:
                    logging.info('train layer idx: ' + str(idx))

    def store_params(self, iter, filename=None):
        if filename is None:
            fout = open('{}.{}'.format(self.__class__.__name__, iter), 'wb')
        else:
            fout = open(filename, 'wb')
        params = lasagne.layers.get_all_param_values(self.l)
        # cPickle.dump(params, fout, cPickle.HIGHEST_PROTOCOL)
        pickle.dump(params, fout)
        fout.close()

    def load_params(self, filename):
        fin = open(filename, 'rb')
        params = pickle.load(fin)
        lasagne.layers.set_all_param_values(self.l, params)
        fin.close()

    def predict(self, tx, tm, twx, tcm, tgaze, tlemma=None, tpos=None, reload_model_path=None):
        i = 0
        pys = []
        prev_params = None
        while i < self.tx.shape[0]:
            # j = min(self.x.shape[0], i + self.test_batch_size)
            j = i + self.test_batch_size
            s_x, s_m, s_wx, s_cm = tx[i: j], tm[i: j], twx[i: j], tcm[i: j]
            s_gaze = tgaze[i: j] if self.use_gaze else None
            s_lemma = tlemma[i: j] if self.use_lemma else None
            s_pos = tpos[i: j] if self.use_pos else None

            pys.append(self.test_fn(s_x, s_m, s_wx, s_cm, s_gaze, s_lemma, s_pos))
            i = j
        py = np.vstack(tuple(pys))
        if self.use_crf:
            return py.flatten(), py.flatten()
        else:
            return py.argmax(axis=1), py

    def get_char_embedding(self, x, cm):
        return self.char_fn(x, cm)

    def set_embedding(self, word2embedding, word_index):
        t_embedding = self.embedding.get_value()
        emb_hit_cnt = 0
        # for word, embedding in word2embedding.iteritems():
        for word in word2embedding.keys():
            embedding = word2embedding[word]
            if word not in word_index: continue
            emb_hit_cnt += 1
            ind = word_index[word]
            t_embedding[ind, : embedding.shape[0]] = embedding
        print('emb_hit_cnt', emb_hit_cnt)
        self.embedding.set_value(t_embedding)

    def set_w2v(self, word2embedding, word_index):
        t_embedding = self.embedding.get_value()
        emb_hit_cnt = 0
        for word, ind in word_index.iteritems():
            if word not in word2embedding: continue
            emb_hit_cnt += 1
            embedding = word2embedding[word]
            t_embedding[ind, : embedding.shape[0]] = embedding
        print('emb_hit_cnt', emb_hit_cnt)
        self.embedding.set_value(t_embedding)

    def set_embedding_pkl(self, words, embeddings, word_index, lower=True):
        t_embedding = self.embedding.get_value()
        emb_words = set()
        for i in range(len(words)):
            word = words[i].lower() if lower else words[i]
            if word not in word_index: continue
            emb_words.add(word)
            ind = word_index[word]
            embedding = embeddings[i]
            t_embedding[ind, : embedding.shape[0]] = embedding
        print('emb_hit_cnt', len(emb_words))
        self.embedding.set_value(t_embedding)

    def train(self, eval_func):
        print("training...")

        print("\t".join(['epoch', 'iter', 'max_f1', 'f1', 'prec', 'recall']))
        max_f1 = 0.0
        previous_epoch_that_update_dict = 0
        step_idx = 0
        for epoch in range(self.epoch):
            ind = np.random.permutation(self.x.shape[0])
            i = 0
            iter = 0
            while i < self.x.shape[0]:
                iter += 1
                j = min(self.x.shape[0], i + self.batch_size)
                s_x, s_y, s_m, s_wx = self.x[ind[i: j]], self.y[ind[i: j]], self.m[ind[i: j]], self.wx[ind[i: j]]
                if NOISE:
                    noise = np.random.randint(self.char_cnt, size=s_x.shape)
                    noise_mask = np.random.binomial(1, NOISE_RATE, s_x.shape)
                    s_x = np.array(noise * noise_mask + s_x * (1 - noise_mask), dtype=np.int32)
                s_cm = self.cm[ind[i: j]]
                s_gaze = self.gaze[ind[i: j]] if self.use_gaze else None
                s_lemma = self.lemma[ind[i: j]] if self.use_lemma else None
                s_pos = self.pos[ind[i: j]] if self.use_pos else None
                loss = self.train_fn(s_x, s_y, s_m, s_wx, s_cm, s_gaze, s_lemma, s_pos)
                i = j
                period = PERIOD if epoch > self.min_epoch else MIN_PERIOD
                if iter * self.batch_size % period == 0 and epoch > 3:
                    py, py_score = self.predict(self.tx, self.tm, self.twx, self.tcm, self.tgaze, self.tlemma,
                                                self.tpos)
                    if not self.transfer and (epoch - previous_epoch_that_update_dict > 5) or self.transfer and (epoch - previous_epoch_that_update_dict > 5):
                        # if epoch - previous_epoch_that_update_dict > commons.ner_early_stopping_epochs:
                        logging.info('converge at epoch ' + str(previous_epoch_that_update_dict))
                        return
                    if self.ind2word is not None:
                        acc, f1, metric_result = eval_func(py, self.test_data_dic, self.ty, self.tm,
                                                           ind2word=self.ind2word, x=self.twx, apply_rule=False)
                    else:
                        acc, f1, metric_result = eval_func(py, self.test_data_dic, self.ty, self.tm, debug=False,
                                                           apply_rule=False)

                    if f1 > max_f1:
                        model_name = config.ner_model_prefix + self.category + '_' + str(epoch) + '_' + str(iter)
                        current_model_name = self.save_path + model_name
                        previous_epoch_that_update_dict = epoch
                        self.store_params(iter=iter, filename=current_model_name)
                        logging.info('model ' + current_model_name + ' is saved in path ' + self.save_path)
                        self.delete_ner_model_from_disk(model_name)

                    max_f1 = max(max_f1, f1)
                    logging.info(
                        'epoch ' + str(epoch) + ' ' + str(iter) + ' ' + str(max_f1) + ' ' + str(f1) + ' ' + str(
                            metric_result) + '\n')
                step_idx += 1
                # if step_idx % 5 == 0:
                #     print('if transfer, test all cats and update model dict')
            py, py_score = self.predict(self.tx, self.tm, self.twx, self.tcm, self.tgaze, self.tlemma, self.tpos)
            if self.ind2word is not None:
                acc, f1, metric_result = eval_func(py, self.test_data_dic, self.ty, self.tm, ind2word=self.ind2word,
                                                   x=self.twx, apply_rule=False)
            else:
                acc, f1, metric_result = eval_func(py, self.test_data_dic, self.ty, self.tm, debug=False,
                                                   apply_rule=False)

            max_f1 = max(max_f1, f1)
            logging.info(
                str(epoch) + ' ' + str(iter) + ' ' + str(max_f1) + ' ' + str(f1) + ' ' + str(metric_result) + '\n')

    def delete_ner_model_from_disk(self, model_name):
        file_list = os.listdir(self.save_path)
        for f in file_list:
            # if f != model_name and f != 'bkp':
            if self.category in f and f != model_name:
                os.remove(self.save_path + f)
                logging.info(f + ' is removed from path: ' + self.save_path)

    def step_train_init(self):
        self.epoch = 0
        self.ind = np.random.permutation(self.x.shape[0])
        self.i = 0
        self.iter = 0

    def step_train(self):
        i = self.i
        j = min(self.x.shape[0], i + self.batch_size)
        ind = self.ind
        s_x, s_y, s_m, s_wx = self.x[ind[i: j]], self.y[ind[i: j]], self.m[ind[i: j]], self.wx[ind[i: j]]
        s_cm = self.cm[ind[i: j]]
        s_gaze = self.gaze[ind[i: j]] if self.use_gaze else None
        s_lemma = self.lemma[ind[i: j]] if self.use_lemma else None
        s_pos = self.pos[ind[i: j]] if self.use_pos else None
        if self.x.shape[0] > 0:
            loss = self.train_fn(s_x, s_y, s_m, s_wx, s_cm, s_gaze, s_lemma, s_pos)
        self.iter += 1

        if j == self.x.shape[0]:
            self.i = 0
            self.epoch += 1
            self.iter = 0
            self.ind = np.random.permutation(self.x.shape[0])
        else:
            self.i = j

        period = MIN_PERIOD if self.epoch <= self.min_epoch else PERIOD
        ret_flag = self.iter * self.batch_size % period == 0

        if ret_flag:
            return self.predict(self.tx, self.tm, self.twx, self.tcm, self.tgaze, self.tlemma, self.tpos)
        else:
            return None

    def step_predict(self):
        return self.predict(self.tx, self.tm, self.twx, self.tcm, self.tgaze, self.tlemma, self.tpos)

