import tensorflow as tf
import numpy as np
from modules import encoder_block,match_att
from tensorflow.contrib.layers import l2_regularizer,xavier_initializer,xavier_initializer_conv2d


class QANET(object):
    def __init__(self, config):
        '''

        :param config:
        '''
        self.keep_prob = config.drop_keep_prob
        self.is_train = tf.cast(config.is_train, tf.bool)
        self.n_class = config.n_class
        self.optimizer = config.optimizer
        self.clip_value = config.clip_value
        self.l2_reg = config.l2_reg

        self.word_embeddings = config.word_embedding
        self.word_embed_dim = config.word_embed_size
        self.word_vocab_size = config.word_vocab_size

        self.max_len = config.max_seq_len
        self.hidden_size=config.cnn_hidden

        self.build_graph()

    def build_graph(self):

        self._placeholder_init()
        self._embedding()

        self._encode()
        self._match()
        self._fuse()
        self._decode()

        self.pred()

        self.accu()
        self.loss_op()
        self.train_op()

    def _placeholder_init(self):
        self.s1 = tf.placeholder(tf.int32, [None, self.max_len], name='s1')
        self.s2 = tf.placeholder(tf.int32, [None, self.max_len], name='s2')

        self.s1_mask = tf.placeholder(tf.int32, [None, self.max_len], name='mask_s1')
        self.s2_mask = tf.placeholder(tf.int32, [None, self.max_len], name='mask_s2')

        self.s1_len = tf.placeholder(tf.int32, [None], name='s1_len')
        self.s2_len = tf.placeholder(tf.int32, [None], name='s2_len')

        self.label = tf.placeholder(tf.int32, [None], name='label')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    def _embedding(self):

        with tf.variable_scope('word_embedding'):
            if self.word_embeddings is not None:
                print('use embedding!')
                self.word_embeddings = tf.convert_to_tensor(self.word_embeddings, dtype=tf.float32,
                                                            name='word_embedding')
            else:
                print('random word embedding!')
                self.word_embeddings = tf.get_variable('word_embedding',
                                                       shape=[self.word_vocab_size, self.word_embed_dim],
                                                       dtype=tf.float32)
            print(self.word_embeddings)
            self.s1_embed = tf.nn.embedding_lookup(self.word_embeddings, self.s1)
            self.s2_embed = tf.nn.embedding_lookup(self.word_embeddings, self.s2)

    def _encode(self):
        # input,num_encode_block,num_conv_layers,filter,kernel_size,num_head,fnn_hidden,scope,reuse=None

        self.s1_enc = encoder_block(input=self.s1_embed,
                                   num_encode_block=1,
                                   num_conv_layers=4,
                                   filter=self.hidden_size,
                                   kernel_size=1,
                                   num_head=8,
                                   fnn_hidden=self.hidden_size,
                                   scope='embed_encoder',
                                   reuse=None)
        self.s2_enc = encoder_block(input=self.s2_embed,
                                   num_encode_block=1,
                                   num_conv_layers=4,
                                   filter=self.hidden_size,
                                   kernel_size=1,
                                   num_head=8,
                                   fnn_hidden=self.hidden_size,
                                   scope='embed_encoder',
                                   reuse=True)

    def _match(self,mode='dot',scope='att'):
        batch_size,context_len,num=self.s1_enc.get_shape().as_list()
        _,query_len,_=self.s2_enc.get_shape().as_list()
        print('bs:',batch_size)
        print('context_len',context_len)
        print('query_len',query_len)
        print('embed:',num)
        with tf.variable_scope(scope):
            if mode=='dot':
                att_map=tf.matmul(self.s1_enc,self.s2_enc,transpose_b=True)
            elif mode=='bilinear':

                att_map=tf.matmul(tf.layers.dense(self.s1_enc,num),self.s2_enc,transpose_b=True)
            elif mode=='additive':
                # 计算错误
                Wac = tf.get_variable('add_Wac', shape=[num, num], dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer())
                Waq=tf.get_variable('add_Wac', shape=[num, num], dtype=tf.float32,
                                     initializer=tf.random_uniform_initializer())
                v=tf.get_variable('add_v',shape=[num],dtype=tf.float32,initializer=tf.random_uniform_initializer())
                att_map=tf.matmul(
                    tf.nn.tanh(
                    tf.matmul(self.s1_enc,Wac)+tf.matmul(self.s2_enc,Waq)
                    )
                ,v)
            else:
                raise NotImplementedError

            self.attmap= att_map

    def _fuse(self, encoder_num=3):
        att_map = tf.nn.softmax(self.attmap, axis=-1)
        catt_s1 = tf.matmul(att_map, self.s2_enc)
        s1_conv = self.cnn_encoder(self.s1_enc, self.hidden_size, filter_size=3, reuse=None, scope='s1')
        catt_s1_conv = self.cnn_encoder(catt_s1, self.hidden_size, filter_size=1, reuse=None, scope='att')
        b = tf.get_variable('b', shape=[self.hidden_size], dtype=tf.float32, initializer=tf.constant_initializer())
        model_input = tf.nn.tanh(s1_conv + catt_s1_conv + b)

        with tf.variable_scope('modeling_layer'):
            self.output = []
            model_output = model_input
            reuse = None
            for i in range(encoder_num):
                if i > 0:
                    reuse = True

                # input,num_encode_block,num_conv_layers,filter,kernel_size,num_head,fnn_hidden,scope,reuse=None
                model_output = encoder_block(model_output,
                                             num_encode_block=7,
                                             num_conv_layers=2,
                                             filter=self.hidden_size,
                                             kernel_size=3,
                                             num_head=8,
                                             fnn_hidden=self.hidden_size,
                                             scope='model_encoder',
                                             reuse=reuse)
                self.output.append(model_output)

    def _decode(self):
        with tf.variable_scope('output_layer'):
            output=tf.reduce_mean(self.output[2],axis=1)
            out=tf.layers.dense(output,
                            self.n_class,
                            activation=None,
                            use_bias=None,
                            )
            print( 'out:',out)
            self.output=out

    def cnn_encoder(self,seq,num_out,filter_size=3,act_fn=None,reuse=None,scope=None):

            with tf.variable_scope(scope or 'cnn') as scope:
                cond=tf.contrib.layers.conv1d(
                    seq,
                    num_out,
                    filter_size,
                    padding='SAME',
                    activation_fn=act_fn,
                    reuse=reuse,
                    scope=scope
                )
            return cond

    def pred(self):
        self.pre = self.linear(self.output, self.n_class, 'w_pred', 'b_pred',
                               activation=tf.nn.tanh, regularizar=l2_regularizer(self.l2_reg))

    def accu(self):
        print('logits:', self.pre)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.arg_max(self.pre, dimension=1), dtype=tf.int32),
                                                   tf.cast(self.label, tf.int32)), tf.float32))

    def loss_op(self):
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label, logits=self.pre))
        if self.l2_reg:
            l2loss = tf.add_n([tf.nn.l2_loss(tensor) for tensor in tf.trainable_variables()
                               if tensor.name.endswith("weights:0") or tensor.name.endswith('kernel:0')]) \
                     * tf.constant(self.l2_reg,
                                   dtype='float', shape=[], name='l2_regularization_ratio')
            tf.summary.scalar('l2loss', l2loss)
            self.loss += l2loss

    def train_op(self):
        with tf.name_scope('training'):
            if self.optimizer == 'Adam':
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
            elif self.optimizer == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            elif self.optimizer == 'momentum':
                optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9)
            elif self.optimizer == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
            elif self.optimizer == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            else:
                print('use sgd!')
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

            tvars = tf.trainable_variables()
            for var in tvars:
                print(var.name, var.shape)
            grads = tf.gradients(self.loss, tvars)
            if self.clip_value > 0:
                grads, _ = tf.clip_by_global_norm(grads, self.clip_value)
            self.optim = optimizer.apply_gradients(
                zip(grads, tvars),
            )


    def linear(self,input, outsize, w_name, b_name=None, activation=None, regularizar=None):
        input_size = input.shape[-1]
        w = tf.get_variable(w_name, [input_size, outsize], regularizer=regularizar)
        out = tf.tensordot(input, w,axes=1)
        if b_name is not None:
            b = tf.get_variable(b_name, shape=[outsize])
            out = out + b
        if activation is not None:
            out = activation(out)
        return out



if __name__ == '__main__':
    import config

    config = config.parser_args()
    config.word_vocab_size = 5
    config.char_vocab_size = 6
    config.is_train = True
    model = ABCNN(config)









