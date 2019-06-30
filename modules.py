'''
.py contains some modules needed
'''
import tensorflow as tf
import numpy as np

def conv(input,filter,kernel,scope='con1d',reuse=None):
    with tf.variable_scope(scope,reuse=reuse) as scope:

        output=tf.contrib.layers.conv1d(input,
                                        filter,
                                        kernel,
                                        padding='SAME',
                                        reuse=reuse,
                                        scope=scope)

        return output

def layernorm(input,scope,reuse=None):
    with tf.variable_scope(scope or 'norm',reuse=reuse) as scope:
        output=tf.contrib.layers.layer_norm(input,scope=scope,reuse=reuse)

        return output

def conv_block(input,num_conv_layers,filter,kernel_size,scope='conv_block',reuse=None):
    '''
    built conv_block which is made up of conv * # + norm + residual
    '''
    with tf.variable_scope(scope,reuse=reuse) as scope:

        init_input=input
        for i in range(num_conv_layers):

            input_norm=layernorm(input,scope='conv_norm_%d' % i,reuse=reuse)
            output=conv(input_norm,filter,kernel_size,scope='convd_%d' % i,reuse=reuse)
            #残差连接
            if init_input.get_shape()[-1]==output.get_shape()[-1]:
                output=init_input+output
            input=output
        return input

def multi_head_attention(input,num_head,scope='multi_head_attention',reuse=None):
    with tf.variable_scope(scope,reuse=reuse):
        inputs=layernorm(input,scope='att_norm',reuse=reuse)
        q=inputs
        k=inputs
        v=inputs
        units=inputs.get_shape().as_list()[-1]
        dim=units/num_head
        q_=tf.split(q,num_head,axis=2)
        k_=tf.split(k,num_head,axis=2)
        v_=tf.split(v,num_head,axis=2)
        q_=[tf.layers.dense(qi,dim,activation=tf.nn.relu) for qi in q_]
        k_=[tf.layers.dense(ki,dim,activation=tf.nn.relu) for ki in k_]
        v_=[tf.layers.dense(vi,dim,activation=tf.nn.relu) for vi in v_]

        #[num_head,bs,seq,dim] --->[num_head*bs,seq,dim]
        Q=tf.concat(q_,axis=0)
        K=tf.concat(k_,axis=0)
        V=tf.concat(v_,axis=0)

        q_k=tf.matmul(Q,tf.transpose(K,[0,2,1]))/ (dim ** 0.5)
        att=tf.matmul(tf.nn.softmax(q_k),V)

        attention=tf.concat(tf.split(att,num_head,axis=0),axis=2)

        output=attention+input
        return output

def fnn(input,out_dim,scope='encoder_nn',reuse=None):
    with tf.variable_scope(scope,reuse=reuse):
        inputs=layernorm(input,scope='block_nn_norm',reuse=reuse)
        nnout=tf.layers.dense(inputs,out_dim,activation=tf.nn.relu)
        output=nnout+input
        return output

def encoder_block(input,num_encode_block,num_conv_layers,filter,kernel_size,num_head,fnn_hidden,scope='block',reuse=None):
    '''

    one block consists of conv_block*#+multihead+fnn
    ,which paras are not shared
    :return:
    '''
    with tf.variable_scope(scope ,reuse=reuse) as scope:

        output=input
        for i in range(num_encode_block):
            conv=conv_block(output,num_conv_layers,filter,kernel_size,scope='conv_block_%d' % i,reuse=reuse)
            att=multi_head_attention(conv,num_head,scope='multi_head_att%d' % i,reuse=reuse)
            output=fnn(att,fnn_hidden,scope='encoder_nn%d' % i,reuse=reuse)
        return output

def match_att(p,q,max_p,max_q,scope='attention'):
    with tf.variable_scope(scope):
        p_=tf.tile(tf.expand_dims(p,axis=2),[1,1,max_q,1])
        q_=tf.tile(tf.expand_dims(q,axis=1),[1,max_p,1,1])
        pq_concat=tf.concat([p_,q_,p_*q_],axis=-1)
        att=tf.layers.dense(pq_concat,1,activation=None,use_bias=False)
        print('match_att:',att)
        return tf.reshape(att,[-1,max_p,max_q])


if __name__=='__main__':
    a=tf.convert_to_tensor(np.random.uniform(-1,1,[3,10,500]))
    encoder_block(a,1,4,128,7,8,128,scope='block')
