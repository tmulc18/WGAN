import tensorflow as tf

def dense_layer(x,num_nodes,activation='relu',name_suffix='',bias_init_val=1,is_plain=False,n_prev=500.):
    """
    Dense fully connected layer
    y = activation(xW+b)
    
    Inputs
    ------------------
    x : input tensor
    
    num_nodes : number of nodes in the layer
    
    acitvation (optional: activation function to use 
         one of ['relu','sigmoid',None]
         default is 'relu'
    
    name_suffix (optional) : the suffix to append to variable names
    
    bias_init_val (optional) : the initial value of the bias

    
    Outputs
    ---------------------
    y : the output tensor
    """
    input_shape = x.get_shape()
    initializer_W = tf.random_normal(stddev=.01,shape=[int(input_shape[-1]),num_nodes])
    if is_plain: initializer_W = tf.random_uniform(minval=-(6.**.5)/((n_prev+num_nodes)**.5),\
        maxval=(6.**.5)/((n_prev+num_nodes)**.5),shape=[int(input_shape[-1]),num_nodes])

    W=tf.get_variable('W'+name_suffix,initializer=initializer_W)
    b=tf.get_variable('b'+name_suffix,initializer=tf.constant(bias_init_val,shape = [num_nodes],dtype=tf.float32))
    
    logits = tf.matmul(x,W)+b
    
    if activation == None:
        y = logits
    elif activation == 'relu':
        y = tf.nn.relu(logits)
    elif activation == 'sigmoid':
        y = tf.nn.sigmoid(logits)
    else:
        raise ValueError("Enter a valid activation function")
    
    return y
