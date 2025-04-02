import tensorflow as tf

def optimizer(learner, loss, learning_rate, momentum=0.9):
    optimizer = None
    if learner.lower() == "adagrad":
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate, initial_accumulator_value=1e-8)
    elif learner.lower() == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate)
    elif learner.lower() == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate)
    elif learner.lower() == "gd":
        optimizer = tf.keras.optimizers.SGD(learning_rate)
    elif learner.lower() == "momentum":
        optimizer = tf.keras.optimizers.SGD(learning_rate, momentum=momentum)
    else:
        raise ValueError("Please select a suitable optimizer")

    return optimizer

def pairwise_loss(loss_function, y, margin=1):
    loss = None
    if loss_function.lower() == "bpr":
        loss = -tf.reduce_sum(tf.math.log_sigmoid(y))
    elif loss_function.lower() == "hinge":
        loss = tf.reduce_sum(tf.maximum(y + margin, 0))
    elif loss_function.lower() == "square":
        loss = tf.reduce_sum(tf.square(1 - y))
    else:
        raise Exception("Please choose a suitable loss function")
    return loss

def pointwise_loss(loss_function, y_rea, y_pre):
    loss = None
    if loss_function.lower() == "cross_entropy":
        loss = tf.losses.sigmoid_cross_entropy(y_rea, y_pre)
    elif loss_function.lower() == "square":
        loss = tf.reduce_sum(tf.square(y_rea - y_pre))
    else:
        raise Exception("Please choose a suitable loss function")
    return loss
