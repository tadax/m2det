import tensorflow as tf

def calc_loss(y_true, y_pred):
    '''
    input shape:
    y_true: (?, num_boxes, 4 + num_classes + 1)
    y_pred: (?, num_boxes, 4 + num_classes)
    num_classes is including the back-ground class
    last element of y_true denotes if the box is positive or negative
    '''

    abs_loss = tf.abs(y_true[:, :, :4] - y_pred[:, :, :4])
    sq_loss = 0.5 * ((y_true[:, :, :4] - y_pred[:, :, :4]) ** 2)
    l1_loss = tf.where(tf.less(abs_loss, 1.0), sq_loss, abs_loss - 0.5)
    loc_loss = tf.reduce_sum(l1_loss, axis=-1)

    y_pred_cls = tf.maximum(tf.minimum(y_pred[:, :, 4:], 1 - 1e-15), 1e-15)
    conf_loss = -tf.reduce_sum(y_true[:, :, 4:-1] * tf.log(y_pred_cls), axis=-1)

    batch_size = tf.shape(y_true)[0]
    num_boxes = tf.to_float(tf.shape(y_true)[1])

    pos_loc_loss = tf.reduce_sum(loc_loss * y_true[:, :, -1], axis=1)
    pos_conf_loss = tf.reduce_sum(conf_loss * y_true[:, :, -1], axis=1) 
    
    num_pos = tf.reduce_sum(y_true[:, :, -1], axis=-1) # shape: (?,)
    num_neg = tf.minimum(3 * num_pos, num_boxes - num_pos) # neg_pos_ratio is 3

    pos_num_neg_mask = tf.greater(num_neg, 0)
    has_min = tf.to_float(tf.reduce_any(pos_num_neg_mask)) # it would be 0.0 if ALL num_neg are 0
    num_neg = tf.concat(axis=0, values=[num_neg, [(1 - has_min) * 100]])
    # minimum value under the condition the value > 0
    num_neg_batch = tf.reduce_min(tf.boolean_mask(num_neg, tf.greater(num_neg, 0)))
    num_neg_batch = tf.to_int32(num_neg_batch)
    max_confs = tf.reduce_max(y_pred[:, :, 5:], axis=2) # except backgound class
    _, indices = tf.nn.top_k(max_confs * (1 - y_true[:, :, -1]), k=num_neg_batch)
    batch_idx = tf.expand_dims(tf.range(0, batch_size), 1)
    batch_idx = tf.tile(batch_idx, (1, num_neg_batch))
    full_indices = (tf.reshape(batch_idx, [-1]) * tf.to_int32(num_boxes) + tf.reshape(indices, [-1]))
    neg_conf_loss = tf.gather(tf.reshape(conf_loss, [-1]), full_indices)
    neg_conf_loss = tf.reshape(neg_conf_loss, [batch_size, num_neg_batch])
    neg_conf_loss = tf.reduce_sum(neg_conf_loss, axis=1)

    total_loss = pos_conf_loss + neg_conf_loss
    total_loss /= (num_pos + tf.to_float(num_neg_batch))
    num_pos = tf.where(tf.not_equal(num_pos, 0), num_pos, tf.ones_like(num_pos)) # avoid division by 0
    total_loss += pos_loc_loss / num_pos
    return tf.reduce_mean(total_loss)
