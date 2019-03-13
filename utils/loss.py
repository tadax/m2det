import tensorflow as tf

def _classification_loss(cls_outputs, cls_targets, positive_mask, num_positives, num_anchors, batch_size, alpha=0.25, gamma=2.0):
    num_negatives = tf.minimum(3 * num_positives, num_anchors - num_positives) # neg_pos_ratio is 3
    negative_mask = tf.greater(num_negatives, 0)

    cls_outputs = tf.sigmoid(cls_outputs)
    conf_loss = -tf.reduce_sum(cls_targets * tf.log(cls_outputs), axis=-1)
    pos_conf_loss = tf.reduce_sum(conf_loss * positive_mask, axis=1) 
    
    has_min = tf.to_float(tf.reduce_any(negative_mask)) # it would be 0.0 if ALL num_neg are 0
    num_neg = tf.concat(axis=0, values=[num_negatives, [(1 - has_min) * 100]])
    # minimum value under the condition the value > 0
    num_neg_batch = tf.reduce_min(tf.boolean_mask(num_negatives, tf.greater(num_negatives, 0)))
    num_neg_batch = tf.to_int32(num_neg_batch)
    max_confs = tf.reduce_max(cls_outputs[:, :, 1:], axis=2) # except backgound class
    _, indices = tf.nn.top_k(max_confs * (1 - positive_mask), k=num_neg_batch)
    batch_idx = tf.expand_dims(tf.range(0, batch_size), 1)
    batch_idx = tf.tile(batch_idx, (1, num_neg_batch))
    full_indices = (tf.reshape(batch_idx, [-1]) * tf.to_int32(num_anchors) + tf.reshape(indices, [-1]))
    neg_conf_loss = tf.gather(tf.reshape(conf_loss, [-1]), full_indices)
    neg_conf_loss = tf.reshape(neg_conf_loss, [batch_size, num_neg_batch])
    neg_conf_loss = tf.reduce_sum(neg_conf_loss, axis=1)

    classification_loss = pos_conf_loss + neg_conf_loss
    classification_loss /= (num_positives + tf.to_float(num_neg_batch))

    return classification_loss
    
def _box_loss(box_outputs, box_targets, positive_mask, num_positives, delta=0.1):
    normalizer = num_positives * 4
    # to avoid division by 0
    normalizer = tf.where(tf.not_equal(normalizer, 0), normalizer, tf.ones_like(normalizer))

    sq_loss = 0.5 * (box_targets - box_outputs) ** 2
    abs_loss = 0.5 * delta ** 2 + delta * (tf.abs(box_outputs - box_targets) - delta)
    l1_loss = tf.where(tf.less(tf.abs(box_outputs - box_targets), delta), sq_loss, abs_loss)
    box_loss = tf.reduce_sum(l1_loss, axis=-1)
    box_loss = tf.reduce_sum(box_loss * positive_mask, axis=-1)
    box_loss = box_loss / normalizer
    return box_loss

def calc_loss(y_true, y_pred, box_loss_weight=50.0):
    """
    Args:
        y_true: [batch_size, num_anchors, 4 + num_classes + 1]
        y_pred: [batch_size, num_anchors, 4 + num_classes]
            num_classes is including the back-ground class
            last element of y_true denotes if the box is positive or negative:
    Returns:
        total_loss:

    cf. https://github.com/tensorflow/tpu/blob/master/models/official/retinanet/retinanet_model.py
    """
    
    box_outputs = y_pred[:, :, :4]
    box_targets = y_true[:, :, :4]
    cls_outputs = y_pred[:, :, 4:]
    cls_targets = y_true[:, :, 4:-1]
    positive_mask = y_true[:, :, -1]
    batch_size = tf.shape(y_true)[0]
    num_anchors = tf.to_float(tf.shape(y_true)[1])
    num_positives = tf.reduce_sum(positive_mask, axis=-1) # shape: [batch_size,]

    box_loss = _box_loss(box_outputs, box_targets, positive_mask, num_positives)
    classification_loss = _classification_loss(cls_outputs, cls_targets, positive_mask, num_positives, num_anchors, batch_size)
    total_loss = classification_loss + box_loss_weight * box_loss

    return tf.reduce_mean(total_loss)
