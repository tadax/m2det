import tensorflow as tf

def _classification_loss(cls_outputs, cls_targets, num_positives, alpha=0.25, gamma=2.0):
    pass
    
def _box_loss(box_outputs, box_targets, num_positives, delta=0.1):
    normalizer = tf.where(tf.not_equal(num_positives, 0), num_positives * 4, tf.ones_like(num_positives)) # avoid division by 0
    sq_loss = 0.5 * (box_targets - box_outputs) ** 2
    abs_loss = 0.5 * delta ** 2 + delta * (tf.abs(box_outputs - box_targets) - delta)
    l1_loss = tf.where(tf.less(tf.abs(box_outputs - box_targets), delta), sq_loss, abs_loss)
    loc_loss = tf.reduce_sum(l1_loss, axis=-1)
    loc_loss = loc_loss / tf.expand_dims(normalizer, 1)
    return loc_loss

def calc_loss(y_true, y_pred, box_loss_weight=50.0):
    """
    Args:
        y_true: [batch_size, num_anchors, 4 + num_classes + 1]
        y_pred: [batch_size, num_anchors, 4 + num_classes]
            num_classes is including the back-ground class
            last element of y_true denotes if the box is positive or negative:
    Returns:
        total_loss:
    """
    
    num_positives = tf.reduce_sum(y_true[:, :, -1], axis=-1) # shape: [batch_size,]

    box_outputs = y_pred[:, :, :4]
    box_targets = y_true[:, :, :4]
    mask = y_true[:, :, -1]
    box_loss = _box_loss(box_outputs, box_targets, num_positives)
    box_loss = tf.reduce_sum(box_loss * mask, axis=1)

    y_pred_cls = tf.maximum(tf.minimum(y_pred[:, :, 4:], 1 - 1e-15), 1e-15)
    conf_loss = -tf.reduce_sum(y_true[:, :, 4:-1] * tf.log(y_pred_cls), axis=-1)

    batch_size = tf.shape(y_true)[0]
    num_boxes = tf.to_float(tf.shape(y_true)[1])

    pos_conf_loss = tf.reduce_sum(conf_loss * y_true[:, :, -1], axis=1) 
    
    num_negatives = tf.minimum(3 * num_positives, num_boxes - num_positives) # neg_pos_ratio is 3

    pos_num_neg_mask = tf.greater(num_negatives, 0)
    has_min = tf.to_float(tf.reduce_any(pos_num_neg_mask)) # it would be 0.0 if ALL num_neg are 0
    num_neg = tf.concat(axis=0, values=[num_negatives, [(1 - has_min) * 100]])
    # minimum value under the condition the value > 0
    num_neg_batch = tf.reduce_min(tf.boolean_mask(num_negatives, tf.greater(num_negatives, 0)))
    num_neg_batch = tf.to_int32(num_neg_batch)
    max_confs = tf.reduce_max(y_pred[:, :, 5:], axis=2) # except backgound class
    _, indices = tf.nn.top_k(max_confs * (1 - y_true[:, :, -1]), k=num_neg_batch)
    batch_idx = tf.expand_dims(tf.range(0, batch_size), 1)
    batch_idx = tf.tile(batch_idx, (1, num_neg_batch))
    full_indices = (tf.reshape(batch_idx, [-1]) * tf.to_int32(num_boxes) + tf.reshape(indices, [-1]))
    neg_conf_loss = tf.gather(tf.reshape(conf_loss, [-1]), full_indices)
    neg_conf_loss = tf.reshape(neg_conf_loss, [batch_size, num_neg_batch])
    neg_conf_loss = tf.reduce_sum(neg_conf_loss, axis=1)

    classification_loss = pos_conf_loss + neg_conf_loss
    classification_loss /= (num_positives + tf.to_float(num_neg_batch))

    total_loss = classification_loss + box_loss_weight * box_loss

    return tf.reduce_mean(total_loss)
