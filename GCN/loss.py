import tensorflow as tf

def masked_softmax_cross_entropy(preds, labels, mask):
	loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)

	mask = tf.cast(mask, dtype=tf.float32)
	mask /= tf.reduce_mean(mask)

	loss *= mask

	return tf.reduce_mean(loss)