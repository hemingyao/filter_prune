a = tf.constant([[[1,2],[3,4]],[[2,3],[4,5]]])
b = tf.Variable([1,2,3,4])

c = tf.tile(b,[4])
d = tf.reshape(b,[2,2])
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

b = sess.run(d)
print(b)