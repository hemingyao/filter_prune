import tensorflow as tf 

def random_crop(image, crop_padding_height, crop_padding_width):
	img_size = image.get_shape().as_list()
	if crop_padding_height>img_size[0]:
		image = tf.image.resize_image_with_crop_or_pad(image, target_height=40, target_width=40)
		image = tf.random_crop(image, img_size)
	else:
		image = tf.random_crop(image, [crop_padding_height, crop_padding_width, img_size[-1]])
	return image


def random_flip_left_right(image):
	return tf.image.random_flip_left_right(image)
