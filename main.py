import numpy as np
import tensorflow as tf
from ..config import config

__all__ = ['Predictor']

hidden_size = 64

x_size = 48
y_size = 48
num_labels = 10
flatten_to = 3 * 3 * 256

class Predictor(object):
    emo = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt", "unknown", "NF"]

    def __init__(self, model_path=config.data_folder+'predictor/model.ckpt'):
        tf.reset_default_graph()

        self.image_ph = tf.placeholder(tf.float32, [None, x_size, y_size, 1])
        self.label_ph = tf.placeholder(tf.float32, [None, num_labels])
        
        self.W1 = tf.Variable(tf.truncated_normal([3, 3, 1, 32], stddev=0.01))
        self.b1 = tf.Variable(tf.constant(0.01, shape=[32]))

        self.W2 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.01))
        self.b2 = tf.Variable(tf.constant(0.01, shape=[64]))

        self.W3 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.01))
        self.b3 = tf.Variable(tf.constant(0.01, shape=[128]))

        self.W4 = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.01))
        self.b4 = tf.Variable(tf.constant(0.01, shape=[256]))

        self.W5 = tf.Variable(tf.truncated_normal([flatten_to, hidden_size], stddev=0.01))
        self.b5 = tf.Variable(tf.constant(0.01, shape=[hidden_size]))

        self.W6 = tf.Variable(tf.truncated_normal([hidden_size, hidden_size], stddev=0.01))
        self.b6 = tf.Variable(tf.constant(0.01, shape=[hidden_size]))

        self.W7 = tf.Variable(tf.truncated_normal([hidden_size, num_labels], stddev=0.01))
        self.b7 = tf.Variable(tf.constant(0.01, shape=[num_labels]))

        self.conv_layer1 = tf.nn.relu(tf.nn.conv2d(self.image_ph, self.W1, padding='SAME', strides=[1, 1, 1, 1]) + self.b1)
        self.maxpool_layer1 = tf.nn.max_pool(self.conv_layer1, [1, 2, 2, 1], padding='SAME', strides=[1, 2, 2, 1])
        
        self.conv_layer2 = tf.nn.relu(tf.nn.conv2d(self.maxpool_layer1, self.W2, padding='SAME', strides=[1, 1, 1, 1]) + self.b2)
        self.maxpool_layer2 = tf.nn.max_pool(self.conv_layer2, [1, 2, 2, 1], padding='SAME', strides=[1, 2, 2, 1])
        
        self.conv_layer3 = tf.nn.relu(tf.nn.conv2d(self.maxpool_layer2, self.W3, padding='SAME', strides=[1, 1, 1, 1]) + self.b3)
        self.maxpool_layer3 = tf.nn.max_pool(self.conv_layer3, [1, 2, 2, 1], padding='SAME', strides=[1, 2, 2, 1])
       
        self.conv_layer4 = tf.nn.relu(tf.nn.conv2d(self.maxpool_layer3, self.W4, padding='SAME', strides=[1, 1, 1, 1]) + self.b4)
        self.maxpool_layer4 = tf.nn.max_pool(self.conv_layer4, [1, 2, 2, 1], padding='SAME', strides=[1, 2, 2, 1])
      
        self.flattened = tf.reshape(self.maxpool_layer4, [-1, flatten_to])
        
        self.flattened_drop = tf.nn.dropout(self.flattened, 0.5)
        
        self.hidden_layer1 = tf.nn.relu(tf.matmul(self.flattened_drop, self.W5) + self.b5)

        self.hidden_layer2 = tf.nn.relu(tf.matmul(self.hidden_layer1, self.W6) + self.b6)

        self.output_layer = tf.matmul(self.hidden_layer2, self.W7) + self.b7

        self.predictor = tf.nn.softmax(self.output_layer)

        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()

        self.sess.run(self.init)
        self.loader = tf.train.Saver()
        self.loader.restore(self.sess, model_path)

    def predict(self, img):
        if img.shape[0] != 48 or img.shape[1] != 48:
            raise Exception('invalid input image size')

        self.a = np.reshape(img, [48 * 48])
        self.a = [int(i) for i in self.a]
        self.X = np.array(self.a)
        self.D = self.X.shape
        self.d = int(np.sqrt(self.D))
        self.X = self.X.reshape(1, 1, self.d, self.d)
        self.X = self.X.transpose((0, 2, 3, 1))
        self.X = self.X.astype(np.float32)
        return self.sess.run(self.predictor, feed_dict={self.image_ph: self.X})


predictor = Predictor()


if __name__ == '__main__':
    from PIL import Image

    img = Image.open('../dataset/fer+/data/FER2013Test/fer0032237.png')
    gray = np.array(img.convert('L'))
    predictor = Predictor()
    result = predictor.predict(gray)
    print(result)