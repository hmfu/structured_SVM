from utils import Data
import tensorflow as tf
import numpy as np
import os
import re

class Data_formater(object):

    def __init__(self):
        self.lyric_array = None
        self.lyric_topic_tag_array = None
        self.embedding_table = None
        self.lyric_num = None
        self.vocab_num = None

    def load_lyrics(self, data_directory, lyric_fixed_length):

        self.lyric_tuple_list = []
        vocab_set = set()

        for topic_name in [name for name in os.listdir(data_directory) if not name.startswith('.')]:
            topic_directory = os.path.join(data_directory, topic_name)

            for lyric_file_name in [file_name for file_name in os.listdir(topic_directory) if not file_name.startswith('.')]:
                tag = lyric_file_name.split('_')[0]
                file_directory = os.path.join(topic_directory, lyric_file_name)

                with open(file_directory, 'r') as input_file:
                    word_list = re.sub('[^a-zA-Z\']+', ' ', input_file.read()).replace("'", '').lower().split()

                    self.lyric_tuple_list += [(topic_name, tag, word_list)]
                    vocab_set = vocab_set.union(word_list)

        # Build vocab to index dictionary and its reverse.
        self.lyric_num = len(self.lyric_tuple_list)
        self.vocab_idx_dictionary = {}
        next_vocab_idx = 0

        for vocab in vocab_set:
            if vocab in self.embedding_dictionary:
                self.vocab_idx_dictionary[vocab] = next_vocab_idx
                next_vocab_idx += 1

        self.idx_vocab_dictionary = {v: k for k, v in self.vocab_idx_dictionary.items()}
        self.vocab_num = next_vocab_idx

        # Build embedding table to be used in the model.
        self.embedding_table = []
        
        for idx in range(self.vocab_num):
            self.embedding_table += [self.embedding_dictionary[self.idx_vocab_dictionary[idx]]]

        self.embedding_table = np.array(self.embedding_table)
        
        # Convert words in the lyric tuples to index and fix their lengths. 
        # Index vocab_num is for OOV and vocab_num + 1 for padding.
        for lyric_tuple_idx in range(self.lyric_num):
            lyric_tuple = self.lyric_tuple_list[lytic_tuple_idx]
            
            word_idx_list = [self.vocab_idx_dictionary.get(word, self.vocab_num) for word in lyric_tuple[2]]
            word_idx_list = word_idx_list[: lyric_fixed_length] if len(word_idx_list) > lyric_fixed_length \
                    else word_idx_list + [self.vocab_num + 1] * (lyric_fixed_length - len(word_idx_list))
            
            self.lyric_tuple_list[lyric_tuple_idx] = (lyric_tuple[0], lyric_tuple[1], word_idx_list)
        
        # Build lyric array to be used as input in the model.
        self.lyric_array = np.array([lyric_tuple[2] for lyric_tuple in self.lyric_tuple_list])
        self.lyric_topic_tag_array = np.array([[lyric_tuple[0], lyric_tuple[1]] for lyric_tuple in self.lyric_tuple_list])

    def load_embedding_dictionary(self, file_name):
        
        self.embedding_dictionary = {}
        self.dictionary_key_set = set(list(self.embedding_dictionary.keys()))

    def load_images(self, data_directory, slice_idx_list):

        self.image_tuple_list = []

        for slice_idx in slice_idx_list:
            slice_directory = os.path.join(data_directory, str(slice_idx))

            for topic_name in [name for name in os.listdir(slice_directory) if not name.startswith('.')]:
                topic_directory = os.path.join(slice_directory, topic_name)

                for tag_name in [name for name in os.listdir(topic_directory) if not name.startswith('.')]:
                    tag_directory = os.path.join(topic_directory, tag_name)

                    for img_file_name in [file_name for file_name in os.listdir(tag_directory)if not file_name.startswith('.')]:
                        image_vector = np.load(os.path.join(tag_directory, img_file_name))
                        
                        self.image_tuple_list += [(topic_name, tag_name, image_vector)]

class Structured_SVM(object):

    def __init__(self, weight_stddev, bias_stddev):
        self.weight_stddev = weight_stddev
        self.bias_stddev = bias_stddev

        self.loss = None

    def build_model(self, image_dimension, lyric_length, lyric_num, vocab_num, fully_connected_layer_num, embedding_dimension, C, embedding_table):
        
        with tf.name_scope('input_placeholders'):
            
            self.image_input = tf.placeholder(dtype = tf.float32, shape = [None, image_dimension])
            self.lyric_input = tf.placeholder(dtype = tf.float32, shape = [None, lyric_num, lyric_length])
            self.lyric_gap = tf.placeholder(dtype = tf.float32, shape = [None]) # 0.0 for answers, 0.5 for same topic, 1.0 for negative samples.
            self.C = tf.constant(dtype = tf.float32, shape = [])
            self.embedding_table = tf.constant(embedding_table)

        with tf.name_scope('lyric_cnn'):

            embedded_lyric = tf.nn.embedding_lookup(self.embedding_table, self.lyric_input)
            
            



        with tf.name_scope('get_scores'):

            image_expanded = tf.tile(tf.expand_dims(self.image_input, axis = 1), multiples = [1, 1, lyric_num])
            image_lyric_pair = tf.concat([image_expanded, self.lyric_input], axis = 2)

            node_num_list = self.get_node_num_list(image_dimension+lyric_length, embedding_dimension, fully_connected_layer_num)
            
            for layer_idx in range(fully_connected_layer_num):
                
                node_num = node_num_list[layer_idx]
                image_lyric_pair = self.fully_connected_layer(image_lyric_pair, node_num)
                
                if layer_idx != fully_connected_layer_num - 1:
                    image_lyric_pair = tf.nn.relu(image_lyric_pair)

            self.w_vector = tf.Variable(tf.random_normal(shape = [embedding_dimension], stddev = self.weight_stddev))
            
            self.pair_score = tf.reduce_sum(image_lyric_pair * w, axis = 2)

        with tf.name_scope('get_loss'):
            
            w_regularization_loss = tf.norm(self.w_vector)
            
            answer_score_expanded = tf.tile(tf.slice(self.pair_score, [0, 0], [-1, 1]), multiples = [1, lyric_num])
            gap_loss = tf.C * tf.maximum(0, self.lytic_gap + answer_score_expanded - self.pair_score) # Not sure about the syntex.

            self.loss = gap_loss + w_regularization_loss

    def train_model(self, learning_rate, epoch_num, batch_size):
        optimizer = tf.train.AdamOptimizer(learning_rage = learning_rate).minimize(self.loss)

        with tf.Session(config = config) as sess:
            sess.run(global_variables_initializer())

            for epoch_idx in range(epoch_num):
                

                for batch_idx in range(batch_num):
                    pass

    def time_distributed_conv_layer(self, input_placeholder, filter_size, filter_num):
        input_placeholder = tf.time.keras.layers.TimeDistributed(tf.nn.conv2d())


    def shuffle_dictionary(self, dictionary):
        sample_num = len(dictionary[list(dictionary.keys())[0]])
        order = np.random.permutation(sample_num)
        return {k: v[order] for k, v in dictionary.get_items()}

    def get_batch(self, feed_dict, batch_size, start_idx):
        return {k: v[start_idx: start_idx + batch_size] for k, v in feed_dict.get_items()}

    def get_node_num_list(self, input_dimension, output_dimension, layer_num):
        gap = (output_dimension - input_dimension) / layer_num
        return [input_dimension - gap * layer_idx for layer_idx in range(layer_num)]

    def fully_connected_layer(self, input_placeholder, output_dimension): 
        output_dimension = tf.cast(input_placeholder.shape[1], tf.int32)

        weight = tf.Variable(tf.random_normal(shape = [input_dimension, output_dimension], stddev = self.wieght_stddev))
        bias = tf.Variable(tf.random_normal(shape = [output_dimension], stddev = self.bias_stddev))

        return tf.matmul(input_placeholder, weight) + bias

if __name__ == '__main__':
    data_formater = Data_formater()
    data_formater.load_lyrics('lyric_data')
    data_formater.load_images('cross_va_data', list(range(10)))
