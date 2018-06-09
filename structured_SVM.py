from utils import Data
import tensorflow as tf
import numpy as np
import random
import pickle
import os
import re

class Data_formater(object):

    def __init__(self):
        self.lyric_array = None
        self.lyric_topic_tag_array = None
        self.embedding_table = None
        self.lyric_num = None
        self.vocab_num = None

    def build_model_inputs(self):
        image_input = []
        lyric_gap_input = []
        answer_lyric_idx_input = []
        
        for image_tuple in self.image_tuple_list:
            for lyric_tuple_idx in range(len(self.lyric_topic_tag_array)):
                lyric_tuple = tuple(self.lyric_topic_tag_array[lyric_tuple_idx])
                
                if image_tuple[:2] == lyric_tuple[:2]:
                    image_input += [image_tuple[2]]
                    lyric_gap_input += [[self.get_lyric_gap(image_tuple[:2], tuple(arr)) for arr in self.lyric_topic_tag_array]]
                    answer_lyric_idx_input += [[lyric_tuple_idx]]

        self.image_input = np.array(image_input)
        self.lyric_gap_input = np.array(lyric_gap_input)
        self.answer_lyric_idx_input = np.array(answer_lyric_idx_input)

    def get_lyric_gap(self, tuple1, tuple2):
        if tuple1 == tuple2:
            return 0.0
        elif tuple1[0] == tuple2[0]:
            return 0.5
        else:
            return 1.0

    def load_lyrics(self, data_directory, lyric_fixed_length):

        self.lyric_length = lyric_fixed_length
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

        self.embedding_table += [[self.embedding_table[random.randint(0, self.vocab_num - 1)][d] for d in range(self.embedding_dimension)]]
        self.embedding_table += [[0] * self.embedding_dimension]
        self.embedding_table = np.array(self.embedding_table, dtype = np.float32)
        
        # Convert words in the lyric tuples to index and fix their lengths. 
        # Index vocab_num is for OOV and vocab_num + 1 for padding.
        for lyric_tuple_idx in range(self.lyric_num):
            lyric_tuple = self.lyric_tuple_list[lyric_tuple_idx]
            
            word_idx_list = [self.vocab_idx_dictionary.get(word, self.vocab_num) for word in lyric_tuple[2]]
            word_idx_list = word_idx_list[: lyric_fixed_length] if len(word_idx_list) > lyric_fixed_length \
                    else word_idx_list + [self.vocab_num + 1] * (lyric_fixed_length - len(word_idx_list))
            
            self.lyric_tuple_list[lyric_tuple_idx] = (lyric_tuple[0], lyric_tuple[1], word_idx_list)
        
        # Build lyric array to be used as input in the model.
        self.lyric_array = np.array([lyric_tuple[2] for lyric_tuple in self.lyric_tuple_list])
        self.lyric_topic_tag_array = np.array([[lyric_tuple[0], lyric_tuple[1]] for lyric_tuple in self.lyric_tuple_list])

    def load_embedding_dictionary(self, vectors_file_name, word2id_file_name):
        
        with open(vectors_file_name, 'rb') as vectors_file:
            vectors_array = pickle.load(vectors_file)
        
        with open(word2id_file_name, 'rb') as word2id_file:
            word2id_dictionary = pickle.load(word2id_file)
        
        print (vectors_array[:5], word2id_dictionary)
        self.embedding_dimension = len(vectors_array[0])

        word_candidates = list(word2id_dictionary.keys())
        self.embedding_dictionary = {word: list(vectors_array[word2id_dictionary[word]]) for word in word_candidates}

        #self.embedding_dimension = 3
        #self.embedding_dictionary = {'i': [3.2, 4.5, 6.7], 'you': [5.4, 6.3, 9.6], 'love': [2.1, 7.8, 4.5], 'the': [3.2, 5.6, 7.1]}

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

        self.image_dimension = len(self.image_tuple_list[0][2])

class Structured_SVM(object):

    def __init__(self, weight_stddev, bias_stddev):
        self.weight_stddev = weight_stddev
        self.bias_stddev = bias_stddev

    def build_model(self, image_dimension, lyric_length, lyric_num, fully_connected_layer_num, embedding_dimension, C, embedding_table, pool_size, conv_filter_num_list, filter_size):
        
        with tf.name_scope('input_placeholders'):
            
            self.image_input = tf.placeholder(dtype = tf.float32, shape = [None, image_dimension])
            self.lyric_input = tf.placeholder(dtype = tf.int32, shape = [lyric_num, lyric_length])
            self.lyric_gap = tf.placeholder(dtype = tf.float32, shape = [None, lyric_num]) # 0.0 for answers, 0.5 for same topic, 1.0 for negative samples.
            self.C = tf.constant(C, dtype = tf.float32)
            self.embedding_table = tf.constant(embedding_table)
            self.answer_lyric_idx = tf.placeholder(dtype = tf.int32, shape = [None, 1])

        with tf.name_scope('lyric_conv'):

            embedded_lyric = tf.nn.embedding_lookup(self.embedding_table, self.lyric_input)
            sudo_batched_embedded_lyric = tf.expand_dims(embedded_lyric, axis = 0)

            for conv_idx in range(len(conv_filter_num_list)):
                filter_num = conv_filter_num_list[conv_idx]
                sudo_batched_embedded_lyric = self.conv_layer(sudo_batched_embedded_lyric, filter_size, filter_num, pool_size)
                sudo_batched_embedded_lyric = tf.nn.relu(sudo_batched_embedded_lyric)

            lyric_feature = tf.contrib.layers.flatten(tf.squeeze(sudo_batched_embedded_lyric, squeeze_dims = 0))
            sudo_batched_lyric_feature = tf.tile(tf.expand_dims(lyric_feature, axis = 0), multiples = [-1, 1, 1])

        with tf.name_scope('get_scores'):

            image_expanded = tf.tile(tf.expand_dims(self.image_input, axis = 1), multiples = [1, lyric_num, 1])
            image_lyric_pair = tf.concat([image_expanded, sudo_batched_lyric_feature], axis = 2)

            node_num_list = self.get_node_num_list(image_dimension+lyric_length, embedding_dimension, fully_connected_layer_num)
            print ('node_num_list, embedding_dimension: ', node_num_list, embedding_dimension)
            for layer_idx in range(fully_connected_layer_num):
                
                node_num = node_num_list[layer_idx]
                image_lyric_pair = self.fully_connected_layer(image_lyric_pair, node_num)
                
                if layer_idx != fully_connected_layer_num - 1:
                    image_lyric_pair = tf.nn.relu(image_lyric_pair)

            self.w_vector = tf.Variable(tf.random_normal(shape = [embedding_dimension], stddev = self.weight_stddev))
            
            self.pair_score = tf.reduce_sum(image_lyric_pair * self.w_vector, axis = 2)

        with tf.name_scope('get_loss'):
          
            image_num = tf.shape(self.pair_score)[0]
            gather_idx = tf.concat([tf.expand_dims(tf.range(image_num), axis = 1), self.answer_lyric_idx], axis = 1)
            answer_score = tf.gather_nd(self.pair_score, gather_idx)
            answer_score_expanded = tf.tile(tf.expand_dims(answer_score, axis = 1), multiples = [1, lyric_num])
            
            gap_loss = self.C * tf.maximum(0.0, self.lyric_gap + answer_score_expanded - self.pair_score)
            w_regularization_loss = tf.norm(self.w_vector)

            self.loss = gap_loss + w_regularization_loss

    def train_model(self, learning_rate, epoch_num, batch_size, image_input, lyric_gap_input, answer_lyric_idx_input, lyric_array):
        optimizer = tf.train.AdamOptimizer(learning_rage = learning_rate).minimize(self.loss)
        
        complete_feed_dict = {self.image_input: image_input, self.lyric_gap: lyric_gap_input, self.answer_lyric_idx: answer_lyric_idx_input}

        sample_num = len(image_input)
        batch_num = sample_num // batch_size

        with tf.Session(config = config) as self.sess:
            self.sess.run(global_variables_initializer())

            for epoch_idx in range(epoch_num):
                complete_feed_dict = self.shuffle_dictionary(complete_feed_dict)

                for batch_idx in range(batch_num):
                    batch_feed_dict = self.get_batch(complete_feed_dict, batch_size, batch_idx * batch_size)
                    batch_feed_dict[self.lyric_input] = lyric_array
                    
                    history = self.sess.run(optimizer)

    def conv_layer(self, input_placeholder, filter_size, filter_num, pool_size):
        conv_filter = tf.Variable(tf.random_normal(shape = [1, filter_size, tf.cast(input_placeholder.shape[3], tf.int32), filter_num], stddev = self.weight_stddev, dtype = tf.float32))
        input_placeholder = tf.nn.conv2d(input_placeholder, conv_filter, [1, 1, 1, -1], padding = 'SAME')
        output_placeholder = tf.nn.max_pool(input_placeholder, [1, 1, pool_size, 1], [1, 1, 1, 1], padding = 'SAME')
        return output_placeholder

    def shuffle_dictionary(self, dictionary):
        sample_num = len(dictionary[list(dictionary.keys())[0]])
        order = np.random.permutation(sample_num)
        return {k: v[order] for k, v in dictionary.get_items()}

    def get_batch(self, feed_dict, batch_size, start_idx):
        return {k: v[start_idx: start_idx + batch_size] for k, v in feed_dict.get_items()}

    def get_node_num_list(self, input_dimension, output_dimension, layer_num):
        gap = (output_dimension - input_dimension) / layer_num
        return [int(input_dimension + gap * (layer_idx + 1)) for layer_idx in range(layer_num)]

    def fully_connected_layer(self, input_placeholder, output_dimension): 
        
        input_dimension1 = tf.cast(input_placeholder.shape[1], tf.int32)
        input_dimension2 = tf.cast(input_placeholder.shape[2], tf.int32)
        
        reshaped_input_placeholder = tf.reshape(input_placeholder, [-1, input_dimension2])

        weight = tf.Variable(tf.random_normal(shape = [input_dimension2, output_dimension], stddev = self.weight_stddev))
        bias = tf.Variable(tf.random_normal(shape = [output_dimension], stddev = self.bias_stddev))

        return tf.reshape(tf.matmul(reshaped_input_placeholder, weight) + bias, [-1, input_dimension1, output_dimension])

if __name__ == '__main__':
    print ('running data...')
    data_formater = Data_formater()
    data_formater.load_embedding_dictionary(vectors_file_name = 'vectors.pkl', word2id_file_name = 'word2id.pkl')
    data_formater.load_lyrics('lyric_data', lyric_fixed_length = 40)
    data_formater.load_images('cross_va_data', list(range(10)))
    data_formater.build_model_inputs()

    print ('running model...')
    structured_SVM = Structured_SVM(weight_stddev = 0.1, bias_stddev = 0.01)
    structured_SVM.build_model(image_dimension = data_formater.image_dimension, lyric_length = data_formater.lyric_length, lyric_num = data_formater.lyric_num, fully_connected_layer_num = 2, embedding_dimension = 32, C = 0.8, embedding_table = data_formater.embedding_table, pool_size = 2, conv_filter_num_list = [16, 32], filter_size = 3)
