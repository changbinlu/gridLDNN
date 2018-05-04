import tensorflow as tf
import os
import sys
sys.path.insert(0, '/home/changbinli/script/rnn/')
import logging
import time
import datetime
from dataloader import get_train_data, get_valid_data, get_scenes_weight
import numpy as np
'''
For block_intepreter with rectangle 
'''
logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class HyperParameters:
    def __init__(self, VAL_FOLD, FOLD_NAME):
        self.MODEL_SAVE = False
        self.RESTORE = False
        self.OLD_EPOCH = 0
        if not self.RESTORE:
            # Set up log directory
            self.LOG_FOLDER = './log/' + FOLD_NAME + '/'
            if not os.path.exists(self.LOG_FOLDER):
                os.makedirs(self.LOG_FOLDER)
            # Set up model directory individually
            self.SESSION_DIR = self.LOG_FOLDER + str(VAL_FOLD) + '/'
            if not os.path.exists(self.SESSION_DIR):
                os.makedirs(self.SESSION_DIR)
        else:
            self.RESTORE_DATE = '20180422'
            self.OLD_EPOCH = 2
            self.LOG_FOLDER = './log/' + self.RESTORE_DATE + '/'
            self.SESSION_DIR = self.LOG_FOLDER + str(VAL_FOLD) + '/'

        # training
        self.LEARNING_RATE = 0.001
        self.SCENES = ['scene1']
        # LSTM
        self.NUM_HIDDEN = 1024
        self.NUM_LSTM = 3
        # MLP
        self.NUM_NEURON = 1024
        self.NUM_MLP = 3

        # dropout
        self.OUTPUT_THRESHOLD = 0.5
        self.INPUT_KEEP_PROB = 1.0
        self.OUTPUT_KEEP_PROB = 0.9
        self.BATCH_SIZE = 50
        self.EPOCHS = 3
        self.FORGET_BIAS = 0.9
        self.TIMELENGTH = 1000
        self.MAX_GRAD_NORM = 5.0
        self.NUM_CLASSES = 13

        # Get rectangle
        self.VAL_FOLD = VAL_FOLD
        self.TRAIN_SET, self.PATHS = get_train_data(self.VAL_FOLD,self.SCENES,self.EPOCHS,self.TIMELENGTH)
        self.VALID_SET = get_valid_data(self.VAL_FOLD,self.SCENES, 1, self.TIMELENGTH)
        self.TOTAL_SAMPLES = len(self.PATHS)
        self.NUM_TRAIN = len(self.TRAIN_SET)
        self.NUM_TEST = len(self.VALID_SET)
        self.SET = {'train': self.TRAIN_SET,
               'test': self.VALID_SET}

    def _read_py_function(self,filename):
        filename = filename.decode(sys.getdefaultencoding())
        fx, fy = np.array([]).reshape(0, 160), np.array([]).reshape(0, 13)
        # each filename is : path1&start_index&end_index@path2&start_index&end_index
        # the total length was defined before
        for instance in filename.split('@'):
            p, start, end = instance.split('&')
            data = np.load(p)
            x = data['x'][0]
            y = data['y'][0]
            fx = np.concatenate((fx, x[int(start):int(end)]), axis=0)
            fy = np.concatenate((fy, y[int(start):int(end)]), axis=0)
        l = np.array([fx.shape[0]])
        return fx.astype(np.float32), fy.astype(np.int32), l.astype(np.int32)

    def read_dataset(self,path_set, batchsize):
        dataset = tf.data.Dataset.from_tensor_slices(path_set)
        dataset = dataset.map(
            lambda filename: tuple(tf.py_func(self._read_py_function, [filename], [tf.float32, tf.int32, tf.int32])))
        # batch = dataset.padded_batch(batchsize, padded_shapes=([None, None], [None, None], [None]))
        batch = dataset.batch(batchsize)
        return batch


    def unit_lstm(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.NUM_HIDDEN, forget_bias=self.FORGET_BIAS)
        lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell,
                                                  input_keep_prob=self.INPUT_KEEP_PROB,
                                                  output_keep_prob=self.OUTPUT_KEEP_PROB,
                                                  variational_recurrent=True,
                                                  dtype= tf.float32)
        # lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units = self.NUM_HIDDEN,
        #                                                   layer_norm = True,
        #                                                   forget_bias=self.FORGET_BIAS,
        #                                                   dropout_keep_prob= self.OUTPUT_KEEP_PROB)
        return lstm_cell

    def get_state_variables(self,cell):
        # For each layer, get the initial state and make a variable out of it
        # to enable updating its value.
        state_variables = []
        for state_c, state_h in cell.zero_state(self.BATCH_SIZE, tf.float32):
            state_variables.append(tf.contrib.rnn.LSTMStateTuple(
                tf.Variable(state_c, trainable=False),
                tf.Variable(state_h, trainable=False)))
        # Return as a tuple, so that it can be fed to dynamic_rnn as an initial state
        return tuple(state_variables)

    def get_state_update_op(self,state_variables, new_states):
        # Add an operation to update the train states with the last state tensors
        update_ops = []
        for state_variable, new_state in zip(state_variables, new_states):
            # Assign the new state to the state variables on this layer
            update_ops.extend([state_variable[0].assign(new_state[0]),
                               state_variable[1].assign(new_state[1])])
        # Return a tuple in order to combine all update_ops into a single operation.
        # The tuple's actual value should not be used.
        return tf.tuple(update_ops)

    def get_state_reset_op(self,state_variables, cell):
        # Return an operation to set each variable in a list of LSTMStateTuples to zero
        zero_states = cell.zero_state(self.BATCH_SIZE, tf.float32)
        return self.get_state_update_op(state_variables, zero_states)

    def MultiRNN(self, x, weights, seq):
        with tf.variable_scope('lstm', initializer=tf.orthogonal_initializer()):
            mlstm_cell = tf.contrib.rnn.MultiRNNCell([self.unit_lstm() for _ in range(self.NUM_LSTM)], state_is_tuple=True)
            states = self.get_state_variables(mlstm_cell)
            batch_x_shape = tf.shape(x)
            layer = tf.reshape(x, [batch_x_shape[0], -1, 160])
            # init_state = mlstm_cell.zero_state(self.BATCH_SIZE, dtype=tf.float32)
            outputs, new_states = tf.nn.dynamic_rnn(cell=mlstm_cell,
                                               inputs=layer,
                                               initial_state= states,
                                               dtype=tf.float32,
                                               time_major=False,
                                               sequence_length=seq)
            update_op = self.get_state_update_op(states, new_states)
            outputs = tf.reshape(outputs, [-1, self.NUM_HIDDEN])
        with tf.variable_scope('mlp'):
            if self.NUM_MLP == 0:
                top = tf.nn.dropout(tf.matmul(outputs, weights['out']),
                                    keep_prob=self.OUTPUT_KEEP_PROB)
                original_out = tf.reshape(top, [batch_x_shape[0], -1, self.NUM_CLASSES])
                return original_out, update_op
            elif self.NUM_MLP == 1:
                l1 = tf.nn.dropout(tf.matmul(outputs, weights['h1']),
                                    keep_prob=self.OUTPUT_KEEP_PROB)
                l1 = tf.nn.relu(l1)
                top = tf.nn.dropout(tf.matmul(l1, weights['mlpout']),
                                    keep_prob=self.OUTPUT_KEEP_PROB)
                original_out = tf.reshape(top, [batch_x_shape[0], -1, self.NUM_CLASSES])
                return original_out, update_op
            elif self.NUM_MLP == 2:
                l1 = tf.nn.dropout(tf.matmul(outputs, weights['h1']),
                                   keep_prob=self.OUTPUT_KEEP_PROB)
                l1 = tf.nn.relu(l1)
                l2 = tf.nn.dropout(tf.matmul(l1, weights['h2']),
                                   keep_prob=self.OUTPUT_KEEP_PROB)
                l2 = tf.nn.relu(l2)
                top = tf.nn.dropout(tf.matmul(l2, weights['mlpout']),
                                    keep_prob=self.OUTPUT_KEEP_PROB)
                original_out = tf.reshape(top, [batch_x_shape[0], -1, self.NUM_CLASSES])
                return original_out, update_op
            elif self.NUM_MLP == 3:
                l1 = tf.nn.dropout(tf.matmul(outputs, weights['h1']),
                                   keep_prob=self.OUTPUT_KEEP_PROB)
                l1 = tf.nn.relu(l1)
                l2 = tf.nn.dropout(tf.matmul(l1, weights['h2']),
                                   keep_prob=self.OUTPUT_KEEP_PROB)
                l2 = tf.nn.relu(l2)
                l3 = tf.nn.dropout(tf.matmul(l2, weights['h3']),
                                   keep_prob=self.OUTPUT_KEEP_PROB)
                l3 = tf.nn.relu(l3)
                top = tf.nn.dropout(tf.matmul(l3, weights['mlpout']),
                                    keep_prob=self.OUTPUT_KEEP_PROB)
                original_out = tf.reshape(top, [batch_x_shape[0], -1, self.NUM_CLASSES])
                return original_out, update_op

    def setup_logger(self,logger_name, log_file, level=logging.DEBUG):
        l = logging.getLogger(logger_name)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        fileHandler = logging.FileHandler(log_file, mode='w')
        fileHandler.setFormatter(formatter)
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)

        l.setLevel(level)
        l.addHandler(fileHandler)
        l.addHandler(streamHandler)

    def validation_accuracy(self,logits, Y,mask_zero_frames):
        with tf.name_scope("train_accuracy"):
            # add a threshold to round the output to 0 or 1
            # logits is already being sigmoid
            predicted = tf.to_int32(tf.sigmoid(logits) > self.OUTPUT_THRESHOLD)
            TP = tf.count_nonzero(predicted * Y  * mask_zero_frames)
            # mask padding, zero_frame,
            TN = tf.count_nonzero((predicted - 1) * (Y - 1) * mask_zero_frames)
            FP = tf.count_nonzero(predicted * (Y - 1) * mask_zero_frames)
            FN = tf.count_nonzero((predicted - 1) * Y * mask_zero_frames)
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1 = 2 * precision * recall / (precision + recall)
            # TPR = TP/(TP+FN)
            sensitivity = recall
            specificity = TN / (TN + FP)
    def main(self):
        # tensor holder
        train_batch = self.read_dataset(self.SET['train'], self.BATCH_SIZE)
        test_batch = self.read_dataset(self.SET['test'], self.BATCH_SIZE)

        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, train_batch.output_types, train_batch.output_shapes)
        with tf.device('/cpu:0'):
            X, Y, seq = iterator.get_next()
        # get mask matrix for loss fuction, will be used after round output
        mask_zero_frames = tf.cast(tf.not_equal(Y, -1), tf.int32)
        seq = tf.reshape(seq, [self.BATCH_SIZE])  # original sequence length, only used for RNN

        train_iterator = train_batch.make_initializable_iterator()
        test_iterator = test_batch.make_initializable_iterator()
        # Define weights

        weights = {
            'out': tf.get_variable('out', shape=[self.NUM_HIDDEN, self.NUM_CLASSES],
                                   initializer=tf.contrib.layers.xavier_initializer()),

            'h1': tf.get_variable('h1',shape=[self.NUM_HIDDEN, self.NUM_NEURON],
                                  initializer=tf.contrib.layers.xavier_initializer()),
            'h2': tf.get_variable('h2',shape=[self.NUM_NEURON, self.NUM_NEURON],
                                  initializer=tf.contrib.layers.xavier_initializer()),
            'h3': tf.get_variable('h3',shape=[self.NUM_NEURON, self.NUM_NEURON],
                                  initializer=tf.contrib.layers.xavier_initializer()),
            'mlpout': tf.get_variable('mlpout', shape=[self.NUM_NEURON, self.NUM_CLASSES],
                                   initializer=tf.contrib.layers.xavier_initializer())
        }


        logits, update_op = self.MultiRNN(X, weights, seq)

        # Define loss and optimizer
        w = get_scenes_weight(self.SCENES,self.VAL_FOLD)

        with tf.variable_scope('loss'):
            # convert nan to +1
            # add_nan_one = tf.ones(tf.shape(mask_nan), dtype=tf.int32) - mask_nan
            # mask_Y = tf.add(mask_Y * mask_nan, add_nan_one)

            # assign 0 frames zero cost
            number_zero_frame = tf.reduce_sum(tf.cast(tf.equal(Y, -1), tf.int32))
            loss_op = tf.nn.weighted_cross_entropy_with_logits(tf.cast(Y, tf.float32), logits, tf.constant(w))
            # number of frames without zero_frame
            total = tf.cast(tf.reduce_sum(seq) - number_zero_frame, tf.float32)
            # eliminate zero_frame loss
            loss_op = tf.reduce_sum(loss_op * tf.cast(mask_zero_frames, tf.float32)) / total
        with tf.variable_scope('optimize'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE)
            # train_op = optimizer.minimize(loss_op)
            gradients, variables = zip(*optimizer.compute_gradients(loss_op))
            gradients, _ = tf.clip_by_global_norm(gradients, self.MAX_GRAD_NORM)
            train_op = optimizer.apply_gradients(zip(gradients, variables))
        with tf.name_scope("train_accuracy"):
            # add a threshold to round the output to 0 or 1
            # logits is already being sigmoid
            predicted = tf.to_int32(tf.sigmoid(logits) > self.OUTPUT_THRESHOLD)
            TP = tf.count_nonzero(predicted * Y  * mask_zero_frames)
            # mask padding, zero_frame,
            TN = tf.count_nonzero((predicted - 1) * (Y - 1) * mask_zero_frames)
            FP = tf.count_nonzero(predicted * (Y - 1) * mask_zero_frames)
            FN = tf.count_nonzero((predicted - 1) * Y * mask_zero_frames)
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1 = 2 * precision * recall / (precision + recall)
            # TPR = TP/(TP+FN)
            sensitivity = recall
            specificity = TN / (TN + FP)

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        log_name = 'log' + str(self.VAL_FOLD)
        if not self.RESTORE:
            log_dir = self.LOG_FOLDER + str(self.VAL_FOLD)+'.txt'
        else:
            log_dir = self.LOG_FOLDER + 'new' + str(self.VAL_FOLD) + '.txt'
        self.setup_logger(log_name,log_file= log_dir)

        logger = logging.getLogger(log_name)
        tf.logging.set_verbosity(tf.logging.INFO)

        # Start training
        with tf.Session() as sess:
            logger.info('''
                                    K_folder:{}
                                    Epochs: {}
                                    Number of lstm layer: {}
                                    Number of lstm neuron: {}
                                    Number of mlp layer: {}
                                    Number of mlp neuron: {}
                                    Batch size: {}
                                    FORGET_BIAS: {}
                                    TIMELENGTH: {}
                                    Dropout: {}
                                    Scenes:{}'''.format(
                self.VAL_FOLD,
                self.EPOCHS + self.OLD_EPOCH,
                self.NUM_LSTM,
                self.NUM_HIDDEN,
                self.NUM_MLP,
                self.NUM_NEURON,
                self.BATCH_SIZE,
                self.FORGET_BIAS,
                self.TIMELENGTH,
                self.OUTPUT_KEEP_PROB,
                self.SCENES))
            train_handle = sess.run(train_iterator.string_handle())
            test_handle = sess.run(test_iterator.string_handle())
            # Run the initializer if restore == False
            if not self.RESTORE:
                sess.run(init)
            else:
                saver.restore(sess,self.SESSION_DIR + 'model.ckpt')
                print("Model restored.")
            section = '\n{0:=^40}\n'
            logger.info(section.format('Run training epoch'))
            # final_average_loss = 0.0

            # add previous epoch if restore the model
            ee = 1 + + self.OLD_EPOCH
            # initialization for each epoch
            train_cost, sen, spe, f = 0.0, 0.0, 0.0, 0.0

            epoch_start = time.time()

            sess.run(train_iterator.initializer)
            n_batches = int(self.NUM_TRAIN / self.BATCH_SIZE)
            batch_per_epoch = int(n_batches / self.EPOCHS)
            # print(sess.run([seq, train_op],feed_dict={handle:train_handle}))
            for num in range(1, n_batches + 1):

                loss, _, se, sp, tempf1, _ = sess.run([loss_op, train_op, sensitivity, specificity, f1,update_op],
                                                   feed_dict={handle: train_handle})

                logger.debug(
                    'Train cost: %.2f | Accuracy: %.2f | Sensitivity: %.2f | Specificity: %.2f| F1-score: %.2f',
                    loss, (se + sp) / 2, se, sp, tempf1)
                train_cost = train_cost + loss
                sen = sen + se
                spe = spe + sp
                f = tempf1 + f

                if (num % batch_per_epoch == 0):
                    epoch_duration0 = time.time() - epoch_start
                    logger.info(
                        '''Epochs: {},train_cost: {:.3f},Train_accuracy: {:.3f},Sensitivity: {:.3f},Specificity: {:.3f},F1-score: {:.3f},time: {:.2f} sec'''
                            .format(ee ,
                                    train_cost / batch_per_epoch,
                                    ((sen + spe) / 2) / batch_per_epoch,
                                    sen / batch_per_epoch,
                                    spe / batch_per_epoch,
                                    f / batch_per_epoch,
                                    epoch_duration0))

                    # for validation
                    train_cost, sen, spe, f = 0.0, 0.0, 0.0, 0.0
                    v_batches_per_epoch = int(self.NUM_TEST / self.BATCH_SIZE)
                    epoch_start = time.time()
                    sess.run(test_iterator.initializer)
                    for _ in range(v_batches_per_epoch):
                        se, sp, tempf1 = sess.run([sensitivity, specificity, f1], feed_dict={handle: test_handle})
                        sen = sen + se
                        spe = spe + sp
                        f = tempf1 + f
                    epoch_duration1 = time.time() - epoch_start

                    logger.info(
                        '''Epochs: {},Validation_accuracy: {:.3f},Sensitivity: {:.3f},Specificity: {:.3f},F1 score: {:.3f},time: {:.2f} sec'''
                            .format(ee,
                                    ((sen + spe) / 2) / v_batches_per_epoch,
                                    sen / v_batches_per_epoch,
                                    spe / v_batches_per_epoch,
                                    f / v_batches_per_epoch,
                                    epoch_duration1))
                    print(ee)
                    ee += 1
                    train_cost, sen, spe, f = 0.0, 0.0, 0.0, 0.0
                    epoch_start = time.time()
            if self.MODEL_SAVE:
                save_path = saver.save(sess, self.SESSION_DIR + 'model.ckpt')
                print("Model saved in path: %s" % save_path)




if __name__ == "__main__":
    # fname = datetime.datetime.now().strftime("%Y%m%d")
    fname ='test'
    for i in range(1,7):
        with tf.Graph().as_default():
            hyperparameters = HyperParameters(VAL_FOLD=i, FOLD_NAME= fname)
            hyperparameters.main()