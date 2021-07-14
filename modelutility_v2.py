import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import datetime
import itertools


def plot_loss_hist(train_loss_curr, val_loss_curr):
    curr_epoch = np.nonzero(val_loss_curr)[0][-1] + 1

    plt.plot(np.arange(1, curr_epoch + 1), train_loss_curr[:curr_epoch], label='Training')
    plt.plot(np.arange(1, curr_epoch + 1), val_loss_curr[:curr_epoch], label='Validation')
    plt.grid(True)

    plt.title('Epoch {} Loss history'.format(curr_epoch))
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# Run model
def run_model(sess, Xy_var, is_training, loss_var, rel_err_var, train_step, Xy_d, is_training_d,
              epoch_idx=0, batch_size=100, print_every=100):
    
    # shuffle indices
    num_samples = Xy_d[-1].shape[0]
    train_indices = np.arange(num_samples)
    np.random.shuffle(train_indices)

    # counter
    iter_cnt = 0
    tic = time.process_time()

    # keep track of losses and accuracy
    average_rel_err = 0
    average_loss = 0

    # make sure we iterate over the dataset once
    for i in range(int(num_samples / batch_size) + 1):
        # create a feed dictionary for this batch
        start_idx = (i * batch_size) % num_samples
        idxs = train_indices[start_idx:start_idx + batch_size]
        feed_dict_xy = {v: d[idxs] for v, d in zip(Xy_var,Xy_d)}
        feed_dict_train = {is_training: is_training_d}
        feed_dict = {**feed_dict_xy,**feed_dict_train}

        # get actual batch size (batch close to the end<batch_size)
        actual_batch_size = Xy_d[-1][i:i + batch_size].shape[0]

        # during training step perform a training step, duh
        if is_training_d:
            loss, rel_err, _ = sess.run([loss_var, rel_err_var, train_step], feed_dict=feed_dict)
        else:  # at inference
            loss, rel_err = sess.run([loss_var, rel_err_var], feed_dict=feed_dict)

        # print every now and then
        if is_training_d and (iter_cnt % print_every) == 0:
            print("Iteration {0}: with minibatch training loss = {1:.3g} and relative error of {2:.2g}"
                  .format(iter_cnt, loss, rel_err))
        iter_cnt += 1

        average_loss += loss * actual_batch_size
        average_rel_err += rel_err * actual_batch_size

    average_loss /= num_samples
    average_rel_err /= num_samples

    print("Epoch {0}, Overall loss = {1:.3g}, relative error = {2:.3g}, and process time of {3:.2g}s"
          .format(epoch_idx + 1, average_loss, average_rel_err, time.process_time() - tic))

    
    return average_loss, average_rel_err

def load_data(data_folder,output_folder,set_name = 'trainval'): 
    
    # define the input filename format
    image_log_name = 'image_log'
    pv_log_name = 'pv_log'
    pv_pred_name = 'pv_pred'
    times_name = 'times'
    
    print('loading data')
    # load PV output and images
    pv_log = np.load(os.path.join(data_folder,pv_log_name+'_'+set_name+'.npy'))
    images = np.load(os.path.join(data_folder,image_log_name+'_'+set_name+'.npy'))
    pv_pred = np.load(os.path.join(data_folder,pv_pred_name+'_'+set_name+'.npy'))
    times = np.load(os.path.join(data_folder,times_name+'_'+set_name+'.npy'))

    # stack up the history and colors into a unified channel
    print('stacking image log\'s channels')
    images = images.transpose((0,2,3,4,1))
    images = images.reshape((images.shape[0],images.shape[1],images.shape[2],-1))
    
    if set_name == 'trainval':
        # Shuffling by day blocks if the data is training data
        shuffled_idxs = day_block_shuffle(times)
        # Save the shuffling for reproducibility
        np.save(os.path.join(output_folder, 'shuffled_indices'), np.array(shuffled_idxs))
    else:
        shuffled_idxs = np.arange(images.shape[0])
    
    return [images[shuffled_idxs],pv_log[shuffled_idxs],pv_pred[shuffled_idxs], times[shuffled_idxs]] 

def day_block_shuffle(times_trainval):
    # Only keep the date of each time point
    dates_trainval = np.zeros_like(times_trainval, dtype=datetime.date)
    for i in range(len(times_trainval)):
        dates_trainval[i] = times_trainval[i].date()

    # Chop the indices into blocks, so that each block contains the indices of the same day
    unique_dates = np.unique(dates_trainval)
    blocks = []
    for i in range(len(unique_dates)):
        blocks.append(np.where(dates_trainval == unique_dates[i])[0])

    # shuffle the blocks, and chain it back together
    np.random.shuffle(blocks)
    shuffled_indices = list(itertools.chain.from_iterable(blocks))

    return shuffled_indices

def find_idx_with_dates(all_times,test_dates):
    # In an array of datetime, find all points belong to certain dates
    idx=[]
    for test_day in test_dates:
        test_day_end = test_day + datetime.timedelta(days = 1)
        idx+=np.nonzero((all_times>test_day)*(all_times<test_day_end))[0].tolist()
    return idx

def cv_split(Xy_data, fold_index, num_fold):
    # randomly divides into a training set and a validation set
    num_samples = Xy_data[0].shape[0]
    indices = np.arange(num_samples)

    # finding training and validation indices
    val_mask = np.zeros(len(indices), dtype=bool)
    val_mask[int(fold_index / num_fold * num_samples):int((fold_index + 1) / num_fold * num_samples)] = True
    val_indices = indices[val_mask]
    train_indices = indices[np.logical_not(val_mask)]

    # shuffle indices
    np.random.shuffle(val_indices)
    np.random.shuffle(train_indices)
    
    # Initialize the training and validation data set list
    Xy_train = []
    Xy_val = []
    # obtain training and validation data
    for one_data in Xy_data:
        one_train, one_val = one_data[train_indices], one_data[val_indices]
        Xy_train.append(one_train)
        Xy_val.append(one_val)
        
    return Xy_train, Xy_val

def del_checkpoint(save_directory,model_name, last_idx,threshold = 5):
    
    # The last x(5) repetition where we don't see an improved validation loss
    for epoch_idx in range(last_idx - threshold+1, last_idx+1):
        # delete the metagraph, data and index file
        os.remove(os.path.join(save_directory, model_name+'-'+str(epoch_idx)+'.data-00000-of-00001'))
        os.remove(os.path.join(save_directory, model_name+'-'+str(epoch_idx)+'.index'))
        os.remove(os.path.join(save_directory, model_name+'-'+str(epoch_idx)+'.meta'))

def save_history(output_folder, train_loss_hist, train_error_hist, val_loss_hist, val_error_hist):
    np.save(os.path.join(output_folder, "train_loss.npy"), train_loss_hist)
    np.save(os.path.join(output_folder, "train_error.npy"), train_error_hist)
    np.save(os.path.join(output_folder, "val_loss.npy"), val_loss_hist)
    np.save(os.path.join(output_folder, "val_error.npy"), val_error_hist)
        
        
def run_training(num_fold, num_epochs, plotting, output_folder, model_name, device,
                 Xy_var, is_training, loss_var, rel_err_var, train_step, Xy_data,
                batch_size = 100, print_every = 20):

    # initialize loss and rel_err history list
    train_loss_hist = np.zeros([num_fold, num_epochs], dtype='float32')
    train_error_hist = np.zeros_like(train_loss_hist)
    val_loss_hist = np.zeros_like(train_loss_hist)
    val_error_hist = np.zeros_like(train_loss_hist)

    # Initialize a tf graph and session for the training process
    # Config for the session to only use GPU memory as needed
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = device[-1]
    sess = tf.Session(config = config)
    
    for j in range(num_fold):
        
        # Loading in and dividing data
        Xy_train, Xy_val = cv_split(Xy_data, j, num_fold)
        
        # Saving the model
        saver = tf.train.Saver(max_to_keep=6)
        #Create saving path if it didnt exist already
        save_directory = os.path.join(output_folder,'repetition_'+str(j))
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
            
        with tf.device(device[:-1]+'0'):  # cpu or gpu
            # initialize all variable
            sess.run(tf.global_variables_initializer())

            for epoch_idx in range(num_epochs):
                print('Training')
                train_loss, train_error = run_model(sess, Xy_var, is_training, loss_var, rel_err_var, train_step, 
                                                    Xy_train, True, epoch_idx, batch_size, print_every)
                print('Validation')
                val_loss, val_error = run_model(sess, Xy_var, is_training, loss_var, rel_err_var, None, Xy_val, False)
                
                # Storing loss and error history
                train_loss_hist[j, epoch_idx] = train_loss
                train_error_hist[j, epoch_idx] = train_error
                val_loss_hist[j, epoch_idx] = val_loss
                val_error_hist[j, epoch_idx] = val_error

                # Saving
                print('Saving model checkpoint')                
                # Save current model
                save_path = saver.save(sess, os.path.join(save_directory,model_name), global_step=epoch_idx)

                if plotting:
                    plot_loss_hist(train_loss_hist[j], val_loss_hist[j])
                
                # Save loss after every epoch
                save_history(output_folder, train_loss_hist, train_error_hist, val_loss_hist, val_error_hist)
                
                # Stop training when the stopping criteria has been met
                if epoch_idx - np.argmin(val_loss_hist[j, :epoch_idx + 1]) >= 5 and epoch_idx >= 10:
                    print('Validation error has stopped improving! Training on repetition {0} stopped!'.format(j + 1))
                    break

            # delete the separted training and validation variable to release memory
            del Xy_train, Xy_val
            
            # delete the non-optimal models in each training repetition to free up hard drive
            del_checkpoint(save_directory,model_name, epoch_idx)
    
    return save_path, [train_loss_hist, train_error_hist, val_loss_hist, val_error_hist]

def inference_multirep(num_rep, model_name,output_folder,device,
                       Xy_var_test, is_training, pred_y_var, data_test, batch_size = 256):

    # Creating a compute session with the current graph
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = device[-1]
    sess = tf.Session(config = config)

    # Define a saver within the current computational graph
    saver = tf.train.Saver()

    # Restore loss history
    val_loss_hist = np.load(os.path.join(output_folder,"val_loss.npy"))

    # Initialize variable to store test set prediction
    num_samples = data_test[-1].shape[0]
    train_indices = np.arange(num_samples)
    pred_y_value = np.zeros((num_rep, num_samples))
    best_model_idx = np.zeros((num_rep),dtype = int)
    
    # Do the inference for each model
    with tf.device(device[:-1]+'0'):  # cpu or gpu
        for i in range(num_rep):
            # initialize all variable
            sess.run(tf.global_variables_initializer())
            
            best_model_idx[i] = np.argmin(val_loss_hist[i][val_loss_hist[i]>0])
            saver.restore(sess,os.path.join('models',model_name,'repetition_'+str(i), model_name+'-'+str(best_model_idx[i])))
            
            # make sure we iterate over the dataset once
            for j in range(int(num_samples / batch_size) + 1):
                # identify the indices for this batch
                start_idx = (j * batch_size) % num_samples
                idxs = train_indices[start_idx:start_idx + batch_size]
                
                # create a feed dictionary for this batch
                feed_dict_xy = {v: d[idxs] for v, d in zip(Xy_var_test,data_test)}
                feed_dict_train = {is_training: False}                
                feed_dict = {**feed_dict_xy,**feed_dict_train}

                # Obtain prediction value for the batch
                pred_y_value[i,idxs] = sess.run([pred_y_var], feed_dict=feed_dict)

            print('test set inference finished for model {0} of {1}'.format(i+1,num_rep))
    
    return pred_y_value