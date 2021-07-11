from __future__ import print_function

import helpers
import model
###############################################################################
import os,time,cv2

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#tf.reset_default_graph()    # deprecated
tf.compat.v1.reset_default_graph()

#import tensorflow.contrib.slim as slim
import tf_slim as slim

import numpy as np
import time, datetime
import argparse
import random

# use 'Agg' on matplotlib so that plots could be generated even without Xserver
# running
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train for')
parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
parser.add_argument('--checkpoint_step', type=int, default=1, help='How often to save checkpoints (epochs)')
parser.add_argument('--checkpoint_dir', type=str, default='./result/ckpt', help='path to save the trained model')
parser.add_argument('--validation_step', type=int, default=1, help='How often to perform validation (epochs)')
parser.add_argument('--continue_training', type=str2bool, default=None, help='Whether to continue training from a checkpoint')
parser.add_argument('--dataset', type=str, default="./data", help='Dataset you are using.')
parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
parser.add_argument('--num_val_images', type=int, default=8, help='The number of images to used for validations')
parser.add_argument('--validation_dir', type=str, default='./result/validation', help='path to save the trained result')
args = parser.parse_args(args=[])

# Get the names of the classes so we can record the evaluation results
class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))
class_names_string = ""
for class_name in class_names_list:
    if not class_name == class_names_list[-1]:
        class_names_string = class_names_string + class_name + ", "
    else:
        class_names_string = class_names_string + class_name
num_classes = len(label_values)

'''
# Allow GPU for tensorflow < 2.0
config = tf.ConfigProto()   
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)
'''

# Allow GPU for tensorflow >= 2.0
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)


print("Preparing the model ...")
#net_input = tf.placeholder(tf.float32,shape=[None,None,None,3]) 
tf.compat.v1.disable_eager_execution()
net_input = tf.compat.v1.placeholder(tf.float32,shape=[None,None,None,3]) 
#net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes])
net_output = tf.compat.v1.placeholder(tf.float32,shape=[None,None,None,num_classes])

network = model.build(net_input=net_input, num_classes=num_classes)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=net_output))
opt = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.995).minimize(loss, var_list=[var for var in tf.trainable_variables()])

saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())
helpers.count_params()

# Load a previous checkpoint if desired
checkpoint_dir = args.checkpoint_dir
model_checkpoint_name = checkpoint_dir + '/latest_model.ckpt'
if args.continue_training:
    print('******************** Brook: Continue Training: Loaded latest model checkpoint ********************')
    saver.restore(sess, model_checkpoint_name)
else:
    print('******************** Brook: First-time Training: There are no latest model checkpoint ********************')
    
avg_scores_per_epoch = []


# Load the data
print("Loading the data ...")
input_names,output_names = helpers.prepare_data(dataset_dir=args.dataset)

print("\n******************* Brook: Begin training **************************")  
print("Dataset -->", args.dataset)             
print("Num Epochs -->", args.num_epochs)
print("Batch Size -->", args.batch_size)
print("Num Classes -->", num_classes)
print("")

avg_loss_per_epoch = []
avg_scores_per_epoch = []
avg_iou_per_epoch = []

all_id_list = np.arange(0, len(input_names))
# Do the training here   
for epoch in range(args.epoch_start_i, args.num_epochs):
    
    # generate the train and val
    train_input_names = []
    train_output_names = []
    val_input_names = []
    val_output_names = []
    
    val_id_list = random.sample(range(0, len(input_names)), args.num_val_images)
    train_id_list = [id_value for id_value in all_id_list if id_value not in val_id_list]
    
    # subsidiary: increase the randomness for train and val
    for i in range(len(val_id_list)):
      id_random = np.random.randint(0,len(train_id_list))
      if train_id_list[id_random] not in val_id_list:
        tmp = val_id_list[i]
        val_id_list[i] = train_id_list[id_random]
        train_id_list[id_random] = tmp
      else:
        id_random = np.random.randint(0,len(train_id_list))
        
    for val_id in val_id_list:
      val_input_names.append(input_names[val_id])
      val_output_names.append(output_names[val_id])
    val_input_names.sort(), val_output_names.sort()
    
    for train_id in train_id_list:
      train_input_names.append(input_names[train_id])
      train_output_names.append(output_names[train_id])
    train_input_names.sort(), train_output_names.sort()
     
    # Which validation images do we want
    val_indices = []
    num_vals = min(args.num_val_images, len(val_input_names))

    # Set random seed to make sure models are validated on the same validation images.
    # So you can compare the results of different models more intuitively.
    random.seed(16)
    val_indices=random.sample(range(0,len(val_input_names)),num_vals)

  
    # Create directories if needed    
    result_dir = args.validation_dir
    val_out_vis_dir = result_dir + '/' + 'val_out_vis_image_' + str(epoch)
    if not os.path.isdir(val_out_vis_dir):
        os.makedirs(val_out_vis_dir)
        
    current_losses = []

    cnt=0

    # Equivalent to shuffling
    id_list = np.random.permutation(len(train_input_names))

    num_iters = int(np.floor(len(id_list) / args.batch_size))
    st = time.time()
    epoch_st=time.time()
    for i in range(num_iters):
        # st=time.time()

        input_image_batch = []
        output_image_batch = []

        # Collect a batch of images
        for j in range(args.batch_size):
            index = i*args.batch_size + j
            id = id_list[index]
            input_image = helpers.load_image(train_input_names[id])
            output_image = helpers.load_image(train_output_names[id])

            with tf.device('/cpu:0'):
                # Prep the data. Make sure the labels are in one-hot format
                input_image = np.float32(input_image) / 255.0
                output_image = np.float32(helpers.one_hot_it(label=output_image, label_values=label_values))

                input_image_batch.append(np.expand_dims(input_image, axis=0))
                output_image_batch.append(np.expand_dims(output_image, axis=0))

        if args.batch_size == 1:
            input_image_batch = input_image_batch[0]
            output_image_batch = output_image_batch[0]
        else:
            input_image_batch = np.squeeze(np.stack(input_image_batch, axis=1))
            output_image_batch = np.squeeze(np.stack(output_image_batch, axis=1))

        # Do the training
        _,current=sess.run([opt,loss],feed_dict={net_input:input_image_batch,net_output:output_image_batch})
        current_losses.append(current)
        cnt = cnt + args.batch_size
        if cnt % 20 == 0:
            string_print = "Epoch = %d Count = %d Current_Loss = %.4f Time = %.2f"%(epoch,cnt,current,time.time()-st)
            helpers.LOG(string_print)
            st = time.time()

    mean_loss = np.mean(current_losses)
    avg_loss_per_epoch.append(mean_loss)

    # Create directories if needed      
    checkpoint_dir = args.checkpoint_dir
    if not os.path.isdir("%s/%04d"%(checkpoint_dir,epoch)):
        os.makedirs("%s/%04d"%(checkpoint_dir,epoch))
    
    model_checkpoint_name = checkpoint_dir + '/latest_model.ckpt'
    # Save latest checkpoint to same file name
    print("Saving latest checkpoint")
    saver.save(sess,model_checkpoint_name)

    if val_indices != 0 and epoch % args.checkpoint_step == 0:
        print("Saving checkpoint for this epoch")
        saver.save(sess,"%s/%04d/model.ckpt"%(checkpoint_dir,epoch))

    if epoch % args.validation_step == 0:
        print("Performing validation")
        target=open("%s/%04d/val_scores.csv"%(checkpoint_dir,epoch),'w')
        target.write("val_name, avg_accuracy, precision, recall, f1 score, mean iou, %s\n" % (class_names_string))

        scores_list = []
        class_scores_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        iou_list = []


        # Do the validation on a small set of validation images
        for ind in val_indices:

            input_image = np.expand_dims(np.float32(helpers.load_image(val_input_names[ind])),axis=0)/255.0
            gt = helpers.load_image(val_output_names[ind])
            gt = helpers.reverse_one_hot(helpers.one_hot_it(gt, label_values))

            # st = time.time()

            output_image = sess.run(network,feed_dict={net_input:input_image})
            output_image = np.array(output_image[0,:,:,:])
            output_image = helpers.reverse_one_hot(output_image)
            out_vis_image = helpers.colour_code_segmentation(output_image, label_values)

            accuracy, class_accuracies, prec, rec, f1, iou = helpers.evaluate_segmentation(pred=output_image, label=gt, num_classes=num_classes)

            file_name = helpers.filepath_to_name(val_input_names[ind])
            target.write("%s, %f, %f, %f, %f, %f"%(file_name, accuracy, prec, rec, f1, iou))
            for item in class_accuracies:
                target.write(", %f"%(item))
            target.write("\n")

            scores_list.append(accuracy)
            class_scores_list.append(class_accuracies)
            precision_list.append(prec)
            recall_list.append(rec)
            f1_list.append(f1)
            iou_list.append(iou)

            gt = helpers.colour_code_segmentation(gt, label_values)

            file_name = os.path.basename(val_input_names[ind])
            # file_name = os.path.splitext(file_name)[0]
            
            cv2.imwrite(val_out_vis_dir + '/' + file_name, cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))

        target.close()

        avg_score = np.mean(scores_list)
        class_avg_scores = np.mean(class_scores_list, axis=0)
        avg_scores_per_epoch.append(avg_score)
        avg_precision = np.mean(precision_list)
        avg_recall = np.mean(recall_list)
        avg_f1 = np.mean(f1_list)
        avg_iou = np.mean(iou_list)
        avg_iou_per_epoch.append(avg_iou)

        print("\nAverage validation accuracy for epoch # %04d = %f"% (epoch, avg_score))
        print("Average per class validation accuracies for epoch # %04d:"% (epoch))
        for index, item in enumerate(class_avg_scores):
            print("%s = %f" % (class_names_list[index], item))
        print("Validation precision = ", avg_precision)
        print("Validation recall = ", avg_recall)
        print("Validation F1 score = ", avg_f1)
        print("Validation IoU score = ", avg_iou)

    epoch_time=time.time()-epoch_st
    remain_time=epoch_time*(args.num_epochs-1-epoch)
    m, s = divmod(remain_time, 60)
    h, m = divmod(m, 60)
    if s!=0:
        train_time="Remaining training time = %d hours %d minutes %d seconds\n"%(h,m,s)
    else:
        train_time="Remaining training time : Training completed.\n"
    helpers.LOG(train_time)
    scores_list = []


    fig1, ax1 = plt.subplots(figsize=(11, 8))


    ax1.plot(range(epoch+1), avg_scores_per_epoch)
    ax1.set_title("Average validation accuracy vs epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Avg. val. accuracy")


    plt.savefig(result_dir + '/' + 'accuracy_vs_epochs.png')

    plt.clf()

    fig2, ax2 = plt.subplots(figsize=(11, 8))

    ax2.plot(range(epoch+1), avg_loss_per_epoch)
    ax2.set_title("Average loss vs epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Current loss")

    plt.savefig(result_dir + '/' + 'loss_vs_epochs.png')

    plt.clf()

    fig3, ax3 = plt.subplots(figsize=(11, 8))

    ax3.plot(range(epoch+1), avg_iou_per_epoch)
    ax3.set_title("Average IoU vs epochs")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Current IoU")

    plt.savefig(result_dir + '/' + 'iou_vs_epochs.png')
