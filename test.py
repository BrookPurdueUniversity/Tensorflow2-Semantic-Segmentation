
import helpers
import model
###############################################################################
import os,time,cv2
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#tf.reset_default_graph()    # clear all previous sessions
tf.compat.v1.reset_default_graph()

import numpy as np
import easydict

args = easydict.EasyDict({
    'image':"./data/test_image",
    'checkpoint_path': './result/ckpt/latest_model.ckpt',
    'dataset': "./data",
    'mode': 'Predict',
    'continue_training': False,
    'predict_result_dir': "./result/test_result"
})

class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))
num_classes = len(label_values)
print("\n********************** Brook: Begin prediction *********************************")
print("Dataset -->", args.dataset)
print("Num Classes -->", num_classes)
print("Image -->", args.image)


# Initializing network
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

#net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
#net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes]) 
net_input = tf.compat.v1.placeholder(tf.float32,shape=[None,None,None,3]) 
net_output = tf.compat.v1.placeholder(tf.float32,shape=[None,None,None,num_classes])

network = model.build(net_input=net_input, num_classes=num_classes)

sess.run(tf.global_variables_initializer())

saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())

if args.continue_training==True or not args.mode == "train":
    saver.restore(sess, args.checkpoint_path)

if args.image is None:
    ValueError("You must pass an image path when using prediction mode.")

# Create directories if needed
predict_dir = args.predict_result_dir
if not os.path.isdir(predict_dir):
    os.makedirs(predict_dir)
    
ls = os.listdir(args.image)
for i in range(len(ls)):
  dir = args.image + '/' + ls[i]
  loaded_image = helpers.load_image(dir)  
  input_image = np.expand_dims(np.float32(loaded_image),axis=0)/255.0
  
  st = time.time()
  output_image = sess.run(network,feed_dict={net_input:input_image})  # at this point, output_image.shape=(1,H,W,num_classes)
  run_time = time.time()-st
  
  output_image = np.array(output_image[0,:,:,:])   # at this point, output_image.shape=(H,W,num_classes)
  
  output_image = helpers.reverse_one_hot(output_image)  # at this point, output_image.shape=(H,W)
  
  out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
  file_name = ls[i]
  
  out_vis_image = cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR)
  out_vis_image = cv2.resize(out_vis_image, (722, 536))
  cv2.imwrite(predict_dir + '/' + file_name, out_vis_image)
  
  print("")
  print("Finished!")
  print("Wrote image " + "%s/%s_pred.png"%("Test", file_name))