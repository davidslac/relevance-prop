import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#x = tf.placeholder("float", shape = [None,100])
#y_ = tf.placeholder("float", shape = [None,7])

#W2 = tf.Variable(tf.random_uniform([6,6],-.1,.1))
#b2 = tf.Variable(tf.random_uniform([],-.1,.1))

#W1 = tf.Variable(tf.random_uniform([35,7],-.1,.1))
#b1 = tf.Variable(tf.random_uniform([7],-.1,.1))

#W = tf.Variable(tf.random_uniform([100,35],-.1,.1))
#b = tf.Variable(tf.random_uniform([35],-.1,.1))

#y = tf.nn.softmax(tf.matmul (   tf.nn.relu(tf.matmul(x,W) + b) ,W1) + b1 )
#y_ = tf.placeholder("float", shape = [None,7])
#cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#train_step = tf.train.AdagradOptimizer(.1, initial_accumulator_value=0.1, use_l#ocking=False, name='Adagrad')
#train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
#init = tf.initialize_all_variables()
#sess = tf.Session()
#sess.run(init)
#sess.run(W)
#sess.run(b)




 #csv = np.genfromtxt ('mnist_train.csv', delimiter=",")
 #Xq = csv[:,1:785]
 #Yq = csv[:,0]
 #Temp = np.zeros((60000,10))
 #Yq = Yq.astype('int')
 #a = range(0,60000)
 #Temp[a,Yq] = 1

#a = np.load('DeepLearningSet2.npy')
a = np.genfromtxt ('mnist_train.csv', delimiter=",")
b = a
d1 = a.shape[0]
d2 = a.shape[1]

dstar = d2-1
Xq = a[:,1:d2]
Yq = a[:,0]
unique = np.unique(Yq)
NumClasses = len(unique)
Temp = np.zeros((d1,NumClasses))
Yq = Yq.astype('int')
a = range(0,d1)
Temp[a,Yq] = 1
print 'LETTSSSS SEEE WHATS AHPPEE'
print Temp.shape
dime = np.sqrt(dstar)
dime = dime.astype(int)
print dime

#for j in range(0,100):
#    for k in range(0,729):
#        Xtrain = Xq[k*2: (k+1)*2,:]
#        Ytrain = Temp[k*2:(k+1)*2,:]
#        _,W_val,loss_val=sess.run([train_step,W,cross_entropy],feed_dict={x: Xtrain, y_: Ytrain } )
#        print loss_val



#correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#print(sess.run(accuracy, feed_dict={x: Xq, y_: Temp}))#print(accuracy.eval(feed_dict={x: Xq, y_: Temp}))
#Wstar = sess.run(W)

#import matplotlib.pyplot as plt

#for k in range(0,15):
#    temp = Wstar[:,k]
#    Hola = np.reshape(temp,(10,10))
#    print k
#    plt.imshow(Hola)
#    plt.show()

print 'im here0'

x = tf.placeholder("float", shape = [None,dstar])
y_ = tf.placeholder("float", shape = [None,NumClasses])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.011)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

print 'im here'



W_fc1 = tf.Variable(tf.random_normal([784,300],stddev = .011),name = "W1")
b_fc1 = tf.Variable(tf.random_normal([300],stddev = .011),name = "b1")

W_fc2 = tf.Variable(tf.random_normal([300,100],stddev = .011),name = "W2")
b_fc2 = tf.Variable(tf.random_normal([100],stddev = .011),name = "b2")

W_fc3 = tf.Variable(tf.random_normal([100,NumClasses],stddev = .011),name = "W3")
b_fc3 = tf.Variable(tf.random_normal([NumClasses],stddev = .011),name = "b3")


x_flat = tf.reshape(x, [-1, dime*dime])

FirstPre = tf.matmul(x_flat,W_fc1) + b_fc1
FirstPost = tf.nn.relu(FirstPre)

SecondPre = tf.matmul(FirstPost,W_fc2) + b_fc2
SecondPost = tf.nn.relu(SecondPre)

ThirdPre = tf.matmul(SecondPost,W_fc3) + b_fc3
ThirdPost = tf.nn.relu(ThirdPre)

y_out = tf.nn.softmax(ThirdPost)
#W_conv1 = weight_variable([5, 5, 1, 32])
#b_conv1 = bias_variable([32])

#x_image = tf.reshape(x, [-1,dime,dime,1])
#x_image = tf.to_float(x_image, name='ToFloat')


#pre_h_conv1 = conv2d(x_image, W_conv1) + b_conv1
#h_conv1 = tf.nn.relu(pre_h_conv1)
#h_pool1 = max_pool_2x2(h_conv1)
#print h_conv1

#W_conv2 = weight_variable([5, 5, 32, 64])
#b_conv2 = bias_variable([64])

#pre_h_conv2 = conv2d(h_conv1, W_conv2) + b_conv2
#h_conv2 = tf.nn.relu(pre_h_conv2)
#h_pool2 = max_pool_2x2(h_conv2)
#h_pool2 = max_pool_2x2(h_pool2)

#W_fc1 = weight_variable([dime * dime * (64/16), 1024])
#b_fc1 = bias_variable([1024])

#print h_conv2
#h_pool2_flat = tf.reshape(h_pool2, [-1, dime*dime*(64/16)])

#print 'hola'
#print h_pool2_flat
#pre_h_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
#h_fc1 = tf.nn.relu(pre_h_fc1)


#W_fc2 = weight_variable([1024, NumClasses])
#b_fc2 = bias_variable([NumClasses])

#bfr_scr = tf.matmul(h_fc1, W_fc2) + b_fc2
#y_conv=tf.nn.softmax(bfr_scr)

print 'im here 2!'


cross_entropy = -tf.reduce_sum(y_*tf.log(y_out))
train_step = tf.train.AdamOptimizer(2e-5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_out,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
init = tf.initialize_all_variables()

#ngrad = tf.gradients(h_conv1[0,1,2,3], x)[0]
#ngradi = tf.gradients(h_conv2[0,1,2,3],x)[0]
#lossGrad = tf.gradients(cross_entropy, W_fc2)[0]
#lossGrad2 = tf.gradients(cross_entropy,W_fc2,[-2.0] )[0]

sess = tf.Session()
sess.run(init)

print 'im here3'
import timeit

start = timeit.timeit()

for j in range(0,1):
    

    for k in range(8000):
        q = np.random.choice(d1, 1, replace=False)
        #print q
    #batch = mnist.train.next_batch(50)
        Xtrain = Xq[q,:]
        Ytrain = Temp[q,:]
        pass
        #train_accuracy = accuracy.eval(session=sess,feed_dict={x:Xtrain, y_: Ytrain})
        #print "step %d, training accuracy %g"%(k, train_accuracy)
        #_,W_val,loss_val = train_step.run(session=sess,feed_dict={x: Xtrain, y_: Ytrain})
        W1,out=sess.run([W_fc1,y_out],feed_dict={x: Xtrain, y_: Ytrain } )
        #print W1[0][1]
        #print out
        #print 1/0
        #_,loss_val,lgrad,lgrad2,h_conv1d,ngrad1,ngrad2=sess.run([train_step,cross_entropy,lossGrad,lossGrad2,h_conv1[0,1,2,3],ngrad,ngradi],feed_dict={x: Xtrain, y_: Ytrain } )
        _,loss=sess.run([train_step,cross_entropy],feed_dict={x: Xtrain, y_: Ytrain } )
        print loss
        #print loss_val
        #print lgrad
        #print lgrad2


 


q = np.random.choice(d1, 500, replace=False)
print "greeetings from mihir"
print "test accuracy %g"%accuracy.eval(session=sess,feed_dict={
    x: Xq[q,:], y_: Temp[q,:]})
w_conv1 = sess.run(W_conv1)
w_conv2 = sess.run(W_conv2)


import matplotlib.pyplot as plt







