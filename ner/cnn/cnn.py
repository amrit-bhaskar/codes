"""
This is a CNN for relation classification within a sentence. The architecture is based on:

Daojian Zeng, Kang Liu, Siwei Lai, Guangyou Zhou and Jun Zhao, 2014, Relation Classification via Convolutional Deep Neural Network

Performance (without hyperparameter optimization):
Accuracy: 0.7943
Macro-Averaged F1 (without Other relation):  0.7612

Performance Zeng et al.
Macro-Averaged F1 (without Other relation): 0.789


Code was tested with:
- Python 2.7 & Python 3.6
- Theano 0.9.0 & TensorFlow 1.2.1
- Keras 2.0.5
"""
from __future__ import print_function
import numpy as np
import os
np.random.seed(1337)  # for reproducibility
import gzip
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

import sys
if (sys.version_info > (3, 0)):
    import pickle as pkl
else: #Python 2.7 imports
    import cPickle as pkl

import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, concatenate,Dot,Reshape
from keras.layers import Embedding
from keras.layers import Convolution1D, MaxPooling1D, GlobalMaxPooling1D
from keras.regularizers import Regularizer
from keras.preprocessing import sequence
from keras.utils import to_categorical as tc


batch_size = 32
nb_filter = 128
filter_length1 = 2
filter_length2 = 3
filter_length3 = 4
filter_length4 = 5
filter_length5 = 7
filter_length6 = 9

hidden_dims = 100
nb_epoch = 10

position_dims = 100

print("Load dataset")
'''
f = gzip.open('pkl/sem-relations.pkl.gz', 'rb')
data = pkl.load(f)
f.close()

embeddings = data['wordEmbeddings']
yTrain, sentenceTrain, positionTrain1, positionTrain2 = data['train_set']
yTest, sentenceTest, positionTest1, positionTest2  = data['test_set']
'''

embeddings=np.load('ADE-corpus/wordEmbeddings.npy')
yTrain=np.load('ADE-corpus/train_set1.npy')
temp=np.load('ADE-corpus/train_set2.npy')
sentenceTrain, positionTrain1, positionTrain2=temp[0],temp[1],temp[2]

yTest=np.load('ADE-corpus/test_set1.npy')
temp=np.load('ADE-corpus/test_set2.npy')
sentenceTest, positionTest1, positionTest2=temp[0],temp[1],temp[2]



max_position = max(np.max(positionTrain1), np.max(positionTrain2))+1

n_out = max(yTrain)+1
#train_y_cat = np_utils.to_categorical(yTrain, n_out)
max_sentence_len = sentenceTrain.shape[1]

print("sentenceTrain: ", sentenceTrain.shape)
print("positionTrain1: ", positionTrain1.shape)
print("yTrain: ", yTrain.shape)

#yTrain=tc(yTrain,num_classes=2)


print("sentenceTest: ", sentenceTest.shape)
print("positionTest1: ", positionTest1.shape)
print("yTest: ", yTest.shape)



print("Embeddings: ",embeddings.shape)

words_input = Input(shape=(max_sentence_len,), dtype='int32', name='words_input')
words = Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], trainable=True)(words_input)

distance1_input = Input(shape=(max_sentence_len,), dtype='int32', name='distance1_input')
distance1 = Embedding(max_position, position_dims,trainable=True)(distance1_input)

distance2_input = Input(shape=(max_sentence_len,), dtype='int32', name='distance2_input')
distance2 = Embedding(max_position, position_dims,trainable=True)(distance2_input)




output = concatenate([words, distance1, distance2])


output1 = Convolution1D(filters=nb_filter,
                        kernel_size=filter_length1,
                        padding='same',
                        activation='relu',
                        strides=1)(output)

output2 = Convolution1D(filters=nb_filter,
                        kernel_size=filter_length2,
                        padding='same',
                        activation='tanh',
                        strides=1)(output)

output3 = Convolution1D(filters=nb_filter,
                        kernel_size=filter_length3,
                        padding='same',
                        activation='tanh',
                        strides=1)(output)

output4 = Convolution1D(filters=nb_filter,
                        kernel_size=filter_length4,
                        padding='same',
                        activation='tanh',
                        strides=1)(output)

# output5 = Convolution1D(filters=nb_filter,
#                         kernel_size=filter_length5,
#                         padding='same',
#                         activation='tanh',
#                         strides=1)(output)
#
# output6 = Convolution1D(filters=nb_filter,
#                         kernel_size=filter_length6,
#                         padding='same',
#                         activation='tanh',
#                         strides=1)(output)


print(output1.shape)


#output1=concatenate([output1,output3])

dens1=Dense(1,activation='tanh')(output1)
#dens2=Dense(1,activation='tanh')(output2)
#dens3=Dense(1,activation='tanh')(output3)
#dens4=Dense(1,activation='tanh')(output4)

print(dens1.shape)

act1=Activation('softmax')(dens1)
#act2=Activation('softmax')(dens2)
#act3=Activation('softmax')(dens3)
#act4=Activation('softmax')(dens4)
print(act1.shape)




dot1=Dot(axes=1)([act1,output1])
#dot2=Dot(axes=1)([act2,output2])
#dot3=Dot(axes=1)([act3,output3])
#dot4=Dot(axes=1)([act4,output4])

print(dot1.shape)


output = dot1
# we use standard max over time pooling
print(output.shape)
#output = GlobalMaxPooling1D()(output)

output = Dropout(0.5)(output)

output=Reshape((-1,))(output)

print(output.shape)
output = Dense(256, activation='relu')(output)



output = Dense(100, activation='relu')(output)

output = Dense(n_out,activation='softmax')(output)
print(output.shape)

model = Model(inputs=[words_input, distance1_input, distance2_input], outputs=[output])
model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam', metrics=['accuracy'])
model.summary()

print("Start training")

max_prec, max_rec, max_acc, max_f1 = 0,0,0,0

def getPrecision(pred_test, yTest, targetLabel):
    #Precision for non-vague
    targetLabelCount = 0
    correctTargetLabelCount = 0
    
    for idx in range(len(pred_test)):
        if pred_test[idx] == targetLabel:
            targetLabelCount += 1
            
            if pred_test[idx] == yTest[idx]:
                correctTargetLabelCount += 1
    
    if correctTargetLabelCount == 0:
        return 0
    
    return float(correctTargetLabelCount) / targetLabelCount

def predict_classes(prediction):
     return prediction.argmax(axis=-1)

for epoch in range(nb_epoch):
    print("\nEpoch\t"+str(epoch+1)+"\n")
    #model.fit([sentenceTrain, positionTrain1, positionTrain2], yTrain, batch_size=batch_size, verbose=True,epochs=1)
    #amr_temp=model.predict([sentenceTest, positionTest1, positionTest2], verbose=False)
    model.fit([sentenceTrain, positionTrain1, positionTrain2], yTrain, batch_size=batch_size, verbose=True,epochs=1)
    amr_temp = model.predict([sentenceTest, positionTest1, positionTest2], verbose=False)
    pred_test = predict_classes(amr_temp)
    #pred_test = predict_classes(model.predict([sentenceTest, positionTest1, positionTest2], verbose=True))
    #model.save("nre.h5")

    predicted = pred_test.tolist()
    actual = yTest.tolist()

    probabilities=amr_temp.tolist()

    dctLabels = np.mean(pred_test)
    totalDCTLabels = np.sum(yTest)
   
    acc =  np.sum(pred_test == yTest) / float(len(yTest))
    max_acc = max(max_acc, acc)
    print("Accuracy: %.4f (max: %.4f)" % (acc, max_acc))

    f1Sum = 0
    f1Count = 0
    for targetLabel in range(0, max(yTest)+1):
        prec = getPrecision(pred_test, yTest, targetLabel)
        recall = getPrecision(yTest, pred_test, targetLabel)
        f1 = 0 if (prec+recall) == 0 else 2*prec*recall/(prec+recall)
        f1Sum += f1
        f1Count +=1


    # if(epoch==nb_epoch-1):
    #     f2=open("result_table.txt",'w')
    #
    #     f2.write("Actual\tPredicted\tclass1\tclass2\tclass3\n")
    #     for i in range(len(yTest)):
    #         f2.write(str(actual[i])+"\t"+str(predicted[i])+"\t"+str(round(probabilities[i][0],4))+"\t"+str(round(probabilities[i][1],4))+"\t"+str(round(probabilities[i][2],4))+"\n")
    #     f2.close()
    #     from sklearn.metrics import confusion_matrix
    #
    #     conf = confusion_matrix(yTest, pred_test)
    #     with open("conf_matrix", 'w') as f3:
    #         f3.write(np.array2string(conf, separator=', '))
    #     f3.close()




    macroF1 = f1Sum / float(f1Count)
    max_f1 = max(max_f1, macroF1)
    print("Non-other Macro-Averaged F1: %.4f (max: %.4f)\n" % (macroF1, max_f1))
    model.save("ADE-corpus/nre"+str(epoch+1)+".h5")
