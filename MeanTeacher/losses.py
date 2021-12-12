import tensorflow as tf 
import numpy as np 



def cal_classification_cost(logits, labels) :
    """ Commputing classification cost , after removing labels -1 of unlabelled data and then calculating
    the binary cross entropy .
    """
    # applicable = tf.not_equal(labels, -1)
    # Change -1s to zeros to make cross-entropy computable
    # labels = tf.where(applicable, labels, tf.zeros_like(labels))
    loss=tf.reduce_sum(tf.keras.losses.categorical_crossentropy(labels, logits))
    # tf.keras.losses.kl_divergence()
    # Retain costs only for labeled
    # per_sample=tf.where(applicable[:,1], per_sample, tf.zeros_like(per_sample))
    # Take mean over all examples, not just labeled examples.
    # loss = tf.math.divide(tf.reduce_mean(tf.reduce_sum( per_sample ) ), np.shape ( per_sample )[0] )
    # print('Classification Cost: ',loss)
    return loss

def cal_overall_cost(x_train,y_train,x_unlabel,student, teacher):
    #TODO: need to include noising technique
    ratio=0.5
    logits = student(x_train)
    classification_cost = cal_classification_cost(logits, y_train)
    tar_student = student(x_unlabel)
    tar_teacher = teacher(x_unlabel)
    
    consistency_cost = cal_consistency_cost( tar_student, tar_teacher)
    # print('consistency_cost : ',consistency_cost)
    return (ratio * classification_cost) + ((1 - ratio) * consistency_cost)

# function for consistency cost
def cal_consistency_cost(student_output, teacher_output) :
    return tf.reduce_sum(tf.losses.mean_squared_error(student_output, teacher_output ))

def ema(student_model, teacher_model, alpha=0.999) :
    # taking weights
    student_weights = student_model.get_weights()
    teacher_weights = teacher_model.get_weights()
    # length must be equal otherwise it will not work
    assert len(student_weights ) == len(teacher_weights ), 'length of student and teachers weights are not equal Please check. \n Student: {}, \n Teacher:{}'.format (
        len(student_weights ), len (teacher_weights ) )
    new_layers = []
    for i, layers in enumerate ( student_weights ) :
        new_layer = alpha * (teacher_weights[i]) + (1 - alpha) * layers
        new_layers.append(new_layer)
    teacher_model.set_weights(new_layers)
    return teacher_model