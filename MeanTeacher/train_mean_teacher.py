import tensorflow as tf 
from Utils.utils import data_slices
from Bert.bert_model import bert_model
from MeanTeacher.losses import cal_overall_cost,ema



def train_mean_teacher(x_train, y_train, x_unlabel,pretrained_weights, epochs,batch_size,lr,max_len, alpha):
    # preparing the training dataset
    train_dataset,unlabel_dataset = data_slices(x_train, y_train,x_unlabel,batch_size)
    # declaring optimiser
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    # creating model
    student = bert_model(pretrained_weights,max_len,lr)#call for the model here
    teacher = bert_model(pretrained_weights,max_len,lr) #call for model here
    print('alpha:', alpha)

    # declaring metrics
    train_metrics = tf.keras.metrics.BinaryAccuracy(name='Binary_Accuracy')
    progbar = tf.keras.utils.Progbar(len(train_dataset), stateful_metrics=['Accuracy', 'Overall_Loss'])
    # epochs = args.epochs
    step_counter = 0
    for epoch in range(epochs):
        tf.print(f'\nepoch {epoch + 1}')
        # iterator_noise = iter(noise_dataset)
        for step, ((input_ids,attention_ids, y_batch_train),(input_ids_un,attention_ids_un)) in enumerate(zip(train_dataset,unlabel_dataset)):
            with tf.GradientTape() as tape:
                overall_cost = cal_overall_cost([input_ids,attention_ids], 
                                                y_batch_train,
                                                [input_ids_un,attention_ids_un],
                                                student, 
                                                teacher)

            grads = tape.gradient(overall_cost, student.trainable_weights)
            optimizer.apply_gradients((grad, var) for (grad, var) in zip(grads, student.trainable_weights) if grad is not None)
            # applying student weights to teacher
            step_counter += 1
            teacher = ema(student, teacher, alpha=alpha)
            # calculating training accuracy
            logits_t = teacher([input_ids,attention_ids])
            train_acc = train_metrics(tf.argmax(y_batch_train, 1), tf.argmax(logits_t, 1))
            progbar.update(step, values=[('Accuracy', train_acc), ('Overall_Loss', overall_cost)])

    return student, teacher