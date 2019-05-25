import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import util


def main():
    """ Main function """

    # Parameters
    train_data_size = 40000
    batch_size = 64
    epoch = 30
    step_size = train_data_size // batch_size
    display_size = step_size // 2
    learning_rate = 0.001

    train_filename = 'cifar-100-python/train'
    test_filename = 'cifar-100-python/test'
    text_label_filename = 'cifar-100-python/meta'

    # 1. Read training and test data.
    print("Reading training and test data files...")
    train_data = util.unpickle(train_filename)
    test_data = util.unpickle(test_filename)
    text_labels = util.unpickle(text_label_filename)

    raw_train_images, raw_train_label = util.split_images_and_labels(train_data)
    raw_test_images, test_labels = util.split_images_and_labels(test_data)
    fine_label_names, coarse_label_names = util.split_labels(text_labels)

    # Construct fine class to superclass mapping
    superclass_mapping = util.construct_superclass_mapping(
        train_data[b'fine_labels'],
        train_data[b'coarse_labels'])
    fine_labels = util.decode_binary_text(fine_label_names)
    coarse_labels = util.decode_binary_text(coarse_label_names)


    # 2. Pre-process the data
    # Calculate mean image using train data and subtract all images from it
    print("Pre-processing the data...")
    # Cast to float type first TODO and maybe normalize the data
    raw_train_images_float = raw_train_images.astype(float)
    raw_test_images_float = raw_test_images.astype(float)

    mean_image = raw_train_images[:train_data_size].sum(axis=0) / train_data_size
    raw_train_images_float -= mean_image
    raw_test_images_float -= mean_image

    # Format that we want (-1, 3, 32, 32)
    formatted_train_images = util.format_data(raw_train_images_float)
    test_images = util.format_data(raw_test_images_float)

    # 3. Split the train images and labels into train and validation set
    train_images, train_labels, vali_images, vali_labels = util.split_train_and_validation(
        formatted_train_images, raw_train_label, train_data_size)
    vali_super_labels = np.array(train_data[b'coarse_labels'][train_data_size:])
    test_super_labels = np.array(test_data[b'coarse_labels'])

    # 4. Construct the graph
    print("Constructing the graph...")
    # Inputs
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    y_real = tf.placeholder(tf.int64, shape=(None,))

    augmentation = tf.map_fn(tf.image.random_flip_up_down, x)

    # Outputs, cross entropy calculation, and optimizer
    y_predict = util.lenet_5(x)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y_predict, labels=y_real)
    loss_op = tf.reduce_mean(cross_entropy)
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op)

    # 5. Calculate confusion matrix and accuracy (batch and validation)
    # Fine class label prediction: confusion matrix, correct predictions, and accuracy
    labels_predicted = tf.argmax(y_predict, 1)
    confusion_matrix_fine = tf.confusion_matrix(y_real, labels_predicted)
    correct_prediction = tf.equal(labels_predicted, y_real)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    correct_top_5 = tf.nn.in_top_k(y_predict, y_real, 5)
    accuracy_fine_top_5 = tf.reduce_mean(tf.cast(correct_top_5, tf.float32))

    # Super class label prediction: confusion matrix, correct predictions, and accuracy
    mapped_labels = tf.placeholder(tf.int64, shape=(None,))
    confusion_matrix_super = tf.confusion_matrix(y_real, mapped_labels)
    correct_prediction_super = tf.equal(mapped_labels, y_real)
    accuracy_super = tf.reduce_mean(tf.cast(correct_prediction_super, tf.float32))

    top_5_labels = tf.nn.top_k(y_predict, 5)

    # Add results to summaries
    loss_summary = tf.summary.scalar('Loss', loss_op)
    accuracy_summary = tf.summary.scalar('Accuracy: Fine Labels', accuracy)
    accuracy_summary_super = tf.summary.scalar('Accuracy: Super Labels', accuracy_super)

    # 6. Start the training
    print("Start training...")
    total_steps = 0     # count the number of steps it takes throughout training

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter('logs' + '/train', sess.graph)
        vali_writer = tf.summary.FileWriter('logs' + '/validation', sess.graph)

        for epoch_count in range(1, epoch + 1):
            for step_count in range(1, step_size + 1):
                total_steps += 1

                # Input training data
                train_batch, label_batch = util.get_random_batch(train_images, train_labels, batch_size)
                #train_batch = sess.run(augmentation, feed_dict={x: train_batch})

                sess.run(train_op, feed_dict={x: train_batch, y_real: label_batch})

                train_loss_summary, train_acc_summary, loss, acc = sess.run(
                    [loss_summary, accuracy_summary, loss_op, accuracy],
                    feed_dict={x: train_batch, y_real: label_batch})

                train_writer.add_summary(train_loss_summary, total_steps)
                train_writer.add_summary(train_acc_summary, total_steps)

                # Print validation accuracy every so often
                if step_count % display_size == 0:
                    # Validation: fine class label accuracy measurement
                    vali_acc_fine = sess.run(accuracy, feed_dict={x: vali_images, y_real: vali_labels})
                    vali_acc_summary_fine = sess.run(accuracy_summary, feed_dict={x: vali_images, y_real: vali_labels})
                    vali_writer.add_summary(vali_acc_summary_fine, total_steps)
                    print('Validation fine label accuracy: {:.5f}'.format(vali_acc_fine))

                    # Validation: super class label accuracy measurement
                    labels = sess.run(labels_predicted, feed_dict={x: vali_images, y_real: vali_super_labels})
                    new_labels = util.map_class(labels, superclass_mapping)

                    vali_acc_super = sess.run(accuracy_super,
                        feed_dict={x: vali_images, y_real: vali_super_labels, mapped_labels: new_labels})
                    vali_acc_summary_super = sess.run(accuracy_summary_super,
                        feed_dict={x: vali_images, y_real: vali_super_labels, mapped_labels: new_labels})
                    vali_writer.add_summary(vali_acc_summary_super, total_steps)
                    print('Number of steps: {}'.format(total_steps))
                    print('Validation super label accuracy: {:.5f}'.format(vali_acc_super))
            print("{} epochs finished".format(epoch_count))

        # Test: Fine class label accuracy measurement and confusion matrix
        test_acc_fine_top_1 = sess.run(accuracy, feed_dict={x: test_images, y_real: test_labels})
        test_acc_fine_top_5 = sess.run(accuracy_fine_top_5, feed_dict={x: test_images, y_real: test_labels})

        con_matrix_fine = sess.run(confusion_matrix_fine, feed_dict={x: test_images, y_real: test_labels})

        # Display the first ten images
        first_ten_predictions = sess.run(labels_predicted, feed_dict={x: test_images, y_real: test_labels})[:10]
        prediction_text_labels = util.map_text_labels(first_ten_predictions, fine_labels)
        true_text_labels = util.map_text_labels(test_labels[:10], fine_labels)
        print(prediction_text_labels)
        print(true_text_labels)

        first_ten_images = util.format_data(raw_test_images[:10])
        display_image = util.combine_ten_images(first_ten_images)
        plt.imshow(display_image)
        plt.savefig("Result", bbox_inches='tight')

        # Test: Super class label accuracy measurement
        labels = sess.run(labels_predicted, feed_dict={x: test_images, y_real: test_super_labels})
        new_labels = util.map_class(labels, superclass_mapping)
        test_acc_super = sess.run(accuracy_super,
            feed_dict={x: test_images, y_real: test_super_labels, mapped_labels: new_labels})
        con_matrix_super = sess.run(confusion_matrix_super,
            feed_dict={x: test_images, y_real: test_super_labels, mapped_labels: new_labels})

        top_5_labels = sess.run(top_5_labels, feed_dict={x: test_images})[1]
        util.map_all_classes(top_5_labels, superclass_mapping)
        correctness_test_top_5 = util.correct_in_top_5_super(top_5_labels, test_super_labels)
        test_acc_top_5_super = sum(correctness_test_top_5) / len(correctness_test_top_5)

        # Save our result
        output_result(
            [
                'Number of steps taken: {}\n'.format(total_steps),
                'Test fine label accuracy (top 1): {:.5f}\n'.format(test_acc_fine_top_1),
                'Test fine label accuracy (top 5): {:.5f}\n'.format(test_acc_fine_top_5),
                'Test super label accuracy (top 1): {:.5f}\n'.format(test_acc_super),
                'Test super label accuracy (top 5): {:.5f}\n'.format(test_acc_top_5_super)
            ]
        )
        save_confusion_matrix(con_matrix_fine, 'Fine-Label-Confusion-Matrix.txt')
        save_confusion_matrix(con_matrix_super, 'Super-Label-Confusion-Matrix.txt')
        save_heatmap(con_matrix_fine, 'Heatmap-fine-label')
        save_heatmap(con_matrix_super, 'Heatmap-super-label')

        print("Training finished!")


def output_result(texts):
    """ Output our test result statistics """
    with open('result.txt', 'w') as wFile:
        for t in texts:
            wFile.write(t)


def save_confusion_matrix(matrix, name):
    """ Save the confusion matrix as text file """
    np.savetxt(name, matrix, fmt='%i', delimiter=',')


def save_heatmap(map, name):
    """ Save heatmap obtained from the confusion matrix """
    sns.heatmap(map)
    plt.savefig(name, bbox_inches='tight')


if __name__ == '__main__':
    main()
