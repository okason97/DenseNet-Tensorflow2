import os
import tensorflow as tf

def train(model=None, epochs=10, batch_size=32, format_paths=True,
          train_gen=None, train_len=None, val_gen=None, val_len=None,
          train_loss=None, train_accuracy=None, val_loss=None, val_accuracy=None,
          train_step=None, test_step=None, checkpoint_path=None, max_patience=25,
          train_summary_writer=None, val_summary_writer=None, csv_output_file=None):

    min_loss = 100
    min_loss_acc = 0
    patience = 0

    results = 'epoch,loss,accuracy,val_loss,val_accuracy\n'

    for epoch in range(epochs):
        batches = 0
        for images, labels in train_gen:
            train_step(images, labels)
            batches += 1
            if batches >= train_len / batch_size:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break

        batches = 0
        for val_images, val_labels in val_gen:
            test_step(val_images, val_labels)
            batches += 1
            if batches >= val_len / batch_size:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break

        results += '{},{},{},{},{}\n'.format(
            epoch,
            train_loss.result(),
            train_accuracy.result()*100,
            val_loss.result(),
            val_accuracy.result()*100)
        print ('Epoch: {}, Train Loss: {}, Train Acc:{}, Test Loss: {}, Test Acc: {}'.format(
            epoch,
            train_loss.result(),
            train_accuracy.result()*100,
            val_loss.result(),
            val_accuracy.result()*100))

        if (val_loss.result() < min_loss):
            min_loss = val_loss.result()
            min_loss_acc = val_accuracy.result()
            patience = 0
            # serialize weights to HDF5
            if format_paths:
                checkpoint_path = checkpoint_path.format(epoch=epoch, val_loss=min_loss, val_accuracy=min_loss_acc)
            model.save_weights(checkpoint_path)
        else:
            patience += 1

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
            train_loss.reset_states()           
            train_accuracy.reset_states()           

        with val_summary_writer.as_default():
            tf.summary.scalar('loss', val_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', val_accuracy.result(), step=epoch)
            val_loss.reset_states()           
            val_accuracy.reset_states()
            
        if patience >= max_patience:
            break

    file = open(csv_output_file, 'w') 
    file.write(results) 
    file.close()

    return min_loss, min_loss_acc