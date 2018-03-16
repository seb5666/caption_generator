import caption_generator
from keras.callbacks import ModelCheckpoint

def train_model(weight = None, batch_size=1024, epochs = 10):

    cg = caption_generator.CaptionGenerator()
    model = cg.create_model()

    if weight != None:
        model.load_weights(weight)

    counter = 0
    file_name = 'weights-improvement-{epoch:02d}-2.hdf5'
    checkpoint = ModelCheckpoint(file_name, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    model.fit_generator(cg.data_generator(batch_size=batch_size), steps_per_epoch=cg.total_samples/batch_size, epochs=epochs, verbose=1, callbacks=callbacks_list)
    try:
        model.save('Models/WholeModel_2.h5', overwrite=True)
        model.save_weights('Models/Weights_2.h5',overwrite=True)
    except:
        print("Error in saving model.")
    print("Training complete...\n")

if __name__ == '__main__':
    train_model(epochs=50, batch_size=1024)
