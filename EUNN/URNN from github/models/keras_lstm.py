import keras

class KerasLSTM():

    def __init__(self, input_dim, output_dim, hidden_size, timesteps):
        self.hidden_size = hidden_size
        self.timesteps = timesteps
        self.input_dim = input_dim

        self.input = keras.layers.Input(
            (self.timesteps, self.input_dim))
        self.hidden = keras.layers.recurrent.SimpleRNN(
            self.hidden_size,
            activation='tanh',
            recurrent_activation='sigmoid',
            return_sequences = False)(self.input)
        self.output = keras.layers.Dense(output_dim)(self.hidden)

        self.model = keras.models.Model(self.input, self.output)

    def train(self, dataset, batch_size, epochs):
        X_train, Y_train, X_test, Y_test = dataset.load_data()

        self.model.compile(loss='mse', optimizer=keras.optimizers.RMSprop(lr = 0.001))
        tensorboard = keras.callbacks.TensorBoard(log_dir='./log', histogram_freq=0.01, write_graph=True, write_images=False, write_grads = True,batch_size = batch_size)

        self.model.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs,
            validation_data = (X_test, Y_test), callbacks=[tensorboard])

        score = self.model.evaluate(X_test, Y_test)
        print("Accuracy:", 1 - score)