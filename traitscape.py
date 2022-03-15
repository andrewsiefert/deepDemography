import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
import tensorflow as tf
from tensorflow import keras


class TraitScape:
    """Deep learning model of tree growth using FIA data"""

    def __init__(self, X, y):
        self.X = X
        self.y = y

        self.clim_ind = range(0, 21)
        self.soil_ind = range(21, 30)
        self.trait_ind = range(30, 58)
        self.dia_ind = 58

        self.clim_features = self.X.columns[self.clim_ind].tolist()
        self.soil_features = self.X.columns[self.soil_ind]
        self.trait_features = self.X.columns[self.trait_ind].tolist()
        self.dia_features = self.X.columns[self.dia_ind]

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None

        self.model = None
        self.history = None

    def split_data(self, test_size=0.1):
        """Split data into training and test sets"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size,
                                                                                random_state=42)
    def null_mse(self):
        """Calculate baseline (null) mean squared error on test set"""
        return np.mean((self.y_test - np.mean(self.y_test))**2)

    def create_transformer(self):
        """Fit power transformer to training set and apply to training and test sets"""
        self.transformer = PowerTransformer()
        self.transformer.fit(self.X_train)

        self.X_train_scaled = self.transformer.transform(self.X_train)
        self.X_test_scaled = self.transformer.transform(self.X_test)

    def create_model(self, e_hidden, e_units, g_hidden, g_units, dropout, lr, epochs):
        """Set up neural network architecture and train model

        Parameters
        e_hidden: number of hidden layers in climate, soil, and trait submodels
        e_units: number of units per hidden layer in climate, soil, and trait submodels
        g_hidden: number of hidden layers in growth submodel
        g_units: number of units per hidden layer in growth submodel
        dropout: dropout rate
        lr: learning rate
        epochs: number of training epochs
        """

        # input layers
        clim_input = keras.Input(shape=(21,), name='clim_input')
        soil_input = keras.Input(shape=(9,), name='soil_input')
        trait_input = keras.Input(shape=(28,), name='trait_input')
        dia_input = keras.Input(shape=(1,), name='dia_input')

        # climate embedding
        clim_output = keras.layers.Dense(e_units, activation='relu', name='clim_hidden1')(clim_input)
        if dropout > 0:
            clim_output = keras.layers.Dropout(dropout)(clim_output)
        for i in range(e_hidden - 1):
            clim_output = keras.layers.Dense(e_units, activation='relu')(clim_output)
            if dropout > 0:
                clim_output = keras.layers.Dropout(dropout)(clim_output)
        clim_output = keras.layers.Dense(2, activation='sigmoid', name='clim_output')(clim_output)

        # soil embedding
        soil_output = keras.layers.Dense(e_units, activation='relu', name='soil_hidden1')(soil_input)
        if dropout > 0:
            soil_output = keras.layers.Dropout(dropout)(soil_output)
        for i in range(e_hidden - 1):
            soil_output = keras.layers.Dense(e_units, activation='relu')(soil_output)
            if dropout > 0:
                soil_output = keras.layers.Dropout(dropout)(soil_output)
        soil_output = keras.layers.Dense(2, activation='sigmoid', name='soil_output')(soil_output)

        # trait embedding
        trait_output = keras.layers.Dense(e_units, activation='relu', name='trait_hidden1')(trait_input)
        if dropout > 0:
            trait_output = keras.layers.Dropout(dropout)(trait_output)
        for i in range(e_hidden - 1):
            trait_output = keras.layers.Dense(e_units, activation='relu')(trait_output)
            if dropout > 0:
                trait_output = keras.layers.Dropout(dropout)(trait_output)
        trait_output = keras.layers.Dense(2, activation='sigmoid', name='trait_output')(trait_output)

        # combine embeddings and predict growth
        output = keras.layers.concatenate([clim_output, soil_output, trait_output, dia_input], name='flatten')
        for i in range(0, g_hidden):
            output = keras.layers.Dense(g_units, activation='relu')(output)
            if dropout > 0:
                output = keras.layers.Dropout(dropout)(output)
        output = keras.layers.Dense(1, name='growth_output')(output)

        self.model = keras.Model(
            inputs=[clim_input, soil_input, trait_input, dia_input],
            outputs=[output]
        )

        clim_train = self.X_train_scaled[:, self.clim_ind]
        soil_train = self.X_train_scaled[:, self.soil_ind]
        trait_train = self.X_train_scaled[:, self.trait_ind]
        dia_train = self.X_train_scaled[:, self.dia_ind].reshape(-1, 1)

        # set up early stopping
        es = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto',
            baseline=None, restore_best_weights=True
        )

        # compile model
        self.model.compile(
            loss='mse',
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            metrics=['mse']
        )

        # fit model
        self.history = self.model.fit([clim_train, soil_train, trait_train, dia_train],
                            self.y_train,
                            batch_size=64,
                            epochs=epochs,
                            validation_split=0.15,
                            callbacks=[es])

        # trait submodel
        self.trait_model = keras.Model(
            inputs=trait_input,
            outputs=trait_output
        )

        # climate submodel
        self.clim_model = keras.Model(
            inputs=clim_input,
            outputs=clim_output
        )

        # soil submodel
        self.soil_model = keras.Model(
            inputs=soil_input,
            outputs=soil_output
        )

        clim_embedding = keras.Input(shape=(2,), name='clim_embedding')
        soil_embedding = keras.Input(shape=(2,), name='soil_embedding')
        trait_embedding = keras.Input(shape=(2,), name='trait_embedding')

        # growth submodel
        growth = keras.layers.concatenate([clim_embedding, soil_embedding, trait_embedding, dia_input], name='flatten')
        for i in range(0, g_hidden):
            growth = output = keras.layers.Dense(g_units, activation='relu')(growth)
        growth = keras.layers.Dense(1, name='growth_output')(growth)

        self.growth_model = keras.Model(
            inputs=[clim_embedding, soil_embedding, trait_embedding, dia_input],
            outputs=[growth]
        )

        growth_weights = self.model.get_weights()[-(g_hidden * 2 + 2):]
        self.growth_model.set_weights(growth_weights)

    # get predictions
    def predict(self, X):
        X_scaled = self.transformer.transform(X)

        clim_in = X_scaled[:, self.clim_ind]
        soil_in = X_scaled[:, self.soil_ind]
        trait_in = X_scaled[:, self.trait_ind]
        dia_in = X_scaled[:, self.dia_ind].reshape(-1, 1)

        return self.model.predict([clim_in, soil_in, trait_in, dia_in])

    # evaluate model performance (mean squared error) on test set
    def evaluate(self):
        clim_test = self.X_test_scaled[:, self.clim_ind]
        soil_test = self.X_test_scaled[:, self.soil_ind]
        trait_test = self.X_test_scaled[:, self.trait_ind]
        dia_test = self.X_test_scaled[:, self.dia_ind].reshape(-1, 1)

        test = [clim_test, soil_test, trait_test, dia_test]

        return self.model.evaluate(test, self.y_test)

    # get climate, soil, and trait embeddings
    def get_embeddings(self, X):
        X_scaled = self.transformer.transform(X)

        clim_in = X_scaled[:, self.clim_ind]
        soil_in = X_scaled[:, self.soil_ind]
        trait_in = X_scaled[:, self.trait_ind]

        return [self.clim_model.predict(clim_in), self.soil_model.predict(soil_in), self.trait_model.predict(trait_in)]