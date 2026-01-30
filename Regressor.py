from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor(random_state=42,
                   hidden_layer_sizes=(10),
                   max_iter=200,
                   batch_size=1000,
                   activation="relu",
                   validation_fraction=0.2,
                   early_stopping=True) 
