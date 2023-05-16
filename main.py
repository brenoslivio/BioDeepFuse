import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LeakyReLU, Concatenate, Activation, BatchNormalization, Bidirectional, LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l1_l2
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from itertools import product
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score
import optuna
import numpy as np
import seqdata
import argparse
import warnings
import os

# one-hot encoding + 1-mer + 4 convolutional layers + relu activation + batch normalization + 0.2 dropout between conv layers + 1 lstm layers + not bidirectional + 0.2 dropout between lstm layers
# running example:
# python main.py --train train/ --test test/ --epochs 10 --encoding 0 --k 1 --feat_extraction 0 --num_convs 4 --activation 0 \
# --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 0 --lstm_dropout 0.2  --output results/test

def load_data(train_path, test_path, encoding, feat_extraction, k):

    train_data, test_data, max_len = [], [], []

    for enc in range(2):
        if enc == encoding or encoding >= 2: # specific encoding or all encodings
            train, test = seqdata.Seq(train_path, enc, k), seqdata.Seq(test_path, enc, k)
            enc_length = seqdata.pad_data(train, test)

            train_data.append(train)
            test_data.append(test)
            max_len.append(enc_length)

    if feat_extraction or encoding == 2:
        print('Extracting features...')
        train_data[0].feature_extraction(feat_extraction, True)
        test_data[0].feature_extraction(feat_extraction, False)
        max_len.append(train_data[0].features.shape[1])

    return train_data, test_data, max_len

def conv_block(x, conv_params):
    
    for _ in range(conv_params['num_convs']):
        x = Conv1D(filters=128, kernel_size=3, padding='same')(x)
        if conv_params['batch_norm']:
            x = BatchNormalization()(x)

        x = Activation(LeakyReLU())(x) if conv_params['activation'] else Activation('relu')(x)

        x = MaxPooling1D(pool_size=2)(x)

        if conv_params['dropout'] > 0:
            x = Dropout(conv_params['dropout'])(x)

    return x

def lstm_block(x, lstm_params):

    for i in range(lstm_params['num_lstm']):
        
        seq = True if lstm_params['num_lstm'] > 1 and i < lstm_params['num_lstm'] - 1 else False

        if lstm_params['bidirectional']:
            x = Bidirectional(LSTM(128, return_sequences=seq))(x)
        else:
            x = LSTM(128, return_sequences=seq)(x)

        if lstm_params['dropout'] > 0:
            x = Dropout(lstm_params['dropout'])(x)

    return x

def base_layers(encoding, max_len, k, conv_params, lstm_params):

    num_combs = 4 ** k

    if encoding == 0: # One-hot encoding
        input_layer = Input(shape=(max_len, num_combs))

        x = conv_block(input_layer, conv_params)

        x = lstm_block(x, lstm_params)

        x = Flatten()(x)

        x = Dense(256, activation='relu')(x)

        out = Dropout(0.5)(x)

    elif encoding == 1: # K-mer embedding
        input_layer = Input(shape=(max_len,))

        x = Embedding(num_combs, 128, input_length=max_len)(input_layer)

        x = conv_block(x, conv_params)

        x = lstm_block(x, lstm_params)

        x = Flatten()(x)

        x = Dense(256, activation='relu')(x)

        out = Dropout(0.5)(x)

    elif encoding == 2: # no encoding
        input_layer = Input(shape=(max_len,))

        x = BatchNormalization(scale=False, center=False)(input_layer) # scaling

        x = Flatten()(x)

        x = Dense(256, activation='relu')(x)

        out = Dropout(0.5)(x)

    return input_layer, out

def create_model(encoding, feat_extraction, num_labels, max_len, k, conv_params, lstm_params):

    input_layers, outs = [], []

    for enc in range(2):

        if enc == encoding or encoding == 3: # specific encoding or all encodings

            if encoding == 3:
                in_layer, x = base_layers(enc, max_len[enc], k, conv_params, lstm_params)
            else:
                in_layer, x = base_layers(enc, max_len[0], k, conv_params, lstm_params)
            
            input_layers.append(in_layer)
            outs.append(x)

    if encoding == 2 or feat_extraction:
        in_layer, x = base_layers(2, max_len[-1], k, conv_params, lstm_params)
        input_layers.append(in_layer)
        outs.append(x)

    if encoding == 3 or (encoding < 2 and feat_extraction):
        outs = Concatenate()(outs)
    else:
        outs = outs[0]

    # Dense layers
    x = Dense(128, activation='relu')(outs)
    x = Dropout(0.5)(x)
    output_layer = Dense(num_labels, activation='softmax')(x)

    model = Model(inputs=input_layers, outputs=output_layer)

    model.compile(loss='categorical_crossentropy', optimizer= tf.keras.optimizers.Adam(learning_rate=1e-4),
                    metrics= [tf.keras.metrics.Precision(name="precision")])

    model.summary()

    return model

def train_model(model, encoding, train_data, feat_extraction, epochs, patience):

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1)
    ]

    if encoding == 2:
        features = train_data[0].features
    else:
        features = [train.seqs for train in train_data]

        if feat_extraction:
            features.append(train_data[0].features)

    model.fit(features, train_data[0].labels, batch_size=32, epochs=epochs, validation_split=0.2, shuffle=True, callbacks=callbacks)

def report_model(model, encoding, test_data, feat_extraction, output_file):

    if encoding == 2:
        features = test_data[0].features
    else:
        features = [test.seqs for test in test_data]

        if feat_extraction:
            features.append(test_data[0].features)

    model_pred = model.predict(features)
    y_pred = np.argmax(model_pred, axis=1)
    y_true = np.argmax(test_data[0].labels, axis=1)

    report = classification_report(y_true, y_pred, target_names=test_data[0].names, output_dict=True)
    
    df_report = pd.DataFrame(report).T

    df_report.to_csv(output_file)

if __name__ == '__main__':
    warnings.filterwarnings(action='ignore', category=FutureWarning)
    warnings.filterwarnings('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    SEED = 0
    tf.keras.utils.set_random_seed(SEED)  # sets seeds for base-python, numpy and tf
    tf.config.experimental.enable_op_determinism()

    parser = argparse.ArgumentParser()
    parser.add_argument('-train', '--train', help='Folder with FASTA training files')
    parser.add_argument('-test', '--test', help='Folder with FASTA testing files')
    parser.add_argument('-epochs', '--epochs', default=10, help='Number of epochs to train')
    parser.add_argument('-patience', '--patience', default=10, help='Epochs to stop training after loss plateau')
    parser.add_argument('-encoding', '--encoding', default=0, help='Encoding - 0: One-hot encoding, 1: K-mer embedding, 2: No encoding (only feature extraction), 3: All encodings (without feature extraction)')
    parser.add_argument('-k', '--k', default=1, help='Length of k-mers')
    parser.add_argument('-feat_extraction', '--feat_extraction', default=[], nargs='+', help='Features to be extracted, e.g., 1 2 3 4 5 6 7 8 9 10. \
                        1 = NAC, 2 = DNC, 3 = TNC, 4 = kGap, 5 = ORF, 6 = Fickett Score, 7 = Graphs, 8 = Shannon Entropy, 9 = Tsallis Entropy, 10 = repDNA')

    # Choose between conventional and deep learning algorithms
    parser.add_argument('-algorithm', '--algorithm', default=2, help='Algorithm - 0: Support Vector Machines (SVM), 1: Extreme Gradient Boosting (XGBoost), 2: Deep Learning')

    # CNN parameters
    parser.add_argument('-num_convs', '--num_convs', default=1, help='Number of convolutional layers')
    parser.add_argument('-activation', '--activation', default=0, help='Activation to use - 0: ReLU, 1: Leaky ReLU; Default: ReLU')
    parser.add_argument('-batch_norm', '--batch_norm', default=0, help='Use Batch Normalization for Convolutional Layers - 0: False, 1: True; Default: False')
    parser.add_argument('-cnn_dropout', '--cnn_dropout', default=0, help='Dropout rate between Convolutional layers - 0 to 1')

    # LSTM parameters
    parser.add_argument('-num_lstm', '--num_lstm', default=1, help='Number of LSTM layers')
    parser.add_argument('-bidirectional', '--bidirectional', default=0, help='Use Bidirectional LSTM - 0: False, 1: True; Default: False')
    parser.add_argument('-lstm_dropout', '--lstm_dropout', default=0, help='Dropout rate between LSTM layers - 0 to 1')

    # Output folder
    parser.add_argument('-output', '--output', default=0, help='Output folder for classification reports.')

    args = parser.parse_args()

    train_path = args.train
    test_path = args.test

    algorithm = int(args.algorithm)
    epochs = int(args.epochs)
    patience = int(args.patience)
    encoding = int(args.encoding)
    k = int(args.k)

    if args.feat_extraction:
        feat_extraction = [int(i) for i in args.feat_extraction]
    else:
        feat_extraction = args.feat_extraction

    output_folder = args.output

    conv_params = {'num_convs': int(args.num_convs), 'activation': int(args.activation), 'batch_norm': int(args.batch_norm) , 'dropout': float(args.cnn_dropout)}

    lstm_params = {'num_lstm': int(args.num_lstm), 'bidirectional': int(args.bidirectional), 'dropout': float(args.lstm_dropout)}

    train_data, test_data, max_len = load_data(train_path, test_path, encoding, feat_extraction, k)

    num_labels = len(train_data[0].names)

    os.makedirs(output_folder, exist_ok=True)

    if algorithm == 2:
        model = create_model(encoding, feat_extraction, num_labels, max_len, k, conv_params, lstm_params)

        tf.keras.utils.plot_model(
            model,
            to_file= f'{output_folder}/model.png',
            show_shapes=False,
            show_dtype=False,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=False,
            dpi=96,
            layer_range=None,
            show_layer_activations=False
        )

        train_model(model, encoding, train_data, feat_extraction, epochs, patience)

        report_model(model, encoding, test_data, feat_extraction, f'{output_folder}/results.csv')
    else:
        X_train, y_train = train_data[0].features, np.argmax(train_data[0].labels, axis=1)
        X_test, y_test = test_data[0].features, np.argmax(test_data[0].labels, axis=1)

        def objective(trial):
            
            if algorithm == 0:
                params = {
                    'C': trial.suggest_loguniform('C', 1e-4, 1e2),
                    'gamma': trial.suggest_loguniform('gamma', 1e-4, 1e2),
                }

                model = make_pipeline(StandardScaler(), SVC(**params, kernel = 'rbf', probability = True, random_state = SEED))
            elif algorithm == 1:
                params = {
                    'max_depth': trial.suggest_int('max_depth', 1, 9),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 1.0),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
                    'subsample': trial.suggest_loguniform('subsample', 0.01, 1.0),
                    'colsample_bytree': trial.suggest_loguniform('colsample_bytree', 0.01, 1.0),
                    'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
                    'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
                    'eval_metric': 'mlogloss',
                    'use_label_encoder': False
                }

                model = make_pipeline(StandardScaler(), XGBClassifier(**params, random_state=SEED))

            scores = cross_val_score(model, X_train, y_train, n_jobs=-1, cv=5, scoring='precision_weighted')
            weighted_precision = scores.mean()

            return weighted_precision

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=100)
        print(study.best_trial)

        if algorithm == 0:
            model = make_pipeline(StandardScaler(), SVC(**study.best_trial.params, kernel = 'rbf', probability = True, random_state = SEED))
        elif algorithm == 1:
            model = make_pipeline(StandardScaler(), XGBClassifier(**study.best_trial.params, random_state=SEED))

        model.fit(X_train, y_train)
        model_pred = model.predict(X_test)

        report = classification_report(y_test, model_pred, target_names=test_data[0].names, output_dict=True)

        df_report = pd.DataFrame(report).T

        print(df_report)

        df_report.to_csv(f'{output_folder}/results.csv')