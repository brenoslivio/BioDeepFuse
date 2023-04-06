import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate, BatchNormalization, Bidirectional, LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import Model
from itertools import product
import pandas as pd
from sklearn.metrics import classification_report
import numpy as np
import seqdata
import argparse
import warnings
import os

# one-hot encoding + features
# python main.py --train train/ --test test/ --epochs 10 --encoding 0 --feat_extraction 1 --output results.csv
# all encodings without features
# python main.py --train train/ --test test/ --epochs 10 --encoding 4 --feat_extraction 0 --output results.csv
# all encodings + features
# python main.py --train train/ --test test/ --epochs 10 --encoding 4 --feat_extraction 1 --output results.csv

def load_data(train_path, test_path, encoding, feat_extraction):

    train_data, test_data, max_len = [], [], []

    for enc in range(3):
        if enc == encoding or encoding >= 3:
            train, test = seqdata.Seq(train_path, enc), seqdata.Seq(test_path, enc)
            enc_length = seqdata.pad_data(train, test)

            train_data.append(train)
            test_data.append(test)
            max_len.append(enc_length)

    if feat_extraction or encoding == 3:
        print('Extracting features...')
        train_data[0].feature_extraction([1, 2, 3, 4, 5, 6, 7, 8], True)
        test_data[0].feature_extraction([1, 2, 3, 4, 5, 6, 7, 8], False)
        max_len.append(train_data[0].features.shape[1])

    return train_data, test_data, max_len

def base_layers(encoding, max_len):

    if encoding == 0: # One-hot encoding
        input_layer = Input(shape=(max_len, 4))

        x = Conv1D(filters=128, kernel_size=3, activation='relu')(input_layer)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        x = Conv1D(filters=128, kernel_size=3, activation='relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        out = Flatten()(x)
    elif encoding == 1: # Label encoding
        input_layer = Input(shape=(max_len,))

        x = Embedding(5, 32, input_length=max_len)(input_layer)
        x = Conv1D(filters=128, kernel_size=3, activation='relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        x = Conv1D(filters=128, kernel_size=3, activation='relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        out = Flatten()(x)
    elif encoding == 2: # k-mer encoding
        input_layer = Input(shape=(max_len,))

        x = Embedding(4096, 32, input_length=max_len)(input_layer)
        x = Conv1D(filters=128, kernel_size=3, activation='relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        x = Conv1D(filters=128, kernel_size=3, activation='relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        out = Flatten()(x)
    elif encoding == 3: # no encoding
        input_layer = Input(shape=(max_len,))

        out = Flatten()(input_layer)

    return input_layer, out

def create_model(encoding, feat_extraction, num_labels, max_len):

    input_layers, outs = [], []

    for enc in range(3):

        if enc == encoding or encoding == 4:

            if encoding == 4:
                in_layer, x = base_layers(enc, max_len[enc])
            else:
                in_layer, x = base_layers(enc, max_len[0])
            
            input_layers.append(in_layer)
            outs.append(x)

    if encoding == 3 or feat_extraction:
        in_layer, x = base_layers(3, max_len[-1])
        input_layers.append(in_layer)
        outs.append(x)

    if encoding == 4 or feat_extraction:
        outs = Concatenate()(outs)
    else:
        outs = outs[0]

    # Dense layers
    x = Dense(128, activation='relu')(outs)
    x = Dropout(0.5)(x)
    output_layer = Dense(num_labels, activation='softmax')(x)

    model = Model(inputs=input_layers, outputs=output_layer)

    model.compile(loss='categorical_crossentropy', optimizer='adam', 
                metrics= [tf.keras.metrics.Precision(name="Precision")])

    model.summary()

    return model

def train_model(model, encoding, train_data, feat_extraction, epochs):

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)
    ]

    if encoding == 3:
        features = train_data[0].features
    else:
        features = [train.seqs for train in train_data]

        if feat_extraction:
            features.append(train_data[0].features)

    model.fit(features, train_data[0].labels, batch_size=32, epochs=epochs, validation_split=0.1, callbacks=callbacks)

def report_model(model, encoding, test_data, feat_extraction, output_file):

    if encoding == 3:
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

    tf.keras.utils.set_random_seed(0)  # sets seeds for base-python, numpy and tf
    tf.config.experimental.enable_op_determinism()

    parser = argparse.ArgumentParser()
    parser.add_argument('-train', '--train', help='Folder with FASTA training files')
    parser.add_argument('-test', '--test', help='Folder with FASTA testing files')
    parser.add_argument('-epochs', '--epochs', help='Number of epochs to train')
    parser.add_argument('-encoding', '--encoding', default=0, help='Encoding - 0: One-hot encoding, 1: Label encoding 2: k-mer encoding, 3: No encoding (only feature extraction), 4: All')
    parser.add_argument('-feat_extraction', '--feat_extraction', default=0, help='Add biological sequences descriptors (0 = False, 1 = True; Default: False)')
    parser.add_argument('-output', '--output', default=0, help='Output filename for classification report.')

    args = parser.parse_args()

    train_path = str(args.train)
    test_path = str(args.test)
    epochs = int(args.epochs)
    encoding = int(args.encoding)
    feat_extraction = int(args.feat_extraction)
    output_file = str(args.output)

    train_data, test_data, max_len = load_data(train_path, test_path, encoding, feat_extraction)

    num_labels = len(train_data[0].names)

    model = create_model(encoding, feat_extraction, num_labels, max_len)

    train_model(model, encoding, train_data, feat_extraction, epochs)

    report_model(model, encoding, test_data, feat_extraction, output_file)
