# -- Conventional --

python main.py --algorithm 0 --train train/ --test test/ --feat_extraction 1 2 3 4 5 6 --output results/svm
python main.py --algorithm 1 --train train/ --test test/ --feat_extraction 1 2 3 4 5 6 --features_exist 1 --output results/xgboost

# -- Deep Learning --

## -- CNN -- 

### k = 1

#### Enc I - One-hot encoding

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 1 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_1conv_k1
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 1 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_2conv_k1
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 1 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_3conv_k1
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 1 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_4conv_k1

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 1 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_1conv_k1_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 1 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_2conv_k1_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 1 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_3conv_k1_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 1 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_4conv_k1_concat1_bio

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 2 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_1conv_k1_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 2 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_2conv_k1_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 2 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_3conv_k1_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 2 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_4conv_k1_concat2_bio

#### Enc II - k-mer embedding

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --k 1 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc1_cnn_1conv_k1
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --k 1 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc1_cnn_2conv_k1
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --k 1 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc1_cnn_3conv_k1
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --k 1 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc1_cnn_4conv_k1

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 1 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc1_cnn_1conv_k1_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 1 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc1_cnn_2conv_k1_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 1 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc1_cnn_3conv_k1_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 1 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc1_cnn_4conv_k1_concat1_bio

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 2 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc1_cnn_1conv_k1_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 2 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc1_cnn_2conv_k1_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 2 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc1_cnn_3conv_k1_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 2 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc1_cnn_4conv_k1_concat2_bio

#### Enc I + Enc II - both encodings

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --k 1 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc2_cnn_1conv_k1
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --k 1 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc2_cnn_2conv_k1
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --k 1 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc2_cnn_3conv_k1
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --k 1 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc2_cnn_4conv_k1

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 1 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc2_cnn_1conv_k1_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 1 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc2_cnn_2conv_k1_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 1 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc2_cnn_3conv_k1_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 1 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc2_cnn_4conv_k1_concat1_bio

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 2 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc2_cnn_1conv_k1_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 2 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc2_cnn_2conv_k1_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 2 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc2_cnn_3conv_k1_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 2 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc2_cnn_4conv_k1_concat2_bio

### k = 2

#### Enc I - One-hot encoding

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 2 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_1conv_k2
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 2 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_2conv_k2
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 2 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_3conv_k2
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 2 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_4conv_k2

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 1 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_1conv_k2_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 1 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_2conv_k2_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 1 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_3conv_k2_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 1 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_4conv_k2_concat1_bio

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 2 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_1conv_k2_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 2 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_2conv_k2_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 2 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_3conv_k2_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 2 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_4conv_k2_concat2_bio

#### Enc II - k-mer embedding

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --k 2 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc1_cnn_1conv_k2
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --k 2 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc1_cnn_2conv_k2
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --k 2 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc1_cnn_3conv_k2
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --k 2 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc1_cnn_4conv_k2

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 1 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc1_cnn_1conv_k2_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 1 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc1_cnn_2conv_k2_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 1 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc1_cnn_3conv_k2_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 1 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc1_cnn_4conv_k2_concat1_bio

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 2 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc1_cnn_1conv_k2_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 2 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc1_cnn_2conv_k2_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 2 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc1_cnn_3conv_k2_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 2 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc1_cnn_4conv_k2_concat2_bio

#### Enc I + Enc II - both encodings

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --k 2 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc2_cnn_1conv_k2
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --k 2 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc2_cnn_2conv_k2
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --k 2 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc2_cnn_3conv_k2
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --k 2 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc2_cnn_4conv_k2

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 1 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc2_cnn_1conv_k2_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 1 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc2_cnn_2conv_k2_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 1 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc2_cnn_3conv_k2_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 1 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc2_cnn_4conv_k2_concat1_bio

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 2 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc2_cnn_1conv_k2_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 2 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc2_cnn_2conv_k2_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 2 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc2_cnn_3conv_k2_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 2 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc2_cnn_4conv_k2_concat2_bio

### k = 3

#### Enc I - One-hot encoding

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 3 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_1conv_k3
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 3 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_2conv_k3
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 3 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_3conv_k3
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 3 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_4conv_k3

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 1 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_1conv_k3_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 1 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_2conv_k3_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 1 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_3conv_k3_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 1 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_4conv_k3_concat1_bio

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 2 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_1conv_k3_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 2 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_2conv_k3_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 2 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_3conv_k3_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 2 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_4conv_k3_concat2_bio

#### Enc II - k-mer embedding

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --k 3 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc1_cnn_1conv_k3
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --k 3 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc1_cnn_2conv_k3
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --k 3 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc1_cnn_3conv_k3
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --k 3 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc1_cnn_4conv_k3

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 1 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc1_cnn_1conv_k3_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 1 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc1_cnn_2conv_k3_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 1 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc1_cnn_3conv_k3_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 1 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc1_cnn_4conv_k3_concat1_bio

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 2 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc1_cnn_1conv_k3_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 2 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc1_cnn_2conv_k3_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 2 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc1_cnn_3conv_k3_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 2 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc1_cnn_4conv_k3_concat2_bio

#### Enc I + Enc II - both encodings

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --k 3 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc2_cnn_1conv_k3
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --k 3 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc2_cnn_2conv_k3
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --k 3 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc2_cnn_3conv_k3
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --k 3 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc2_cnn_4conv_k3

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 1 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc2_cnn_1conv_k3_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 1 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc2_cnn_2conv_k3_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 1 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc2_cnn_3conv_k3_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 1 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc2_cnn_4conv_k3_concat1_bio

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 2 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc2_cnn_1conv_k3_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 2 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc2_cnn_2conv_k3_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 2 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc2_cnn_3conv_k3_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 2 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc2_cnn_4conv_k3_concat2_bio

## -- CNN-BiLSTM -- 

### k = 1

#### Enc I - One-hot encoding

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 1 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc0_cnn_bilstm_1conv_k1
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 1 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc0_cnn_bilstm_2conv_k1
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 1 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc0_cnn_bilstm_3conv_k1
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 1 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc0_cnn_bilstm_4conv_k1

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 1 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc0_cnn_bilstm_1conv_k1_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 1 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc0_cnn_bilstm_2conv_k1_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 1 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc0_cnn_bilstm_3conv_k1_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 1 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc0_cnn_bilstm_4conv_k1_concat1_bio

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 2 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc0_cnn_bilstm_1conv_k1_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 2 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc0_cnn_bilstm_2conv_k1_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 2 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc0_cnn_bilstm_3conv_k1_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 2 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc0_cnn_bilstm_4conv_k1_concat2_bio

#### Enc II - k-mer embedding

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --k 1 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc1_cnn_bilstm_1conv_k1
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --k 1 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc1_cnn_bilstm_2conv_k1
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --k 1 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc1_cnn_bilstm_3conv_k1
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --k 1 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc1_cnn_bilstm_4conv_k1

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 1 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc1_cnn_bilstm_1conv_k1_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 1 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc1_cnn_bilstm_2conv_k1_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 1 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc1_cnn_bilstm_3conv_k1_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 1 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc1_cnn_bilstm_4conv_k1_concat1_bio

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 2 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc1_cnn_bilstm_1conv_k1_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 2 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc1_cnn_bilstm_2conv_k1_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 2 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc1_cnn_bilstm_3conv_k1_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 2 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc1_cnn_bilstm_4conv_k1_concat2_bio

#### Enc I + Enc II - both encodings

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --k 1 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc2_cnn_bilstm_1conv_k1
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --k 1 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc2_cnn_bilstm_2conv_k1
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --k 1 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc2_cnn_bilstm_3conv_k1
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --k 1 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc2_cnn_bilstm_4conv_k1

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 1 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc2_cnn_bilstm_1conv_k1_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 1 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc2_cnn_bilstm_2conv_k1_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 1 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc2_cnn_bilstm_3conv_k1_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 1 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc2_cnn_bilstm_4conv_k1_concat1_bio

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 2 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc2_cnn_bilstm_1conv_k1_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 2 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc2_cnn_bilstm_2conv_k1_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 2 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc2_cnn_bilstm_3conv_k1_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 2 --k 1 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc2_cnn_bilstm_4conv_k1_concat2_bio

### k = 2

#### Enc I - One-hot encoding

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 2 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc0_cnn_bilstm_1conv_k2
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 2 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc0_cnn_bilstm_2conv_k2
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 2 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc0_cnn_bilstm_3conv_k2
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 2 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc0_cnn_bilstm_4conv_k2

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 1 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc0_cnn_bilstm_1conv_k2_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 1 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc0_cnn_bilstm_2conv_k2_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 1 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc0_cnn_bilstm_3conv_k2_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 1 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc0_cnn_bilstm_4conv_k2_concat1_bio

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 2 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc0_cnn_bilstm_1conv_k2_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 2 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc0_cnn_bilstm_2conv_k2_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 2 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc0_cnn_bilstm_3conv_k2_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 2 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc0_cnn_bilstm_4conv_k2_concat2_bio

#### Enc II - k-mer embedding

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --k 2 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc1_cnn_bilstm_1conv_k2
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --k 2 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc1_cnn_bilstm_2conv_k2
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --k 2 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc1_cnn_bilstm_3conv_k2
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --k 2 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc1_cnn_bilstm_4conv_k2

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 1 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc1_cnn_bilstm_1conv_k2_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 1 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc1_cnn_bilstm_2conv_k2_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 1 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc1_cnn_bilstm_3conv_k2_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 1 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc1_cnn_bilstm_4conv_k2_concat1_bio

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 2 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc1_cnn_bilstm_1conv_k2_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 2 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc1_cnn_bilstm_2conv_k2_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 2 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc1_cnn_bilstm_3conv_k2_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 2 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc1_cnn_bilstm_4conv_k2_concat2_bio

#### Enc I + Enc II - both encodings

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --k 2 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc2_cnn_bilstm_1conv_k2
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --k 2 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc2_cnn_bilstm_2conv_k2
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --k 2 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc2_cnn_bilstm_3conv_k2
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --k 2 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc2_cnn_bilstm_4conv_k2

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 1 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc2_cnn_bilstm_1conv_k2_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 1 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc2_cnn_bilstm_2conv_k2_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 1 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc2_cnn_bilstm_3conv_k2_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 1 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc2_cnn_bilstm_4conv_k2_concat1_bio

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 2 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc2_cnn_bilstm_1conv_k2_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 2 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc2_cnn_bilstm_2conv_k2_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 2 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc2_cnn_bilstm_3conv_k2_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 2 --k 2 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc2_cnn_bilstm_4conv_k2_concat2_bio

### k = 3

#### Enc I - One-hot encoding

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 3 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc0_cnn_bilstm_1conv_k3
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 3 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc0_cnn_bilstm_2conv_k3
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 3 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc0_cnn_bilstm_3conv_k3
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 3 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc0_cnn_bilstm_4conv_k3

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 1 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc0_cnn_bilstm_1conv_k3_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 1 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc0_cnn_bilstm_2conv_k3_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 1 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc0_cnn_bilstm_3conv_k3_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 1 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc0_cnn_bilstm_4conv_k3_concat1_bio

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 2 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc0_cnn_bilstm_1conv_k3_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 2 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc0_cnn_bilstm_2conv_k3_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 2 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc0_cnn_bilstm_3conv_k3_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --concat 2 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc0_cnn_bilstm_4conv_k3_concat2_bio

#### Enc II - k-mer embedding

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --k 3 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc1_cnn_bilstm_1conv_k3
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --k 3 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc1_cnn_bilstm_2conv_k3
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --k 3 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc1_cnn_bilstm_3conv_k3
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --k 3 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc1_cnn_bilstm_4conv_k3

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 1 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc1_cnn_bilstm_1conv_k3_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 1 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc1_cnn_bilstm_2conv_k3_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 1 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc1_cnn_bilstm_3conv_k3_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 1 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc1_cnn_bilstm_4conv_k3_concat1_bio

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 2 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc1_cnn_bilstm_1conv_k3_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 2 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc1_cnn_bilstm_2conv_k3_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 2 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc1_cnn_bilstm_3conv_k3_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 1 --concat 2 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc1_cnn_bilstm_4conv_k3_concat2_bio

#### Enc I + Enc II - both encodings

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --k 3 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc2_cnn_bilstm_1conv_k3
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --k 3 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc2_cnn_bilstm_2conv_k3
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --k 3 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc2_cnn_bilstm_3conv_k3
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --k 3 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc2_cnn_bilstm_4conv_k3

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 1 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc2_cnn_bilstm_1conv_k3_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 1 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc2_cnn_bilstm_2conv_k3_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 1 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc2_cnn_bilstm_3conv_k3_concat1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 1 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc2_cnn_bilstm_4conv_k3_concat1_bio

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 2 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc2_cnn_bilstm_1conv_k3_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 2 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc2_cnn_bilstm_2conv_k3_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 2 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc2_cnn_bilstm_3conv_k3_concat2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 3 --concat 2 --k 3 --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output results/enc2_cnn_bilstm_4conv_k3_concat2_bio
