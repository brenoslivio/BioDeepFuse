# -- Conventional --

python main.py --algorithm 0 --train train/ --test test/ --feat_extraction 1 2 3 4 5 6 --output results/svm
python main.py --algorithm 1 --train train/ --test test/ --feat_extraction 1 2 3 4 5 6 --output results/svm

# -- Deep Learning --

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 2 --k 1 --feat_extraction 1 2 3 4 5 6 --num_convs 2 --activation 0 --batch_norm 0 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/bio

# One-hot encoding

## k = 1

### CNN 

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 1 --feat_extraction 1 2 3 4 5 6 --num_convs 1 --activation 0 --batch_norm 0 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_1conv_k1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 1 --feat_extraction 1 2 3 4 5 6 --num_convs 2 --activation 0 --batch_norm 0 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_2conv_k1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 1 --feat_extraction 1 2 3 4 5 6 --num_convs 3 --activation 0 --batch_norm 0 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_3conv_k1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 1 --feat_extraction 1 2 3 4 5 6 --num_convs 4 --activation 0 --batch_norm 0 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_4conv_k1_bio

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 1 --feat_extraction 1 2 3 4 5 6 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_1conv_norm_k1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 1 --feat_extraction 1 2 3 4 5 6 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_2conv_norm_k1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 1 --feat_extraction 1 2 3 4 5 6 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_3conv_norm_k1_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 1 --feat_extraction 1 2 3 4 5 6 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_4conv_norm_k1_bio

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 2 --feat_extraction 1 2 3 4 5 6 --num_convs 1 --activation 0 --batch_norm 0 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_1conv_k2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 2 --feat_extraction 1 2 3 4 5 6 --num_convs 2 --activation 0 --batch_norm 0 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_2conv_k2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 2 --feat_extraction 1 2 3 4 5 6 --num_convs 3 --activation 0 --batch_norm 0 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_3conv_k2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 2 --feat_extraction 1 2 3 4 5 6 --num_convs 4 --activation 0 --batch_norm 0 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_4conv_k2_bio

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 2 --feat_extraction 1 2 3 4 5 6 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_1conv_norm_k2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 2 --feat_extraction 1 2 3 4 5 6 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_2conv_norm_k2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 2 --feat_extraction 1 2 3 4 5 6 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_3conv_norm_k2_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 2 --feat_extraction 1 2 3 4 5 6 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_4conv_norm_k2_bio

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 3 --feat_extraction 1 2 3 4 5 6 --num_convs 1 --activation 0 --batch_norm 0 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_1conv_k3_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 3 --feat_extraction 1 2 3 4 5 6 --num_convs 2 --activation 0 --batch_norm 0 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_2conv_k3_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 3 --feat_extraction 1 2 3 4 5 6 --num_convs 3 --activation 0 --batch_norm 0 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_3conv_k3_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 3 --feat_extraction 1 2 3 4 5 6 --num_convs 4 --activation 0 --batch_norm 0 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_4conv_k3_bio

python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 3 --feat_extraction 1 2 3 4 5 6 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_1conv_norm_k3_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 3 --feat_extraction 1 2 3 4 5 6 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_2conv_norm_k3_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 3 --feat_extraction 1 2 3 4 5 6 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_3conv_norm_k3_bio
python main.py --train train/ --test test/ --epochs 100 --patience 20 --encoding 0 --k 3 --feat_extraction 1 2 3 4 5 6 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_4conv_norm_k3_bio

### LSTM

### biLSTM

### CNN-LSTM

### CNN-biLSTM
