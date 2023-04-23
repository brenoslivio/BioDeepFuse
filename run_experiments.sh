# One-hot encoding

## k = 1

### CNN 

python main.py --train train/ --test test/ --epochs 50 --patience 10 --encoding 0 --k 1 --num_convs 2 --activation 0 --batch_norm 0 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/1_enc0_cnn_2conv_k1
python main.py --train train/ --test test/ --epochs 50 --patience 10 --encoding 0 --k 1 --feat_extraction 1 2 3 4 5 6 --num_convs 2 --activation 0 --batch_norm 0 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/2_enc0_cnn_2conv_k1_biological
python main.py --train train/ --test test/ --epochs 50 --patience 10 --encoding 0 --k 1 --feat_extraction 7 8 9 --num_convs 2 --activation 0 --batch_norm 0 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/3_enc0_cnn_2conv_k1_math
python main.py --train train/ --test test/ --epochs 50 --patience 10 --encoding 0 --k 1 --feat_extraction 10 --num_convs 2 --activation 0 --batch_norm 0 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/4_enc0_cnn_2conv_k1_repDNA
python main.py --train train/ --test test/ --epochs 50 --patience 10 --encoding 0 --k 1 --feat_extraction 1 2 3 4 5 6 7 8 9 --num_convs 2 --activation 0 --batch_norm 0 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/5_enc0_cnn_2conv_k1_biomath
python main.py --train train/ --test test/ --epochs 50 --patience 10 --encoding 0 --k 1 --feat_extraction 1 2 3 4 5 6 7 8 9 10 --num_convs 2 --activation 0 --batch_norm 0 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/6_enc0_cnn_2conv_k1_all

### LSTM

### biLSTM

### CNN-LSTM

### CNN-biLSTM
