
# Only features

python main.py --train train/ --test test/ --epochs 100 --patience 10 --encoding 2 --k 1 --feat_extraction 0 --num_convs 1 --activation 0 --batch_norm 0 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/only_features

# One-hot encoding

## k = 1

python main.py --train train/ --test test/ --epochs 100 --patience 10 --encoding 0 --k 1 --feat_extraction 0 --num_convs 1 --activation 0 --batch_norm 0 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_1conv_k1
python main.py --train train/ --test test/ --epochs 100 --patience 10 --encoding 0 --k 1 --feat_extraction 0 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_norm_1conv_k1
python main.py --train train/ --test test/ --epochs 100 --patience 10 --encoding 0 --k 1 --feat_extraction 1 --num_convs 1 --activation 0 --batch_norm 0 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_1conv_k1_features
python main.py --train train/ --test test/ --epochs 100 --patience 10 --encoding 0 --k 1 --feat_extraction 1 --num_convs 1 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_norm_1conv_k1_features

# python main.py --train train/ --test test/ --epochs 100 --patience 10 --encoding 0 --k 1 --feat_extraction 0 --num_convs 2 --activation 0 --batch_norm 0 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_2conv_k1/
# python main.py --train train/ --test test/ --epochs 100 --patience 10 --encoding 0 --k 1 --feat_extraction 0 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_norm_2conv_k1/

# python main.py --train train/ --test test/ --epochs 100 --patience 10 --encoding 0 --k 1 --feat_extraction 1 --num_convs 2 --activation 0 --batch_norm 0 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_2conv_k1_features/
# python main.py --train train/ --test test/ --epochs 100 --patience 10 --encoding 0 --k 1 --feat_extraction 1 --num_convs 2 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_norm_2conv_k1_features/

# python main.py --train train/ --test test/ --epochs 100 --patience 10 --encoding 0 --k 1 --feat_extraction 0 --num_convs 3 --activation 0 --batch_norm 0 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_3conv_k1/
# python main.py --train train/ --test test/ --epochs 100 --patience 10 --encoding 0 --k 1 --feat_extraction 0 --num_convs 3 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_norm_3conv_k1/
# python main.py --train train/ --test test/ --epochs 100 --patience 10 --encoding 0 --k 1 --feat_extraction 0 --num_convs 4 --activation 0 --batch_norm 0 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_4conv_k1/
# python main.py --train train/ --test test/ --epochs 100 --patience 10 --encoding 0 --k 1 --feat_extraction 0 --num_convs 4 --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output results/enc0_cnn_norm_4conv_k1/
