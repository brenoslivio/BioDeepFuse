# -- Conventional --

python main.py --algorithm 0 --train data/train/ --test data/test/ --feat_extraction 1 2 3 4 5 6 --output results/svm
python main.py --algorithm 1 --train data/train/ --test data/test/ --feat_extraction 1 2 3 4 5 6 --features_exist 1 --output results/xgboost

# -- Deep Learning --

## -- CNN --

for k in 1 2 3
do
    for enc in 0 1 2
    do
        for num_convs in 1 2 3 4
        do  
            output="results/enc${enc}_cnn_${num_convs}conv_k${k}"

            python main.py --train data/train/ --test data/test/ --epochs 100 --patience 20 --encoding ${enc} --k ${k} --num_convs ${num_convs} --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output ${output}
        done

        for concat in 1 2
        do
            for num_convs in 1 2 3 4
            do  
                output="results/enc${enc}_cnn_${num_convs}conv_k${k}"

                if [ "${concat}" -eq 1 ]; then
                output="${output}_concat1_bio"
                elif [ "${concat}" -eq 2 ]; then
                output="${output}_concat2_bio"
                fi
                
                python main.py --train data/train/ --test data/test/ --epochs 100 --patience 20 --encoding ${enc} --concat ${concat} --k ${k} --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs ${num_convs} --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 0 --bidirectional 0 --lstm_dropout 0.2 --output ${output}
            done
        done
    done
done

## -- CNN-BiLSTM -- 

for k in 1 2 3
do
    for enc in 0 1 2
    do
        for num_convs in 1 2 3 4
        do  
            output="results/enc${enc}_cnn_bilstm_${num_convs}conv_k${k}"

            python main.py --train data/train/ --test data/test/ --epochs 100 --patience 20 --encoding ${enc} --k ${k} --num_convs ${num_convs} --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output ${output}
        done

        for concat in 1 2
        do
            for num_convs in 1 2 3 4
            do  
                output="results/enc${enc}_cnn_bilstm_${num_convs}conv_k${k}"

                if [ "${concat}" -eq 1 ]; then
                output="${output}_concat1_bio"
                elif [ "${concat}" -eq 2 ]; then
                output="${output}_concat2_bio"
                fi
                
                python main.py --train data/train/ --test data/test/ --epochs 100 --patience 20 --encoding ${enc} --concat ${concat} --k ${k} --feat_extraction 1 2 3 4 5 6 --features_exist 1 --num_convs ${num_convs} --activation 0 --batch_norm 1 --cnn_dropout 0.2 --num_lstm 1 --bidirectional 1 --lstm_dropout 0.2 --output ${output}
            done
        done
    done
done