Modified version of the code at https://github.com/una-dinosauria/human-motion-prediction.git
to re-run our exerpiment execute:
python src/translate.py --action walking --seq_length_out 25 --iterations 20000
ipython ./src/translate.py -- --fft True --GPU 4 --cgru True --fft True --seq_length_in 61 --seq_length_out 60 --window_size 30 --step_size 5
