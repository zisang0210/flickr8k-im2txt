python main.py --phase=train \
    --load \
    --model_file='./models/289999.npy' \
    --attention='fc2' \
    [--train_cnn]

python main.py --phase=test \
    --model_file='./models/289999.npy' \
    --attention='fc2' \
    --beam_size=3

python main.py --phase=eval \
    --model_file='./models/306999.npy' \
    --beam_size=3


scp zshwu@202.114.96.180:/home/zshwu/data/raw_im2txt/summary/events.out.tfevents.1525169478.n0399 ./

python export.py --model_folder='./models/289999.npy' \
    --output_path='./models/frozen_att_nic_test.pb' \
    --attention='fc2'

python export.py --model_folder='../output/289999.npy' \
    --output_path='../data/frozen_att_nic_test.pb' \
    --attention='fc2'

scp zshwu@202.114.96.180:/home/zshwu/data/raw_im2txt/val/results.json ./