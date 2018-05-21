python main.py --phase=train \
    --load \
    --model_file='./models/289999.npy' \
    --attention='bias' \
    --optimizer='SGD' \
    --learning_rate=0.005 

python main.py --phase=train \
    --load \
    --model_file='./models/bias_SGD_0.005_500_Adam_0.0001/290999.npy' \
    --attention='bias' \
    --optimizer='Adam' \
    --learning_rate=0.0001

floyd run  --tensorboard --gpu --env tensorflow-1.4  --data zisang0210/datasets/coco/1:/data --data zisang0210/datasets/faster-rcnn/1:/model "python main.py --phase=train --load_cnn --faster_rcnn_ckpt=/model/faster_rcnn_resnet50_coco_2018_01_28/model.ckpt --faster_rcnn_frozen=/model/exported/frozen_inference_graph.pb --joint_train --attention=bias_fc1 --optimizer=Adam --learning_rate=0.0001"
python main.py --phase=train --load_cnn --faster_rcnn_ckpt=/data/zshwu/data/faster_rcnn_resnet50_coco_2018_01_28/model.ckpt --faster_rcnn_frozen=/data/zshwu/data/frozen_faster_rcnn.pb --joint_train --attention=bias_fc1 --optimizer=Adam --learning_rate=0.0001