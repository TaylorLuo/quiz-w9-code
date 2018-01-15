本地运行方法：
1.下载vgg16.ckpt
2.修改convert_fcn_dataset.py
3.执行脚本：python convert_fcn_dataset.py --data_dir=/usr/downloads/VOCdevkit/VOC2012/ --output_dir=./
生成训练数据集和验证数据集
4.执行脚本：python train.py --checkpoint_path ./vgg_16.ckpt --output_dir /media/taylor/新加卷2/output --dataset_train ./fcn_train.record --dataset_val ./fcn_val.record --batch_size 16 --max_steps 2000
