CHECKPOINT_PATH='stuff/checkpoints/peta_epoch_31.pth'
EXPERIMENT=gender
#VAL_ANNOTATIONS='stuff/data_list/peta/PETA_test_list.txt'
VAL_ANNOTATIONS='stuff/data_list/peta/PETA_val_list.txt'
DATA_PATH='stuff/'
TRAIN_ANNOTATIONS='stuff/data_list/peta/PETA_train_list.txt'
#TRAIN_ANNOTATIONS='stuff/data_list/peta/PETA_val_list.txt'
MODEL_PATH='stuff/pretrained_model/bn_inception-52deb4733.pth'
GENERATE_FILE=True

python main.py --approach=inception_iccv --experiment=$EXPERIMENT -e --resume=$CHECKPOINT_PATH --model_path=$MODEL_PATH \
--train_list_path=$TRAIN_ANNOTATIONS --val_list_path=$VAL_ANNOTATIONS --data_path=$DATA_PATH --generate_file=$GENERATE_FILE