# SupContrast: Supervised Contrastive Learning

Solving the Facial Expression Recognition Task using POSTER++ as backbone and SupCon as loss function

## Installation
Clone git
```bash
!git clone https://github.com/imphetamine/SupContrast.git
```
Install libraries

```bash
!pip install tensorboard_logger
!pip install tensorrt
!pip install thop
```
## Create data folder for RAF-DB dataset (CIFA dataset will be automatically downloaded)
```bash
cd /kaggle/working/SupContrast
!mkdir /kaggle/working/SupContrast/data
!unrar x /content/drive/MyDrive/POSTER_V2/raf-db.rar
```
## Training
### For CIFAR dataset
Pretraining stage:
```bash
cd /kaggle/working/SupContrast
!python main_supcon_original.py --batch_size 32 \
  --learning_rate 0.5 \
  --temp 0.1 \
  --cosine \
  --epochs 30
```
Linear evaluation stage: (ckpt is path of pretrained model from pretraining stage)
``` bash
!python main_linear.py --batch_size 32 \
  --learning_rate 5 \
  --epochs 20 \
  --ckpt /kaggle/input/model-supcon/last.pth
```

### For RAFDB dataset
Pretraining stage:
```bash
cd /kaggle/working/SupContrast
!python main_supcon.py --batch_size 32 \
  --learning_rate 0.5 \
  --temp 0.1 \
  --cosine \
  --epochs 20 \
  --data_folder /kaggle/input/raf-db \
  --dataset 'path' \
  --mean "(0.4914, 0.4822, 0.4465)" \
  --std "(0.2023, 0.1994, 0.2010)" \
  --use_head "False"
```
Linear evaluation stage: (ckpt is path of pretrained model from pretraining stage)
``` bash
!python main_linear_posterv2.py --batch_size 32 \
  --learning_rate 5 \
  --epochs 20 \
  --ckpt /kaggle/input/model-supcon-posterv2/last.pth \
  --dataset 'path' \
  --data_folder /kaggle/input/rafdb-valid/raf-db/train \
  --valid_folder /kaggle/input/rafdb-valid/raf-db/valid
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Reference
```
@Article{khosla2020supervised,
    title   = {Supervised Contrastive Learning},
    author  = {Prannay Khosla and Piotr Teterwak and Chen Wang and Aaron Sarna and Yonglong Tian and Phillip Isola and Aaron Maschinot and Ce Liu and Dilip Krishnan},
    journal = {arXiv preprint arXiv:2004.11362},
    year    = {2020},
}
```
