# BAL: Balancing Diversity and Novelty for Active Learning - Official Pytorch Implementation

## Experiment Setting
Install the requirements
```bash
pip install -r requirements.txt
```

Prepare the dataset in the following format
```
- DATA_PATH
    - DATASET
        - train
            - CLS
                - *.jpg
        - test
            - CLS
                - *.jpg
```
e.g.
```
- data
    - cifar10
        - train
            - 0 
                - airplane_3.jpg
                - airplane_10.jpg
                ...
            - 1 
            ...
            - 9
        - test
            - 0
            ...
            - 9
    - caltech101
        - train
        - test
    - svhn 
        - train
        - test
    ...
```

## Running the Code
1. To train the rotation prediction task on the unlabeled set:
```
python rotation.py \
    --save $SAVE \
    --net vgg16 \
    --dataset cifar10 \
    --datapath $DATA_PATH \
    --lr 0.1 \
    --batch_size 256
```

2. To kmeans cluster pretext features and sort the unlabeled pool:
```
python kmeans.py \
    --net vgg16 \
    --dataset cifar10 \
    --datapath $DATA_PATH \
    --load $LOAD_DIR 
```

3. To train and evaluate on active learning task:
```
python main.py \
    --net vgg16 \
    --dataset cifar10 \
    --datapath $DATA_PATH \
    --per_samples_list 10 10 10 10 10 10 10 10 10 10 \ # change it according to your AL setting
    --addendum 5000 \                                  # change it according to your AL setting
    --save $SAVE \
    --beta 1.0 \
    --milestone 30 60 90 \
    --sort high2low \
    --sampling confidence \
    --first high1st \
    --lr 0.1 \
    --sorted_dataset_path $SORTED_DATASET_PATH
```

## Acknowledgement
Part of the code is modified from [PT4AL](https://github.com/johnsk95/PT4AL) repo.
