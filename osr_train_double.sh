DATASET='cifar-10-100'
for SPLIT_IDX in 0 1 2 3 4; do
    python clip_double.py --dataset=$DATASET --split_idx=${SPLIT_IDX}
done
