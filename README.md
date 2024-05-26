Setting up the environment: All of the required libraries can be installed by running the following commands:

```commandline
pip install -r requirements\requirements.txt
```

Running training script:
Training can be run by:

```commandline
python your_script.py --data_dir "$DATA_DIR" --save_dir "$SAVE_DIR" --batch_size "$BATCH_SIZE" --epochs "$EPOCHS" --lr "$LR" --train_ratio "$TRAIN_RATIO" --val_ratio "$VAL_RATIO" --save_interval "$SAVE_INTERVAL" --attention "$ATTENTION" --initial_orientation "$INITIAL_ORIENTATION" --num_frames "$NUM_FRAMES" --delta_degrees "$DELTA_DEGREES" --fps "$FPS" --fid "$FID"
```
Ex.
```commandline
python .\train.py --data_dir .\dataset\ --save_dir checkpoints --epoch 100 --delta_degrees 45 --num_frames 60  --attention True --lr 1e-4
```

Running test code:
Testing code can be run by:
```commandline
python test.py --data_dir "$DATA_DIR" --save_dir "$SAVE_DIR" --batch_size "$BATCH_SIZE" --attention "$ATTENTION" --initial_orientation "$INITIAL_ORIENTATION" --num_frames "$NUM_FRAMES" --delta_degrees "$DELTA_DEGREES" --fps "$FPS"
```
Ex.
```commandline
python .\test.py --data_dir dataset --save_dir checkpoints --attention=True --num_frames 60
```
