# Review-Spammers

This is a RNN-based deep neural network  which is constructed to classify whether a user on DianPing APP is a spammer.

The dataset we use can be downloaded  [here](https://jbox.sjtu.edu.cn/l/NHfFWh )

## Environment Settings
We use Keras with Tensorflow as the backend.
- Keras version: '2.2.0'
- Tensorflow version: '1.8.0' 

## Example to run the codes.

Build dataset: After download the dataset, it should be put into the "dataset" dirctory.
```
cd dataset
python data_handler.py
```

Run LSTM:
 ```
 python lstm.py
 ```
