# Trading Crude Oil Futures with Neural Networks

I was reading an article which applies a CNN (Convolutional Neural Network) to financial data and the authors claim the resulting strategy have very consistent profitability. You can read the article here: [Financial Markets Prediction with Deep Learning](https://arxiv.org/abs/2104.05413) This sparked my interest so I've decided to replicate their results. I couldn't find the code for this model anywhere on the internet so I wrote it myself. I wrote to the authors asking a copy of their code but nobody has yet got back to me.

## What are The Strategy Results Like?

The authors claim that applying their CNN model to Crude Oil futures yielded a 500% return over 71 months. This is approximately equal to a 43% CAGR (Compound Annual Growth Rate). The strategy was profitable during 60 months of the 71 months. Below is a plot from the article.

![Strategy Performance for Crude Oil Futures](/assets/strategy_performance.png)

## Data

I have used crude oil futures data from [FirstRate Data](https://firstratedata.com/). To stay consistent with the article I've only used 5M candles that provide the open, high, low, close and volume in 5 minute intervals from January 2010 to November 2017. First Rate provides rolling contract data as well as specific expired contract data. Specific expired contract data is more relevant for our usecase because using rolling conract data to calculate daily returns can be problematic if one doesn't have information about the exact dates of the contract roll.

## Model

If you need to brush up on your neural networks knowledge and convolutions know how in particular, I recommend [Dive into Deep Learning](https://d2l.ai/chapter_convolutional-neural-networks/index.html). In order to apply convolutions, the authers first transform the price data into a two dimensional feature space. This means every observation will consist of a two dimensional feature matrix and a single label related to future price movement. Lets take a look at the following illutration from the article:

![Data Structure and CNN Architecture](/assets/feature_space_and_model_architecture.png)

This first rectangular of the above figure describes the feature matrix before the application of any convolutions. Along the x-axis you have different lagged values for various data types. For example o<sub>1</sub> is the opening price for the 5 minute bar in that particular observation, o<sub>2</sub> is the opening price for the previous 5 minute bar etc. Along the y-axis of the feature matrix, we have various candle data types are lined up. These are open price, high price, low price, close price and volume. Each consecutive 5 minute data is organized to form a 5X24 feature matrix.

The 1-D kernels (the red and blue one) scan along the x-axis while each one of them goes to every position of feature matrix by a stride of one. The C' and C" represents the output channels of the red and blue kernel, respectively. Because of padding, the convolutional output channels have the same dimension as the input matrix. A max-pooling layer follows a convolutional layer, and max-pooling layers condense the dimensions of the x-axis. The M<sup>*</sup> and M<sup>**</sup> represent the output channels of the max-pooling operations (the orange and green one) by a stride of three.

The desired effect of kernels sliding along x-axis can be achieved using Conv2D in combination with kernels with height 1. The tensorflow code for the full model is below:

```python
conv2d_strides = 1
kernel_regularizer = 1e-5
adam_initial_learning_rate = 1e-3
dropout_rate = 0.3
model = Sequential()

conv2d_layer1 = Conv2D(32, (1,4), strides = conv2d_strides,
                       kernel_regularizer=regularizers.l2(kernel_regularizer),
                       padding='same', activation='relu', use_bias=True,
                       kernel_initializer='glorot_uniform',
                       input_shape=(5,24,1))


model.add(conv2d_layer1)
model.add(MaxPool2D(pool_size=(1,4), strides=(1,4), padding='valid'))

conv2d_layer2 = Conv2D(64, (1, 3), strides=conv2d_strides,
                       kernel_regularizer=regularizers.l2(kernel_regularizer),
                       padding='same', activation='relu', use_bias=True,
                       kernel_initializer='glorot_uniform')

model.add(conv2d_layer2)
model.add(MaxPool2D(pool_size=(1, 3), strides=(1,3), padding='valid'))

conv2d_layer3 = Conv2D(128, (1, 2), strides=conv2d_strides,
                       kernel_regularizer=regularizers.l2(kernel_regularizer),
                       padding='same', activation='relu', use_bias=True,
                       kernel_initializer='glorot_uniform')

model.add(conv2d_layer3)
model.add(MaxPool2D(pool_size=(1, 2), strides=(1, 2), padding='valid'))

model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(3, activation='softmax'))

optimizer = optimizers.Adam(learning_rate=adam_initial_learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
```
There are three convolutional layers and the number of output layers for each are 32, 64 and 128. As a result of 3 max-pooling operations the x-axis shrink from 24 to 1.
Convolutional layers are followed by two fully connected layers with 1000 and 500 units and an output softmax layer. Model summary is as follows:

![Model Summary](/assets/model_summary.JPG)

## Data Processing

### Collecting Features

Dealing with futures data is a little more complicated than stock data in general. Especially when calculating returns just using a rolling contract data will lead to wrong results because of price jumps during the contract rolls. To avoid this I've used expired contract data to calculate features and label and then stitched the data together based on which contract is more liquid on a given date. All the code related to data processing can be found in prepare_data.py.

For the prepare_data.py function to work you will need 5M data already stored in your hard-drive and you will need a function to return it as a pandas dataframe for a given ticker. For example:

```python
data_out_5M = id.get_presaved_data(ticker='CLZ2015', interval='5M')
print(data_out_5M.head())
```

Should return:

![](/assets/5M_data_example.JPG)

Now we can calculate the necessary features to build the feature matrix. Let's take a look prepare_2H_data() function. First we lag the necessary fields from 1 to 23th lag and save them as columns of the dataframe.

```python
for i in range(1, 24):
          data_out_5M[['open_' + str(i),'high_' + str(i),'low_' + str(i),'close_' + str(i),'volume_' + str(i)]] = \
          data_out_5M[['open','high','low','close','volume']].shift(i)
```
### Resampling Data and Calculating The Labels

The authors mention they run the model prediction only every 2 hours so we resample 5M data into 2H using pandas functionality. We also calculate the return of the past 2 hours and futures 2 hours and store them in fields 'percent_diff'  and 'percent_diff1' respcetively. We also calculate the rolling standard deviation using the past 10 observations. This will be used in calculating the label.

```python
data_2H = data_out_5M.resample('2H').last()
data_2H.dropna(subset='close', inplace=True)
data_2H['percent_diff'] = data_2H['close'].diff()/data_2H['close'].shift(1)
data_2H['std'] = data_2H['percent_diff'].rolling(10).std()
data_2H['percent_diff1'] = data_2H['percent_diff'].shift(-1)
```
We do the above prodecure for each individual expiry and below we combine this data and drop the duplicates as we only want one observation for each datetime. We calculate the label using the future 2H return and past standard deviation. Label will be 1 for returns higher than a threshold level and 0 for returns lower than a threshold level and 1 for the rest of the observations.

```python
data_list.append(data_2H)

raw_data = pd.concat(data_list)
raw_data['settle_date'] = pd.to_datetime(raw_data.index.date)
raw_data['datetime_'] = pd.to_datetime(raw_data.index)
raw_data.sort_values(['datetime_', 'cont_indx'], ascending=[True, False], inplace=True)
raw_data.drop_duplicates(subset='datetime_', keep='first', inplace=True)
raw_data.dropna(subset=['std', 'percent_diff1'], inplace=True)

raw_data['label'] = 1
raw_data.loc[raw_data['percent_diff1']>0.55*raw_data['std'],'label'] = 2
raw_data.loc[raw_data['percent_diff1']<-0.55*raw_data['std'],'label'] = 0

raw_data.to_pickle('cnn_data.pkl')
```

get_features() function in prepare_data.py helps us extract the features from the raw_data. Below is the implementation.

```python
def get_features(**kwargs):

    df = kwargs['df']

    column_list = []
    primary_column_names = ['open', 'high', 'low', 'close', 'volume']
    for col_i in primary_column_names:
        column_list = column_list + [col_i] + [col_i + '_' + str(x) for x in range(1, 24)]

    return df[column_list]
```

Now let't take a look at the columns of the output of get_features:

```python
import strategy_development.cnn.cnn2.prepare_data as prep
import pandas as pd
final_data = prep.prepare_2H_data()
feature_data = prep.get_features(df=final_data)
with pd.option_context('display.width', 150,'display.max_seq_items', None):
    print(feature_data.columns)
```
This will yield the following list:

![](/assets/feature_list.JPG)

As you can see we have all the necessary columns in feature_data. However we will need to reshape it to form a 2 dimensional matrix for each observation. At the moment each observation is a 1 dimensional vector.

### Calculating Indices for Rolling Window Training

The authors use a rolling window methodology to train and test the model. Training is done with 2 years of data. The next 4 weeks of data is used for validation and the following 2 weeks are used for testing the strategy. After the results are collected the starting point of each window is moved by 2 weeks after which model estimation and testing restarts again using this new data. The following function in prepare_data.py will generate the necessary indices to access the training, validation and test data for each iteration of the rolling training.

```python
def prepare_rolling_simulation_indices(**kwargs):

    entire_data = kwargs['entire_data']

    training_size = 60*52*2
    validation_size = 4*60
    test_size = 2*60
    total_size = training_size + validation_size + test_size

    train_start_index = list(range(0, len(entire_data) - total_size, test_size))
    train_end_index = [x + training_size for x in train_start_index]
    validation_start_index = train_end_index
    validation_end_index = [x + validation_size for x in validation_start_index]
    test_start_index = validation_end_index
    test_end_index = [x + test_size for x in test_start_index]
    test_end_index[-1] = -1

    return {'train_start_index_list': train_start_index,'train_end_index_list': train_end_index,
            'validation_start_index_list': validation_start_index, 'validation_end_index_list': validation_end_index,
            'test_start_index_list': test_start_index, 'test_end_index_list': test_end_index}
```

Let's take a look at the output of this function by printing out the 55th and 56th values of the returned lists.

```python
i = 55
print('example case:')
print('train_start_index: ' + str(train_start_index_list[i]))
print('train_end_index: ' + str(train_end_index_list[i]))
print('test_start_index: ' + str(test_start_index_list[i]))
print('test_end_index: ' + str(test_end_index_list[i]))
print('next train_start_index: ' + str(train_start_index_list[i+1]))
print('next train_end_index: ' + str(train_end_index_list[i+1]))
print('next test_start_index: ' + str(test_start_index_list[i+1]))
print('next test_end_index: ' + str(test_end_index_list[i+1]))
```

![](/assets/example_rolling_indices.JPG)

For each training iteration we have train_end_index-train_start_index = 6240 observations. We have approximately 60 (5*12) 2 hour candles in a week. So each training window has 6240/60=104 weeks of data which is equivalent to two years of data as authors prescribe. Test windows came after training windows and they each have test_end_index-test_start_index=120 observations equivalent to 2 weeks of data. Also all the data indices move by 120 from iteration 55 to iteration 56 as prescribed in the article.

### Reshaping the Feature Data into Feature Matrix

As we have discussed we still have each observation as a 1 dimensional vector and we need to reshape this 1x120 vector into 5x24 matrix. The below function in prepare_data.py accomplishes this:

```python
def reshape_data(data_input):

    return np.reshape(data_input, (data_input.shape[0], 5, 24))
```

Let's see the above function in action to make sure it does what it's supposed to do:

```python
# prep.reshape_data is correctly transofrming data into (num_obs,5,24) shaped matrix

import prepare_data as prep
import numpy as np
from sklearn.preprocessing import StandardScaler

final_data = prep.prepare_2H_data()
feature_data = prep.get_features(df=final_data)
rolling_indices_output = prep.prepare_rolling_simulation_indices(entire_data=final_data)

train_start_index_list = rolling_indices_output['train_start_index_list']
train_end_index_list = rolling_indices_output['train_end_index_list']

test_start_index_list = rolling_indices_output['test_start_index_list']
test_end_index_list = rolling_indices_output['test_end_index_list']

i = 50

train_start_index = train_start_index_list[i]
train_end_index = train_end_index_list[i]

test_start_index = test_start_index_list[i]
test_end_index = test_end_index_list[i]

x_train_i = feature_data.iloc[train_start_index:train_end_index, :]
y_train_i = np.array(final_data['label'].iloc[train_start_index:train_end_index])

x_test_i = feature_data.iloc[test_start_index:test_end_index, :]
y_test_i = final_data['label'].iloc[test_start_index:test_end_index]

scaler_i = StandardScaler()
x_train_i_t = scaler_i.fit_transform(x_train_i)
x_test_i_t = scaler_i.transform(x_test_i)

x_train_reshaped = prep.reshape_data(x_train_i_t)
x_test_reshaped = prep.reshape_data(x_test_i_t)

print(x_train_i_t[100])
print(x_train_reshaped[100])
print(x_train_reshaped.shape)
```
The above script returns the following:

![](/assets/reshape_output.JPG)

We have used the 100<sup>th</sup> observation of the 50<sup>th</sup> training iteration as an example. The 100<sup>th</sup> observation of x_train_i_t is a 1x120 vector as expected. And we can see the first row of the 100<sup>th</sup> observation of x_train_reshaped is equal to the first 24 elements of the 100<sup>th</sup> observation of x_train_i_t. Also we print the dimensiont of x_train_reshaped which shows that it's comprised of 6240 observations of 5x24 matrices.

## Training with a GPU in Windows

I have found that the easiest way to train a tensorflow model in windows is to use a docker image. And if you are using visual studio code, one of the most convenient ways to have access to your python library while running a container is to use [Dev Containers Extension](https://code.visualstudio.com/docs/devcontainers/containers) of Visual Studio Code. First you will need to figure out which python libraries and files you will need to run during the training. Then inside the folder that contains all your python files, create a .devcontainer folder. Inside this .devcontainer you will need devcontainer.json, Dockerfile, requirements.txt files. You can find copies of the files I've used in the repository. After this, you will be able to open your folder which contains all your python files in a docker container from within your Visual Studio Code with access to your GPU.

## Training Results

Unfortunately unlike the results presented in the article, for the data I've used the model is not doing a good job in learning any tradable patterns. I've experimented with including more dropout or less dropout layers and batch normalization layers but the results are either overfitting or not learning anything. Remember the model is trained every two weeks so for reporting purposes I've averaged the loss and accuracy results across 146 runs for each of the 200 epochs.

### Training Results With Batch_Size=64, 1 Dropout Layer

<p float="left">
  <img src="/assets/benchmark_learning_curve.JPG" width="500" />
  <img src="/assets/benchmark_accuracy_curve.JPG" width="500"/> 
</p>

The plot on the left shows the progress of the loss curve as the training progresses across epochs. Training loss is going down as expected but validation loss is exploding which is a sign of massive overfitting. These results are supported by the plot on the right as for the training dataset the accuracy reaches almost 90% whereas validation accuracy is less than 50%.

### Training Results With Batch_Size=12, 1 Dropout Layer

<p float="left">
  <img src="/assets/learning_curve_batch_size12.JPG" width="500" />
  <img src="/assets/accuracy_curve_batch_size12.JPG" width="500"/> 
</p>

The authors have used batch_size=12 so I ran the model with this paramater as well but the results are not significantly different. Again massive overfitting can be observed in the above plots.

### Training Results With Batch_Size=64, 5 Dropout Layers

<p float="left">
  <img src="/assets/learning_curve_batch_size64_5dropout.JPG" width="500" />
  <img src="/assets/accuracy_curve_batch_size64_5dropout.JPG" width="500"/> 
</p>

The authors weren't specific about the locations of the dropout layers so here I put a dropout layer after every convolutional layer and fully connected layer which amounts to 5 dropout layers in total. The plots above show that with 5 dropout layers the model has trouble learning anything at all as the training loss remains above the validation loss.

### Training Results With Batch_Size=64, 4 Dropout Layers, 1 Batch Normalization

<p float="left">
  <img src="/assets/learning_curve_batch_size64_4dropout_1bn.JPG" width="500" />
  <img src="/assets/accuracy_curve_batch_size64_4dropout_1bn.JPG" width="500"/> 
</p>

Although the authors have never used batch normalization I've decided to replace one of the dropout layers with a batch normalization layer just to experiment. From the above curves it seems that (4 Dropout, 1 Batch Normalization) model learns a little better than the 5 Dropout model.

### Training Results With Batch_Size=64, 3 Dropout Layers, 1 Batch Normalization

<p float="left">
  <img src="/assets/learning_curve_batch_size64_3dropout_1bn.JPG" width="500" />
  <img src="/assets/accuracy_curve_batch_size64_3dropout_1bn.JPG" width="500"/> 
</p>

Removing one more layer of dropout layer quickly results in overfitting as can be seen from the above plots.

### Training Results With Batch_Size=64, 3 Dropout Layers, 2 Batch Normalizations

<p float="left">
  <img src="/assets/learning_curve_batch_size64_3dropout_2bn.JPG" width="500" />
  <img src="/assets/accuracy_curve_batch_size64_3dropout_2bn.JPG" width="500"/> 
</p>

Adding second layer of batch normalization to (3 Dropout, 1 Batch Normalization) model lowers the amount of curve fitting but this is not nearly enough to remove the overall curve fitting of the model.

## Trading Strategy Results

Although the learning algorithm doesn't seem to generalize well according to above results I still report the results of the following trading strategy. The strategy rules are simple and similar to what's suggested by the authors. At any given time we use the most recently trained model to make predictions and if the 









