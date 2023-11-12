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

```
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

Dealing with futures data is a little more complicated than stock data in general. Especially when calculating returns just using a rolling contract data will lead to wrong results because of price jumps during the contract rolls. To avoid this I've used expired contract data to calculate features and label and then stitched the data together based on which contract is more liquid on a given date.

For the prepare_data.py function to work you will need 5M data already stored in your hard-drive and you will need a function to return it as a pandas dataframe for a given ticker. For example:

```
data_out_5M = id.get_presaved_data(ticker='CLZ2015', interval='5M')
print(data_out_5M.head())
```

Should return:

![](/assets/5M_data_example.JPG)

Now we can calculate the necessary features to build the feature matrix. First we lag the necessary fields from 1 to 23th lag and save them as columns of the dataframe.

```
for i in range(1, 24):
          data_out_5M[['open_' + str(i),'high_' + str(i),'low_' + str(i),'close_' + str(i),'volume_' + str(i)]] = \
          data_out_5M[['open','high','low','close','volume']].shift(i)
```
The authors mention they run the model prediction only every 2 hours so we resample 5M data into 2H using pandas functionality. We also calculate the return of the past 2 hours and futures 2 hours and store them in fields 'percent_diff'  and 'percent_diff1' respcetively.

```
data_2H = data_out_5M.resample('2H').last()
data_2H.dropna(subset='close', inplace=True)
data_2H['percent_diff'] = data_2H['close'].diff()/data_2H['close'].shift(1)
data_2H['std'] = data_2H['percent_diff'].rolling(10).std()
data_2H['percent_diff1'] = data_2H['percent_diff'].shift(-1)
```




