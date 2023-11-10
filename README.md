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






