## 📈 **Program for forecast prices open and close stocks of brazillian**

![Customer Touchpoint Map-1](https://user-images.githubusercontent.com/49824600/217238593-6fd89c1b-ca37-46d5-ac1a-5fe9a218b822.jpg)

>📚 ref.:<br>
>* https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21<br>
>* https://keras.io/api/layers/recurrent_layers/lstm/<br>

## 📊 **Why do forecast price of opening/closing stocks?**


Predicting opening/closing stock prices is an important task for **investors and financial analysts**, as it can provide **valuable insights into market trends and help make informed investment decisions**. In addition, forecasting stock opening/closing prices can also be useful for companies, as it can help them **understand how their stocks are affected by economic and political events**, and thus **adjust their business strategies**.

Forecasting stock opening/closing prices can also help identify **investment opportunities and avoid risks**. For example, if a forecasting model predicts that the price of a stock will rise, this could be an o**pportunity for an investor to buy the stock before the price increases**. Conversely, if a forecasting model predicts that a stock's price will fall, this may be a reason for an investor to sell their stock before the price falls.

In summary, forecasting stock opening/closing prices is a valuable tool for investors and financial analysts as it can **help make informed investment decisions**, **identify investment opportunities and avoid risks.** Furthermore, it can be useful for **companies as it can help them understand how their actions are affected by economic and political events** and thus adjust their **business strategies**.

## 🗿 **LSTM (Long Short-Term Memory)**


The **LSTM (Long Short-Term Memory) network** is a **recurrent neural network that is designed to handle time series of data**. One of the key features of the **LSTM network is its ability to maintain long-term information**. This is achieved through the **use of memory cells, which are capable of storing information** for **a long period of time**. In addition, LSTM networks also have **input, output and forget gates**, which allow **controlling the flow of information** in and **out of the memory cell**.

LSTM is a Deep Learning technique, it is a **variation of RNN (Recurrent Neural Network)** that uses **memory cells to remember important information over time**, which **helps to model time series and other data sequences**. This allows the network to **understand the temporal context** and **make accurate predictions**, even when there is **missing data or input noise**.

In summary, the LSTM network is an advanced Machine Learning technique that is designed **to deal with time series of data and has been widely used for tasks such as time series prediction, natural language processing and speech recognition**. Its ability to **retain long-term information is one of its key features**, and this **makes it ideal for stock price forecasting** and other similar applications.


## 🔰 **Steps of LSTM**
---
>  ### 📌 **Forget gate**
>![Forget gate](https://miro.medium.com/max/828/1*GjehOa513_BgpDDP6Vkw2Q.gif)
>  ### 📌 **Input Gate**
>![Input Gate](https://miro.medium.com/max/828/1*TTmYy7Sy8uUXxUXfzmoKbA.gif)
>  ### 📌 **Cell State**
>![Cell State](https://miro.medium.com/max/828/1*S0rXIeO_VoUVOyrYHckUWg.gif)
>  ### 📌 **Output Gate**
>![Output Gate](https://miro.medium.com/max/828/1*VOXRGhOShoWWks6ouoDN3Q.gif)

## 🥳 **How make forecast in program**

1. Use a **EC2 machine** in AWS with allow for write on **S3**
2. Criete a bucket on S3 and replace on file **```app-programmatic.py```** your name in **```my-data-stocks```**
3. Install packges python in file **```requirements.txt```**
4. Open the file **```app-programmatic.py```** and configure variables days and list_stocks_names
    - **```list_stocks_names```**: is List of stocks for forecast
     - **```days```**: is how many days you forecast after
