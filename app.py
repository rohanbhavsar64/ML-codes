pip install tensorflow
import pandas as pd
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
import streamlit as st
st.title("Stock Analysis")
fd=pd.read_csv('constituents.csv')
f=fd['Symbol'].unique()
#Mt(15px) Lh(1.6)
ticker=(st.text_input("Company")) or ('MMM')
url=f"https://groww.in/us-stocks/{ticker}"
df=pd.DataFrame()
#<fin-streamer class="Fw(500) Pstart(8px) Fz(24px)" data-symbol="INFY" data-test="qsp-price-change" data-field="regularMarketChange" data-trend="txt" data-pricehint="2" value="-0.6700001" active=""><span class="e3b14781 ee3e99dd dde7f18a"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">-0.64</font></font></span></fin-streamer>
headers={'User-Agent':'Mo BeautifulSoupzilla/5.0 (Windows NT 6.3; Win 64 ; x64) Apple WeKit /537.36(KHTML , like Gecko) Chrome/80.0.3987.162 Safari/537.36'}
r=requests.get(url,headers=headers)
web=BeautifulSoup(r.text,'html')
if web.find(class_='usph14HeadingWrapper valign-wrapper vspace-between') is None:
    st.write('Element not found')
else:
    st.write(web.find(class_='usph14HeadingWrapper valign-wrapper vspace-between').text.strip())
    c=str('$'+web.find(class_='uht141Pri contentPrimary displayBase').text)
    st.subheader('Current Price : '+str(c))

#<div class="D(ib) Mt(-5px) Maw(38%)--tab768 Maw(38%) Mend(10px) Ov(h) smartphone_Maw(85%) smartphone_Mend(0px)"><div class="D(ib) "><h1 class="D(ib) Fz(18px)">Infosys Limited (INFY)</h1></div><div class="C($tertiaryColor) Fz(12px)"><span>NYSE - Nasdaq Real Time Price. Currency in USD</span></div></div>
import yfinance as yf
# Get the data for the stock AAPL
j=['Analysis','Profile','Financial','Historical Data','Recommendations']
h=st.sidebar.radio('Field',j)
if h=='Analysis':
    st.subheader('Analysis')
    col1,col2=st.columns(2)
    with col1:
        l = ['1y','5y','max','3mo','1mo','5d']
        i = st.selectbox('Period', l)
        if ticker is None:
           st.write('PLEASE WRITE NAME OF CORRECT TICKER')
        else:
            try:
                df = yf.download(ticker, period=i)
            except Exception as e:
                st.write(f"Error: {e}. PLEASE WRITE NAME OF CORRECT TICKER")
    with col2:
        q=['Candlestick','Line','Bar']
        g=st.radio('Chart- Type',q,horizontal=True)
    if g=='Candlestick':
        fig = go.Figure(data=[go.Candlestick(x=df.index,
                                     open=df['Open'],
                                     high=df['High'],
                                     low=df['Low'],
                                     close=df['Close'])])
        fig.update_layout(title='Candlestick Chart - Closing Price vs Date',
                  yaxis_title='Price',
                  xaxis_title='Date',
                  xaxis_rangeslider_visible=False)
        st.write(fig)
    elif g=='Line':
        if df['Close'][-1] < df['Close'][0]:
            fig = px.line(df, x=df.index, y='Close', color_discrete_sequence=["#9E4033"],title='Close vs Date')
            fig.update_yaxes(showgrid=False)
            st.write(fig)
        else:
            fig = px.line(df, x=df.index, y='Close', color_discrete_sequence=["#4A7230"],title='Closing Price  vs Date')
            fig.update_yaxes(showgrid=False)
            st.write(fig)
    else:
        if df['Close'][-1] < df['Close'][0]:
            fig = px.bar(df, x=df.index, y='Close', color_discrete_sequence=["#9E4033"],title='Closing Price vs Date')
            fig.update_yaxes(showgrid=False)
            st.write(fig)
        else:
            fig = px.bar(df, x=df.index, y='Close', color_discrete_sequence=["#4A7230"],title='Closing Price  vs Date')
            fig.update_yaxes(showgrid=False)
            st.write(fig)

elif h=='Profile':
    url1 = f'https://groww.in/us-stocks/{ticker}'
    res = requests.get(url1, headers=headers)
    w = BeautifulSoup(res.text, 'html')
    v = w.find_all(class_='abs654Para bodyLarge')[0].text
    st.subheader('Description')
    if v!=None:
        st.write(v,fontsize=8)
    else:
        st.write('Profile Not Exist')
elif h=='Financial':
    st.subheader('Financial')
    MMM=yf.Ticker(ticker)
    w=['Balance Sheet','Income Statements','Financial Statements']
    q=st.radio('Select',w,horizontal=True)
    if q=='Balance Sheet':
        df1=MMM.quarterly_balance_sheet.iloc[:10,:4]
        st.table(df1)
    elif q=='Income Statements':
        df7=MMM.income_stmt.iloc[:15,:4]
        st.table(df7)
    else:
        df2=MMM.quarterly_financials.iloc[44:,:]
        s=['Gross Profit','Cost Of Revenue','Total Revenue','Operating Revenue']
        m=st.selectbox('Fanancial',s)
        #m=['Gross Profit','Total Revenue']
        h=0
        if(m=='Gross Profit'):
            h=42
        elif(m=='Cost Of Revenue'):
            h=43
        elif(m=='Total Revanue'):
            h=44
        else:
            h=45
        a1=0
        a2=0
        a3=0
        a4=0
        f=['Yearly','quarterly']
        n=st.radio('Period',f,horizontal=True)
        if(n=='Yearly'):
            a=MMM.financials.iloc[44:,:]
            b=a.columns
            a1=b[0]
            a2=b[1]
            a3=b[2]
            a4=b[3]
        else:
            a=MMM.quarterly_financials.iloc[44:,:] 
            b=a.columns
            a1=b[0]
            a2=b[1]
            a3=b[2]
            a4=b[3]
        b1=0
        b2=0
        b3=0
        b4=0
        if(n=='Yearly'):
            c=MMM.financials.iloc[h,:]
            b1=c[0]
            b2=c[1]
            b3=c[2]
            b4=c[3]
        else:
            c=MMM.quarterly_financials.iloc[h,:]
            b1=c[0]
            b2=c[1]
            b3=c[2]
            b4=c[3]
        data=[[a1,b1],[a2,b2],[a3,b3],[a4,b4]]
        df=pd.DataFrame(data,columns=['Year',m])
        st.write(px.bar(df,x='Year',y=m,color_discrete_sequence=['orange']))
elif h == 'Historical Data':
    import pandas as pd
    import datetime as dt
    from datetime import date
    import matplotlib.pyplot as plt
    import yfinance as yf
    import numpy as np
    import tensorflow as tf
    START = "2015-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")

    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data

    data = load_data('AAPL')
    df = data.drop(['Date', 'Adj Close'], axis=1)

    # Splitting the data into training and testing sets
    train = pd.DataFrame(data[0:int(len(data) * 0.70)])
    test = pd.DataFrame(data[int(len(data) * 0.70): int(len(data))])

    # Scaling the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_close = train.iloc[:, 4:5].values
    test_close = test.iloc[:, 4:5].values
    data_training_array = scaler.fit_transform(train_close)

    x_train = []
    y_train = []

    for i in range(100, data_training_array.shape[0]):
        x_train.append(data_training_array[i-100:i])
        y_train.append(data_training_array[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # Building the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units=60, activation='relu', return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(units=80, activation='relu', return_sequences=True))
    model.add(Dropout(0.4))

    model.add(LSTM(units=120, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['MAE'])
    model.fit(x_train, y_train, epochs=10)
    model.save('keras_model.h5')

    # Preparing the test data
    past_100_days = pd.DataFrame(train_close[-100:])
    test_df = pd.DataFrame(test_close)
    final_df = pd.concat([past_100_days, test_df], ignore_index=True)
    input_data = scaler.transform(final_df)  # Use transform instead of fit_transform

    x_test = []
    y_test = []
    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)

    # Making predictions
    y_pred = model.predict(x_test)
    scale_factor = 1 / scaler.scale_[0]  # Correct scaling factor
    y_pred = y_pred * scale_factor
    y_test = y_test * scale_factor

    # Plotting the results
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, 'b', label="Original Price")
    plt.plot(y_pred, 'r', label="Predicted Price")
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
     
else:
    st.subheader('Recommendations')
    MMM=yf.Ticker(ticker)
    df1=MMM.quarterly_balance_sheet.iloc[:10,:4]
    df2=MMM.quarterly_financials.iloc[44:,:]
    l=MMM.recommendations.iloc[:,0]
    t=st.radio('Period',l,horizontal=True)
    k=0
    if(t=='0m'):
        k=0
    elif(t=='-1m'):
        k=1
    elif(t=='-2m'):
        k=2
    else:
        k=3
    x2=MMM.recommendations.columns[1]
    x3=MMM.recommendations.columns[2]
    x4=MMM.recommendations.columns[3]
    x5=MMM.recommendations.columns[4]
    x6=MMM.recommendations.columns[5]
    y2=MMM.recommendations.iloc[k,:][1]
    y3=MMM.recommendations.iloc[k,:][2]
    y4=MMM.recommendations.iloc[k,:][3]
    y5=MMM.recommendations.iloc[k,:][4]
    y6=MMM.recommendations.iloc[k,:][5]
    data1=[x2,x3,x4,x5,x6]
    data2=[y2,y3,y4,y5,y6]
    #df3=pd.DataFrame(data5,columns=['Suggestions','Vote'])
    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Pie(labels=data1, values=data2, hole=.6)])
    st.write(fig)
    
    
    
    
