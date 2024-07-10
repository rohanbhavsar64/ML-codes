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
ticker=st.selectbox("Company",f)
url=f"https://groww.in/us-stocks/{ticker}"
df=pd.DataFrame()
#<fin-streamer class="Fw(500) Pstart(8px) Fz(24px)" data-symbol="INFY" data-test="qsp-price-change" data-field="regularMarketChange" data-trend="txt" data-pricehint="2" value="-0.6700001" active=""><span class="e3b14781 ee3e99dd dde7f18a"><font style="vertical-align: inherit;"><font style="vertical-align: inherit;">-0.64</font></font></span></fin-streamer>
headers={'User-Agent':'Mo BeautifulSoupzilla/5.0 (Windows NT 6.3; Win 64 ; x64) Apple WeKit /537.36(KHTML , like Gecko) Chrome/80.0.3987.162 Safari/537.36'}
r=requests.get(url,headers=headers)
web=BeautifulSoup(r.text,'html')
st.write(web.find(class_='usph14HeadingWrapper valign-wrapper vspace-between').text)

#<div class="D(ib) Mt(-5px) Maw(38%)--tab768 Maw(38%) Mend(10px) Ov(h) smartphone_Maw(85%) smartphone_Mend(0px)"><div class="D(ib) "><h1 class="D(ib) Fz(18px)">Infosys Limited (INFY)</h1></div><div class="C($tertiaryColor) Fz(12px)"><span>NYSE - Nasdaq Real Time Price. Currency in USD</span></div></div>
c=str('$'+web.find(class_='uht141Pri contentPrimary displayBase').text)
st.subheader('Current Price : '+str(c))
import yfinance as yf
# Get the data for the stock AAPL
j=['Analysis','Profile','Financial','Historical Data','Recommendations','PREDICATION']
h=st.sidebar.radio('Field',j)
if h=='Analysis':
    st.subheader('Analysis')
    col1,col2=st.columns(2)
    with col1:
        l = ['1y','5y','max','3mo','1mo','5d']
        i = st.selectbox('Period', l)
        df = yf.download(ticker, period=i)
    with col2:
        q=['Candlestick','Area','Line','Bar']
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
    elif g=='Area':
        if df['Close'][-1] < df['Close'][0]:
            fig = px.area_chart(df, x=df.index, y='Close', color="#9E4033",title='Close vs Date')
        else:
            fig = st.area_chart(df, x=df.index, y='Close', color="#4A7230",title='Close vs Date')

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
elif h=='Historical Data':
    l = ['5d','1mo','3mo','1y','5y','max']
    i = st.radio('Period', l,horizontal=True)
    df = yf.download(ticker, period=i)
    st.write(df)
elif h=='PREDICATION':
    l = ['5d','1mo','3mo','1y','5y','max']
    i = st.radio('Period', l,horizontal=True)
    df = yf.download(ticker, period=i)
     def LSTM_ALGO(df):
        #Split data into training set and test set
        dataset_train=df.iloc[0:int(0.8*len(df)),:]
        dataset_test=df.iloc[int(0.8*len(df)):,:]
        ############# NOTE #################
        #TO PREDICT STOCK PRICES OF NEXT N DAYS, STORE PREVIOUS N DAYS IN MEMORY WHILE TRAINING
        # HERE N=7
        ###dataset_train=pd.read_csv('Google_Stock_Price_Train.csv')
        training_set=df.iloc[:,4:5].values# 1:2, to store as numpy array else Series obj will be stored
        #select cols using above manner to select as float64 type, view in var explorer

        #Feature Scaling
        from sklearn.preprocessing import MinMaxScaler
        sc=MinMaxScaler(feature_range=(0,1))#Scaled values btween 0,1
        training_set_scaled=sc.fit_transform(training_set)
        #In scaling, fit_transform for training, transform for test
        
        #Creating data stucture with 7 timesteps and 1 output. 
        #7 timesteps meaning storing trends from 7 days before current day to predict 1 next output
        X_train=[]#memory with 7 days from day i
        y_train=[]#day i
        for i in range(7,len(training_set_scaled)):
            X_train.append(training_set_scaled[i-7:i,0])
            y_train.append(training_set_scaled[i,0])
        #Convert list to numpy arrays
        X_train=np.array(X_train)
        y_train=np.array(y_train)
        X_forecast=np.array(X_train[-1,1:])
        X_forecast=np.append(X_forecast,y_train[-1])
        #Reshaping: Adding 3rd dimension
        X_train=np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))#.shape 0=row,1=col
        X_forecast=np.reshape(X_forecast, (1,X_forecast.shape[0],1))
        #For X_train=np.reshape(no. of rows/samples, timesteps, no. of cols/features)
        
        #Building RNN
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import Dropout
        from keras.layers import LSTM
        
        #Initialise RNN
        regressor=Sequential()
        
        #Add first LSTM layer
        regressor.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
        #units=no. of neurons in layer
        #input_shape=(timesteps,no. of cols/features)
        #return_seq=True for sending recc memory. For last layer, retrun_seq=False since end of the line
        regressor.add(Dropout(0.1))
        
        #Add 2nd LSTM layer
        regressor.add(LSTM(units=50,return_sequences=True))
        regressor.add(Dropout(0.1))
        
        #Add 3rd LSTM layer
        regressor.add(LSTM(units=50,return_sequences=True))
        regressor.add(Dropout(0.1))
        
        #Add 4th LSTM layer
        regressor.add(LSTM(units=50))
        regressor.add(Dropout(0.1))
        
        #Add o/p layer
        regressor.add(Dense(units=1))
        
        #Compile
        regressor.compile(optimizer='adam',loss='mean_squared_error')
        
        #Training
        regressor.fit(X_train,y_train,epochs=25,batch_size=32 )
        #For lstm, batch_size=power of 2
        
        #Testing
        ###dataset_test=pd.read_csv('Google_Stock_Price_Test.csv')
        real_stock_price=dataset_test.iloc[:,4:5].values
        
        #To predict, we need stock prices of 7 days before the test set
        #So combine train and test set to get the entire data set
        dataset_total=pd.concat((dataset_train['Close'],dataset_test['Close']),axis=0) 
        testing_set=dataset_total[ len(dataset_total) -len(dataset_test) -7: ].values
        testing_set=testing_set.reshape(-1,1)
        #-1=till last row, (-1,1)=>(80,1). otherwise only (80,0)
        
        #Feature scaling
        testing_set=sc.transform(testing_set)
        
        #Create data structure
        X_test=[]
        for i in range(7,len(testing_set)):
            X_test.append(testing_set[i-7:i,0])
            #Convert list to numpy arrays
        X_test=np.array(X_test)
        
        #Reshaping: Adding 3rd dimension
        X_test=np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
        
        #Testing Prediction
        predicted_stock_price=regressor.predict(X_test)
        
        #Getting original prices back from scaled values
        predicted_stock_price=sc.inverse_transform(predicted_stock_price)
        fig = plt.figure(figsize=(7.2,4.8),dpi=65)
        plt.plot(real_stock_price,label='Actual Price')  
        plt.plot(predicted_stock_price,label='Predicted Price')
          
        plt.legend(loc=4)
        plt.savefig('static/LSTM.png')
        plt.close(fig)
        
        
        error_lstm = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
        
        
        #Forecasting Prediction
        forecasted_stock_price=regressor.predict(X_forecast)
        
        #Getting original prices back from scaled values
        forecasted_stock_price=sc.inverse_transform(forecasted_stock_price)
        
        lstm_pred=forecasted_stock_price[0,0]
        print()
        print("##############################################################################")
        print("Tomorrow's ",quote," Closing Price Prediction by LSTM: ",lstm_pred)
        print("LSTM RMSE:",error_lstm)
        print("##############################################################################")
        return lstm_pred,error_lstm

    st.write(LSTM_ALGO(df))
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
    
    
    
    
