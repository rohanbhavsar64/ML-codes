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
    python
elif h == 'PREDICATION':
    import yfinance as yf
    import matplotlib.pyplot as plt
    import streamlit as st
    import time
    import numpy as np
    import datetime
    import timezone
    from bytewax.dataflow import Dataflow
    from bytewax.inputs import ManualInputConfig, distribute
    from websocket import create_connection

    # Create a Streamlit app
    st.title("Real-Time Stock Price Chart")

    # Get the Apple stock data
    apple_stock = yf.Ticker("AAPL")

    # Create a Matplotlib figure
    fig, ax = plt.subplots()

    # Use Streamlit's pyplot function to display the plot
    plot = st.pyplot(fig)

    # Define the accumulator function
    def build_array():
        return np.empty((0,3))

    # Define the accumulator function
    def acc_values(np_array, ticker):
        return np.insert(np_array, 0, np.array((ticker.time, ticker.price, ticker.dayVolume)), 0)

    # Define the function to get the event time
    def get_event_time(ticker):
        return datetime.utcfromtimestamp(ticker.time/1000).replace(tzinfo=timezone.utc)

    # Configure the `fold_window` operator to use the event time
    cc = EventClockConfig(get_event_time, wait_for_system_duration=timedelta(seconds=10))

    # And a 1 minute tumbling window, that starts at the beginning of the minute
    start_at = datetime.now(timezone.utc)
    start_at = start_at - timedelta(seconds=start_at.second, microseconds=start_at.microsecond)
    wc = TumblingWindowConfig(start_at=start_at, length=timedelta(seconds=60))

    # Define the input source
    ticker_list = ['AMZN', 'MSFT']
    def yf_input(worker_tickers, state):
        ws = create_connection("wss://streamer.finance.yahoo.com/")
        ws.send(json.dumps({"subscribe": worker_tickers}))
        while True:
            message = ws.recv()
            yield (json.loads(message)['ticker'], json.loads(message))

    # Create a Bytewax dataflow
    flow = Dataflow()
    flow.input("input", ManualInputConfig(yf_input, ticker_list, distribute(ticker_list, 1)))

    # Infinite loop to fetch and update stock values
    while True:
        # Get the historical prices for Apple stock
        historical_prices = apple_stock.history(period='1d', interval='1m')

        # Get the latest price and time
        latest_price = historical_prices['Close'].iloc[-1]
        latest_time = historical_prices.index[-1].strftime('%H:%M:%S')

        # Clear the plot and plot the new data
        ax.clear()
        ax.plot(historical_prices.index, historical_prices['Close'], label='Stock Value')
        ax.set_xlabel('Time')
        ax.set_ylabel('Stock Value')
        ax.set_title('Apple Stock Value')
        ax.legend(loc='upper left')
        ax.tick_params(axis='x', rotation=45)

        # Update the plot in the Streamlit app
        plot.pyplot(fig)

        # Show the latest stock value in the app
        st.write(f"Latest Price ({latest_time}): {latest_price}")

        # Sleep for 1 minute before fetching new data
        time.sleep(60)
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
    
    
    
    
