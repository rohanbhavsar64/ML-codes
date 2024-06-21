#t cv2

import pandas as pd
import requests
from bs4 import BeautifulSoup
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
j=['Analysis','Profile','Statistics']
h=st.radio('Field',j,horizontal=True)
if h=='Analysis':
    st.subheader('Analysis')
    col1,col2=st.columns(2)
    with col1:
        l = ['5d','1wk','1mo','3mo','1y','5y','max']
        i = st.selectbox('Period', l)
        df = yf.download(ticker, period=i)
    with col2:
        q=['Area','Line','Bar']
        g=st.radio('Chart- Type',q,horizontal=True)
    if g=='Area':
        if df['Close'][-1]<df['Close'][0]:
            fig=px.area(df,x=df.index,y='Close',color_discrete_sequence=["#9E4033"],title='Closing Price  vs Date')
            fig.update_yaxes(showgrid=False)
            st.write(fig)
        else:
            fig = px.area(df, x=df.index,y='Close', color_discrete_sequence=["#4A7230"],title='Closing Price vs Date')
            fig.update_yaxes(showgrid=False)
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
else:
    st.subheader('Statistics')
    MMM=yf.Ticker(ticker)
    df1=MMM.quarterly_balance_sheet.iloc[:10,:4]
    df2=MMM.quarterly_financials.iloc[44:,:]
    st.table(df1)
    st.table(df2)
    s=['Gross Profit','Cost Of Revenue','Total Revenue','Operating Revenue']
    m=st.selectbox('Fanancial',s)
    h=0
    if(m=='Gross Profit'):
        h=42
    elif(m=='Cost Of Revenue'):
        h=43
    elif(m=='Total Revanue'):
        h=44
    else:
        h=45
    f=['Yearly','quarterly']
    n=st.selectbox('Period',f)
    a1=0
    a2=0
    a3=0
    a4=0
    if(n=='Yearly'):
        a=MMM.financials.iloc[44:,:]
        b=a.columns
        a1=b[0]
        a2=b[1]
        a3=b[2]
        a4=b[3]
        
    else:
        a=MMM.quaterly_financials.iloc[44:,:] 
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
        c=MMM.quaterly_financials.iloc[h,:]
        b1=c[0]
        b2=c[1]
        b3=c[2]
        b4=c[3]
    data=[[a1,b1],[a2,b2],[a3,b3],[a4,b4]]
    df=pd.DataFrame(data,columns=['Year','Total Revanue'])
    st.write(px.bar(df,x='Year',y='Total Revanue'))
    
