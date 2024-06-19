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
    url3 = f'https://groww.in/us-stocks/{ticker}'
    r2 = requests.get(url3, headers=headers)
    w2 = BeautifulSoup(r2.text, 'html')
    a1 = w2.find_all(class_='ustf141Head bodyLarge left-align')[0].text
    a2 = w2.find_all(class_='ustf141Head bodyLarge left-align')[1].text
    a3 = w2.find_all(class_='ustf141Head bodyLarge left-align')[2].text
    a4 = w2.find_all(class_='ustf141Head bodyLarge left-align')[3].text
    a5 = w2.find_all(class_='ustf141Head bodyLarge left-align')[4].text
    a6 = w2.find_all(class_='ustf141Head bodyLarge left-align')[5].text
    a7 = w2.find_all(class_='ustf141Head bodyLarge left-align')[6].text

    b1 = w2.find_all(class_='ustf141Value bodyLargeHeavy right-align')[0].text
    b2 = w2.find_all(class_='ustf141Value bodyLargeHeavy right-align')[1].text
    b3 = w2.find_all(class_='ustf141Value bodyLargeHeavy right-align')[2].text
    b4 = w2.find_all(class_='ustf141Value bodyLargeHeavy right-align')[3].text
    b5 = w2.find_all(class_='ustf141Value bodyLargeHeavy right-align')[4].text
    b6 = w2.find_all(class_='ustf141Value bodyLargeHeavy right-align')[5].text
    b7 = w2.find_all(class_='ustf141Value bodyLargeHeavy right-align')[6].text
    data = [[a1, b1], [a2, b2], [a3,b3], [a4, b4],[a5,b5],[a6,b6],[a7,b7]]
    df17 = pd.DataFrame(data, columns=['history', 'Est.'])
    st.table(df17)
    url4 = f'https://finance.yahoo.com/quote/{ticker}/financials'
    r3 = requests.get(url3, headers=headers)
    w3 = BeautifulSoup(r3.text, 'html')
    st.write(w3.find(class_='column svelte-1xjz32c alt'))

#<div class="D(ib) W(1/2) Bxz(bb) Pend(12px) Va(t) ie-7_D(i) smartphone_D(b) smartphone_W(100%) smartphone_Pend(0px) smartphone_BdY smartphone_Bdc($seperatorColor)" data-test="left-summary-table"><table class="W(100%)"><tbody><tr class="Bxz(bb) Bdbw(1px) Bdbs(s) Bdc($seperatorColor) H(36px) "><td class="C($primaryColor) W(51%)"><span>Previous Close</span></td><td class="Ta(end) Fw(600) Lh(14px)" data-test="PREV_CLOSE-value">107.87</td></tr><tr class="Bxz(bb) Bdbw(1px) Bdbs(s) Bdc($seperatorColor) H(36px) "><td class="C($primaryColor) W(51%)"><span>Open</span></td><td class="Ta(end) Fw(600) Lh(14px)" data-test="OPEN-value">107.60</td></tr><tr class="Bxz(bb) Bdbw(1px) Bdbs(s) Bdc($seperatorColor) H(36px) "><td class="C($primaryColor) W(51%)"><span>Bid</span></td><td class="Ta(end) Fw(600) Lh(14px)" data-test="BID-value">0.00 x 900</td></tr><tr class="Bxz(bb) Bdbw(1px) Bdbs(s) Bdc($seperatorColor) H(36px) "><td class="C($primaryColor) W(51%)"><span>Ask</span></td><td class="Ta(end) Fw(600) Lh(14px)" data-test="ASK-value">106.08 x 800</td></tr><tr class="Bxz(bb) Bdbw(1px) Bdbs(s) Bdc($seperatorColor) H(36px) "><td class="C($primaryColor) W(51%)"><span>Day's Range</span></td><td class="Ta(end) Fw(600) Lh(14px)" data-test="DAYS_RANGE-value">106.75 - 108.12</td></tr><tr class="Bxz(bb) Bdbw(1px) Bdbs(s) Bdc($seperatorColor) H(36px) "><td class="C($primaryColor) W(51%)"><span>52 Week Range</span></td><td class="Ta(end) Fw(600) Lh(14px)" data-test="FIFTY_TWO_WK_RANGE-value">85.35 - 113.14</td></tr><tr class="Bxz(bb) Bdbw(1px) Bdbs(s) Bdc($seperatorColor) H(36px) "><td class="C($primaryColor) W(51%)"><span>Volume</span></td><td class="Ta(end) Fw(600) Lh(14px)" data-test="TD_VOLUME-value"><fin-streamer data-symbol="MMM" data-field="regularMarketVolume" data-trend="none" data-pricehint="2" data-dfield="longFmt" value="4,449,245" active="">4,449,245</fin-streamer></td></tr><tr class="Bxz(bb) Bdbw(1px) Bdbs(s) Bdc($seperatorColor) H(36px) Bdbw(0)! "><td class="C($primaryColor) W(51%)"><span>Avg. Volume</span></td><td class="Ta(end) Fw(600) Lh(14px)" data-test="AVERAGE_VOLUME_3MONTH-value">4,952,249</td></tr></tbody></table></div>
##72C73C
#l-align: inherit;">-0.55</font></font></span></fin-streamer> <fin-streamer class="Fw(500) Pstart(8px) Fz(24px)" data-symbol="INFY" data-field="regularMarketChangePercent" data-trend="