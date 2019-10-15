import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
ChinaBank = pd.read_csv('D:\chingeDownloaded\\translationDatas\\testData\ChinaBank.csv',index_col = 'Date',parse_dates=['Date'])

#ChinaBank.index = pd.to_datetime(ChinaBank.index)
sub = ChinaBank['2014-01':'2014-06']['Close']
train = sub.ix['2014-01':'2014-06']
# test = sub.ix['2014-04':'2014-06']
plt.figure(figsize=(10,10))
print(train)
plt.plot(train)
plt.show()


ChinaBank['Close_diff_1'] = ChinaBank['Close'].diff(1)
ChinaBank['Close_diff_2'] = ChinaBank['Close_diff_1'].diff(1)
fig = plt.figure(figsize=(20,6))
ax1 = fig.add_subplot(131)
ax1.plot(ChinaBank['Close'])
ax2 = fig.add_subplot(132)
ax2.plot(ChinaBank['Close_diff_1'])
ax3 = fig.add_subplot(133)
ax3.plot(ChinaBank['Close_diff_2'])
plt.show()

import statsmodels.api as sm

fig = plt.figure(figsize=(12, 8))

ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(train, lags=20, ax=ax1)
ax1.xaxis.set_ticks_position('bottom')
fig.tight_layout()

ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(train, lags=20, ax=ax2)
ax2.xaxis.set_ticks_position('bottom')
fig.tight_layout()
plt.show()


"""
热力图：
"""
train_results = sm.tsa.arma_order_select_ic(train, ic=['aic', 'bic'], trend='nc', max_ar=8, max_ma=8)

print('AIC', train_results.aic_min_order)
print('BIC', train_results.bic_min_order)

#模型检验->此处通过做残差的自相关函数图进行检验
model = sm.tsa.ARIMA(train, order=(1, 0, 0))
results = model.fit()
resid = results.resid #赋值
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40)
plt.show()

model = sm.tsa.ARIMA(sub, order=(1, 0, 0))
results = model.fit()
# predict_sunspots = results.predict(start=str('2014-04'),end=str('2014-05'),dynamic=False)
predict_sunspots = results.forecast(10)


print(predict_sunspots)
fig, ax = plt.subplots(figsize=(12, 8))
# ax = sub.plot(ax=ax)
# predict_sunspots.plot(ax=ax)
print(predict_sunspots)
plt.show()
