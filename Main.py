# -*- coding: utf-8 -*-
import pandas as pd
from pandas import to_datetime
from datetime import datetime
import traceback

import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

res_output_count = 0

def c_statistic():
    loc = "D:\\"

    statistic_info = {}
    df = pd.read_csv(loc+"acc_08_final.csv", encoding="utf-8", usecols=[4, 5, 6, 7])
    df["TRADE_DATE_PARSED"] = to_datetime(df.TRADE_DATE, format="%Y-%m-%d-%H.%M.%S.%f")
    df["TRADE_DATE_PARSED"] = df["TRADE_DATE_PARSED"].apply(lambda x: datetime.strftime(x, "%Y/%m/%d"))
    df_dt_str = df["TRADE_DATE_PARSED"]
    # print("转换时间格式后：\n",df_dt_str)

    df1 = pd.read_csv(loc+"acc_09_final.csv", encoding="utf-8", usecols=[4, 5, 6, 7])
    df1["TRADE_DATE_PARSED"] = to_datetime(df1.TRADE_DATE, format="%Y-%m-%d-%H.%M.%S.%f")
    df1["TRADE_DATE_PARSED"] = df1["TRADE_DATE_PARSED"].apply(lambda x: datetime.strftime(x, "%Y/%m/%d"))

    df2 = pd.read_csv(loc+"acc_10_final.csv", encoding="utf-8", usecols=[4, 5, 6, 7])
    df2["TRADE_DATE_PARSED"] = to_datetime(df2.TRADE_DATE, format="%Y-%m-%d-%H.%M.%S.%f")
    df2["TRADE_DATE_PARSED"] = df2["TRADE_DATE_PARSED"].apply(lambda x: datetime.strftime(x, "%Y/%m/%d"))

    df3 = pd.read_csv(loc+"acc_11_final.csv", encoding="utf-8", usecols=[4, 5, 6, 7])
    df3["TRADE_DATE_PARSED"] = to_datetime(df3.TRADE_DATE, format="%Y-%m-%d-%H.%M.%S.%f")
    df3["TRADE_DATE_PARSED"] = df3["TRADE_DATE_PARSED"].apply(lambda x: datetime.strftime(x, "%Y/%m/%d"))

    df = pd.concat([df, df1], ignore_index=True)
    df = pd.concat([df,df2],ignore_index=True)
    df = pd.concat([df,df3],ignore_index=True)
    print(df)
    print("*" * 50)
    trade_address_info_ret = {}
    groups = df.groupby(df["TRADE_DATE_PARSED"])
    for group in groups:
        # 检测group行中TRADE_TYPE不等于21，22的值，并剔除
        df_group = group[1]
        # df_group = df_group[df_group.TRADE_TYPE == 21]
        df_group = df_group.query('TRADE_TYPE == 21 or TRADE_TYPE == 22')

        values_count = df_group["TRADE_ADDRESS"].value_counts()

        # print("TRADE_ADDRESS类型及人流量为：\n", values_count, "\n进站地点个数为：\n", values_count)
        trade_address_info = {}
        for index in range(len(values_count.values)):
            trade_address_info[values_count._index._data[index]] = values_count.values[index]
        # print(values_count.values)
        print("*" * 50)
        # print(values_count._index._data)
        # print("*"*50)
        statistic_info[group[0]] = trade_address_info
        trade_address_info_ret = trade_address_info

    print(statistic_info)
    return (trade_address_info_ret,statistic_info)
def draw_pic(trade_address_info,statistic_info):

    test_count = 1
    date_axis = []
    line_y_axis = []
    for key in statistic_info.keys():
        date_axis.append(key)
    index = 1
    is_excption = False
    for key in trade_address_info.keys():
        # if(key == 137):
        #     print("find 137, start breaking....")
        #     continue
        if(key in [137,155,141,159,147,121,143,125,133,145,139,153,149,131,129,123]):
            print("had done with this key")
            continue
        for key_inner in statistic_info.keys():
            try:
                if is_excption == True:
                    try:
                        ave = (statistic_info[key_inner][key] + line_y_axis[len(line_y_axis) - 1]) / 2  # 缺省值处理，取平均数替代
                    except:
                        ave = line_y_axis[len(line_y_axis)-1]
                        print("缺省值处理异常，用前一个值替补")
                    line_y_axis.append(ave)
                    is_excption = False
                    print("缺省值已由均值替代")
                line_y_axis.append(statistic_info[key_inner][key])
            except(Exception):
                print(traceback.format_exc())
                is_excption = True
                continue
        try:
            #画每条线路预测图：
            # col1_np = pd.Series(date_axis)
            # col2_np = pd.Series(line_y_axis)
            # print(col1_np)
            # print(col2_np)
            print(pd.Series(line_y_axis,date_axis))
            train_data = pd.Series(line_y_axis,date_axis)
            c_train(train_data,train_data,key)
            print("完成一次预测")
            # 画所有线路图
            if index % 5 == 0: index = 1
            ax = plt.subplot(int(str(41) + str(index)))
            index = index + 1
            ax.plot(date_axis, line_y_axis, label=key)
            plt.xticks(rotation=90)
            plt.legend()
            plt.subplots_adjust(hspace=0.9)
        except:
            print(traceback.format_exc())
            continue
        finally:
            line_y_axis = []
        # plt.show()
        currentDir = os.getcwd()  # 当前工作路径
        global  res_output_count
        plt.savefig("第"+str(res_output_count)+"次计算图1.png")
        #break
    # test_count = test_count + 1
    # plt.show()
    currentDir = os.getcwd()  # 当前工作路径

    plt.savefig("第" + str(res_output_count) + "次计算图2.png")
    # print("test_count is:",test_count)
    # if(test_count == 2):
    #     return


def c_train(train,sub,key):
    import statsmodels.api as sm
    if(key == 127):
        train[train > 30000] = 30000
        sub[sub > 30000] = 30000
    if(key == 135):
        train[train > 75000] = 75000
        sub[sub > 75000] = 75000
    if(key == 151):
        train[train > 20000] = 20000
        sub[sub > 20000] = 20000
    if(key == 157):
        train[train > 1000] = 1000
        sub[sub > 1000] = 1000
    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(train, lags=20, ax=ax1)
    ax1.xaxis.set_ticks_position('bottom')
    fig.tight_layout()

    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(train, lags=20, ax=ax2)
    ax2.xaxis.set_ticks_position('bottom')
    fig.tight_layout()
    # plt.show()
    currentDir = os.getcwd()  # 当前工作路径
    global res_output_count
    plt.savefig("第" + str(res_output_count) + "次计算图3.png")
    """
    热力图：
    """
    train_results = sm.tsa.arma_order_select_ic(train, ic=['aic', 'bic'], trend='nc', max_ar=8, max_ma=8)

    print('AIC', train_results.aic_min_order)
    print('BIC', train_results.bic_min_order)

    # 模型检验->此处通过做残差的自相关函数图进行检验
    # model = sm.tsa.ARIMA(train, order=(train_results.bic_min_order[0], 1, train_results.bic_min_order[1]))
    # results = model.fit()
    # resid = results.resid  # 赋值
    # fig = plt.figure(figsize=(12, 8))
    # fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40)
    # # plt.show()
    # currentDir = os.getcwd()  # 当前工作路径
    # plt.savefig("第" + str(res_output_count) + "次计算图4.png")

    model = sm.tsa.ARIMA(sub, order=(train_results.bic_min_order[0], 2, train_results.bic_min_order[1]))
    results = model.fit()
    # predict_sunspots = results.predict(start=str('2015-08'), end=str('2015-10'), dynamic=False)
    predict_sunspots = results.forecast(7)
    res_output_count = res_output_count + 1
    print("预测结果为：",predict_sunspots[0])
    fig, ax = plt.subplots(figsize=(12, 8))
    # ax = sub.plot(ax=ax)
    # predict_sunspots.plot(ax=ax)
    # print(predict_sunspots)

    c_writeout("第%d组数据，路线为：%d，预测结果为%s,BIC值为:%s"%(res_output_count,key,str(predict_sunspots[0]),train_results.bic_min_order))
    # plt.show()
    currentDir = os.getcwd()  # 当前工作路径
    # plt.savefig("第" + str(res_output_count) + "次计算图5.png")

def c_writeout(text):
    with open('dataRes.txt', 'a+') as f:
        f.write(text+"\n")
    f.close()

if __name__ == "__main__":
    (trade_address_info, statistic_info) = c_statistic()
    draw_pic(trade_address_info, statistic_info)
    pass
