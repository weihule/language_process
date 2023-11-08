import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import xgboost as xgb
import matplotlib.pyplot as plt

from utils import mkdirs


def dis_run():

    # 定义起始和结束日期  
    start_date = '2022-11-01'  
    end_date = '2023-10-31' 

    # 定义时间间隔  
    interval = pd.Timedelta('1day')  
    
    # 使用date_range()函数生成时间序列  
    time_series = pd.date_range(start=start_date, end=end_date, freq=interval)[::-1] 
    
    save_root = "./outs/分布式12个区域未来12个月发电量预测"
    mkdirs(save_root)

    month_series = [11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    model_xgb = xgb.XGBRegressor()
    
    dis_path = r"D:\Desktop\智慧能源专项赛-赛题二数据\分布式历史发电量-202204-202210数据.xlsx"
    pre_path = r"D:\Desktop\智慧能源专项赛-赛题二数据\天气预报数据\分布式发电量预测天气预报数据-22年11月-23年10月\FBS-SSRD-20221101-20231031.csv"

    data = pd.read_excel(dis_path).iloc[:, :5]
    predict_data = pd.read_csv(pre_path).iloc[:, 1:]

    # 整体去除其中的Nan
    data = data.dropna()
    data = data.groupby("AREA_NO")

    predict_data = predict_data.groupby("AREA_NO")

    for i in range(1, 13, 1):
        out_file = "FBS_" + str(i) + "_energy_output.csv"
        out_path = Path(save_root) / out_file

        sub_dis1 = data.get_group(i)
        predict_sub_dis1 = predict_data.get_group(i)

        # 构造训练集和验证集
        power = sub_dis1["ENERGY（kW·h）"]
        features = sub_dis1[["WEATHER1_IR", "WEATHER2_IR"]]

        X_train, X_test, y_train, y_test = train_test_split(features, power, test_size=0.2, shuffle=False)

        # 开始训练模型
        model_xgb.fit(X_train, y_train)
        # val(model_xgb, X_test, y_test)
        predict_pred = predict(model_xgb, predict_sub_dis1)
        
        df = pd.DataFrame({
            "ENERGY_DATE": time_series,
            "ENERGY": predict_pred
        })

        dic = {
            "MONTH": ["22-Nov", "22-Dec", "23-Jan", "23-Feb", "23-Mar", "23-Apr",
                      "23-May", "23-Jun", "23-Jul", "23-Aug", "23-Sep", "23-Oct"],
            "ENERGY": []
        }

        # 依次找出22年11月份到23年10月份的数据和
        for m in month_series:
            month_mask = (time_series.month == m)

            # 根据月份掩码筛选数据
            filtered_data = df[month_mask]

            # 计算符合月份的能量总和
            energy_sum = filtered_data['ENERGY'].sum()
            energy_sum = int(energy_sum)
            dic["ENERGY"].append(energy_sum)

        # df.to_excel(out_path, index=False)
        
        break


def val(model: xgb.XGBRegressor, X_test, y_test):
    y_pred = model.predict(X_test)

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(0, len(y_pred), 1), y_test, color='blue', label='Original Data')
    plt.plot(np.arange(0, len(y_pred), 1), y_pred, color='red', linewidth=1, label='Predicted Data')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Comparison of Original Data and Predicted Data')
    plt.legend()
    plt.show()


def predict(model, predict_sub_data):
    data_predict = predict_sub_data[["WEATHER1_IR", "WEATHER2_IR"]]
    
    predict_pred = model.predict(data_predict)
    print(data_predict.shape)

    return predict_pred.tolist()


if __name__ == "__main__":
    dis_run()




