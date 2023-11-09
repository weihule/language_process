import numpy as np
import pandas as pd
from pathlib import Path
import pickle

from utils import mkdirs


def power(predict_root, save_root, weight_root):
    # 定义起始和结束日期  
    start_date = '2023-10-01'  
    end_date = '2023-10-08' 

    # 定义时间间隔  
    interval = pd.Timedelta('15min')  
    
    # 使用date_range()函数生成时间序列  
    time_series = pd.date_range(start=start_date, end=end_date, freq=interval) 

    mask = ((time_series.time >= pd.to_datetime('00:00').time()) & (time_series.time <= pd.to_datetime('04:45').time())) |  \
           ((time_series.time >= pd.to_datetime('20:00').time()) & (time_series.time <= pd.to_datetime('23:45').time())) 
    filter_features = ['WEATHER2_IR', 'WEATHER1_IR', 'WEATHER2_TMP', 'WEATHER1_TMP']
    for i in range(1, 13, 1):
        pre_file = "FBS_" + str(i) + "_20231001-20231007_weather.csv"
        out_file = "FBS_" + str(i) + "_power_output.csv"
        out_weight = "FBS_" + str(i) + "_power_output"

        pre_path = Path(predict_root) / pre_file
        out_path = Path(save_root) / out_file
        weight_path = Path(weight_root) / out_weight

        # 加载模型
        loaded_model = pickle.load(open(weight_path, 'rb'))

        data_predict = pd.read_csv(pre_path)
        data_predict = data_predict[filter_features]
        predict_pred = loaded_model.predict(data_predict)
        
        df = pd.DataFrame({
            "TIMESTAMP": time_series,
            "POWER": predict_pred
        })
        df.loc[mask, 'POWER'] = 0

        df.to_excel(out_path, index=False)
        print(f"power {out_file} save success")


def energy(pre_path, save_root, weight_root):
    # 定义起始和结束日期  
    start_date = '2022-11-01'  
    end_date = '2023-10-31' 

    # 定义时间间隔  
    interval = pd.Timedelta('1day')  
    
    # 使用date_range()函数生成时间序列  
    time_series = pd.date_range(start=start_date, end=end_date, freq=interval)[::-1] 

    month_series = [11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    predict_data = pd.read_csv(pre_path).iloc[:, 1:]
    predict_data = predict_data.groupby("AREA_NO")

    for i in range(1, 13, 1):
        out_file = "FBS_" + str(i) + "_energy_output.csv"
        out_weight = "FBS_" + str(i) + "_energy_output"
        predict_sub_dis1 = predict_data.get_group(i)[["WEATHER1_IR", "WEATHER2_IR"]]
        out_path = Path(save_root) / out_file

        # 加载模型
        loaded_model = pickle.load(open(Path(weight_root)/out_weight, 'rb'))
        
        # 预测
        predict_pred = loaded_model.predict(predict_sub_dis1)
        
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

        pd.DataFrame(dic).to_excel(out_path, index=False)
        print(f"{out_file} save success")


def predict(model, predict_sub_data):
    data_predict = predict_sub_data[["WEATHER1_IR", "WEATHER2_IR"]]
    
    predict_pred = model.predict(data_predict)

    return predict_pred.tolist()


def run():
    predict_root_ = r"D:\Desktop\智慧能源专项赛-赛题二数据\天气预报数据\分布式功率预测天气预报数据-23年10月1日-10月7日"
    save_root_ = "./outs/分布式12个区域未来7天功率预测"
    # save_root_ = "./outs"
    save_weight_root_ = "./new_weights/dis1"
    mkdirs(save_root_)
    power(predict_root_, save_root_, save_weight_root_)

    pre_path_ = r"D:\Desktop\智慧能源专项赛-赛题二数据\天气预报数据\分布式发电量预测天气预报数据-22年11月-23年10月\FBS-SSRD-20221101-20231031.csv"
    save_root_ = "./outs/分布式12个区域未来12个月发电量预测"
    save_weight_root_ = "./new_weights/dis2"
    mkdirs(save_root_)
    energy(pre_path_, save_root_, save_weight_root_)


if __name__ == "__main__":
    run()
