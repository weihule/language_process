import paddlets
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from paddlets.datasets.repository import get_dataset, dataset_list
from paddlets.models.model_loader import load
from paddlets.models.forecasting import DeepARModel
from paddlets import TSDataset


def dis_run():
    distributed_root = "/home/8TDISK/weihule/data/智慧能源专项赛-赛题二数据/分布式历史数据"
    Centralized_root = "/home/8TDISK/weihule/data/智慧能源专项赛-赛题二数据/集中式历史数据"

    split_ratio = 0.8

    # 创建DeepAR模型
    model = DeepARModel(
        in_chunk_len=24,
        out_chunk_len=24,
        dropout=0.1,
        batch_size=32,
        max_epochs=40
    )

    observed_cov = ["WEATHER1_TMP", "WEATHER1_PRES", "WEATHER1_RAINFALL", "WEATHER1_TCC",
                    "WEATHER1_IR", "WEATHER1_WS", "WEATHER2_TMP", "WEATHER2_PRES",
                    "WEATHER2_RAINFALL", "WEATHER2_TCC", "WEATHER2_IR", "WEATHER2_WS"]

    for p in Path(distributed_root).iterdir():
        print(str(p))
        data = pd.read_csv(str(p))
        data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'])
        # data["TIMESTAMP"].apply(pd.to_datetime)

        # mask = ((data['TIMESTAMP'].dt.time < pd.to_datetime('20:00').time()) &
        #         (data['TIMESTAMP'].dt.time > pd.to_datetime('04:45').time()))
        # data = data[mask]
        target_dataset = TSDataset.load_from_dataframe(
            data,
            time_col="TIMESTAMP",
            target_cols="POWER",
            observed_cov_cols=observed_cov,
            freq='15T',
            fill_missing_dates=True,
            fillna_method='pre'  # max, min, avg, median, pre, back, zero
        )

        train_dataset, val_test_dataset = target_dataset.split(0.8)
        val_dataset, test_dataset = val_test_dataset.split(0.5)

        # 训练模型
        model.fit(train_tsdataset=train_dataset, valid_tsdataset=val_dataset)
        model.save("./weight/model")

        loaded_model = load("./weight/model")
        res = loaded_model.predict(val_dataset)
        print(res)
        break


if __name__ == "__main__":
    dis_run()




