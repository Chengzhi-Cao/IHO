from pprint import pprint

import pandas as pd

# from modules.metrics import compute_mlc
from modules2.metrics import compute_mlc

def main():
    res_path = "/data/chengzhicao/Medical_Report/R2Gen-main_my/results/iu_xray/res.csv"
    gts_path = "/data/chengzhicao/Medical_Report/R2Gen-main_my/results/iu_xray/gts.csv"
    res_data, gts_data = pd.read_csv(res_path), pd.read_csv(gts_path)
    res_data, gts_data = res_data.fillna(0), gts_data.fillna(0)

    label_set = res_data.columns[1:].tolist()
    res_data, gts_data = res_data.iloc[:, 1:].to_numpy(), gts_data.iloc[:, 1:].to_numpy()
    res_data[res_data == -1] = 0
    gts_data[gts_data == -1] = 0

    metrics = compute_mlc(gts_data, res_data, label_set)
    pprint(metrics)


if __name__ == '__main__':
    main()
