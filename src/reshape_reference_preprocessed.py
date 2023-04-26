import numpy as np
import os
import pandas as pd
import pathlib
from pathlib import Path
import pickle

import utils

FIELD_MAPPING = {
    "TYPE_ACCELEROMETER": "acce",
    "TYPE_ACCELEROMETER_UNCALIBRATED": "acce_uncali",
    "TYPE_GYROSCOPE": "gyro",
    "TYPE_GYROSCOPE_UNCALIBRATED": "gyro_uncali",
    "TYPE_MAGNETIC_FIELD": "magn",
    "TYPE_MAGNETIC_FIELD_UNCALIBRATED": "magn_uncali",
    "TYPE_ROTATION_VECTOR": "ahrs",
    "TYPE_WIFI": "wifi",
    "TYPE_BEACON": "ibeacon",
    "TYPE_WAYPOINT": "waypoint",
}

EXPECTED_NUM_COLUMNS = {
    "acce": 6,
    "acce_uncali": 9,
    "gyro": 6,
    "gyro_uncali": 9,
    "magn": 6,
    "magn_uncali": 9,
    "ahrs": 6,
    "wifi": 7,
    "ibeacon": 10,
    "waypoint": 4,
}

EXPECTED_MISSING_TYPE_COLUMNS = {
    "acce": 5,
    "acce_uncali": 5,
    "gyro": 5,
    "gyro_uncali": 5,
    "magn": 5,
    "magn_uncali": 5,
    "ahrs": 5,
    "wifi": 7,
    "ibeacon": 9,
    "waypoint": 4,
}

COL_NAMES = {
    "acce": ["x_acce", "y_acce", "z_acce", "a_acce"],
    "acce_uncali": [
        "x_acce_uncali",
        "y_acce_uncali",
        "z_acce_uncali",
        "x2_acce_uncali",
        "y2_acce_uncali",
        "z2_acce_uncali",
        "a_acc",
    ],
    "gyro": ["x_gyro", "y_gyro", "z_gyro", "a_gyro"],
    "gyro_uncali": [
        "x_gyro_uncali",
        "y_gyro_uncali",
        "z_gyro_uncali",
        "x2_gyro_uncali",
        "y2_gyro_uncali",
        "z2_gyro_uncali",
        "a_gyro_uncali",
    ],
    "magn": ["x_magn", "y_magn", "z_magn", "a_magn"],
    "magn_uncali": [
        "x_magn_uncali",
        "y_magn_uncali",
        "z_magn_uncali",
        "x2_magn_uncali",
        "y2_magn_uncali",
        "z2_magn_uncali",
        "a_magn_uncali",
    ],
    "ahrs": ["x_ahrs", "y_ahrs", "z_ahrs", "a_ahrs"],
    "wifi": [
        "t1_wifi",
        "ssid_wifi",
        "bssid_wifi",
        "rssid_wifi",
        "freq_wifi",
        "t2_wifi",
    ],
    "ibeacon": [
        "t1_beac",
        "id_beac_1",
        "id_beac_2",
        "id_beac_3",
        "power_beac",
        "rssi_beac",
        "dist_beac",
        "mac_beac",
        "t2_beac",
    ],
    "waypoint": ["time", "floor", "x_waypoint", "y_waypoint"],
}


def reshape_parquet(pickle_path, parquet_path, meta_info, file_id,
                    write_separate_wifi, data_folder, sample_sub_fns,
                    sample_sub_times):
  parquet_data = pd.read_parquet(parquet_path)
  types = parquet_data.column2

  all_vals = {}
  for k, v in FIELD_MAPPING.items():
    parquet_vals = parquet_data.iloc[(types == k).values].values
    if parquet_vals.size:
      if np.all(~((parquet_vals[0] == None) | pd.isnull(parquet_vals[0]))):
        num_non_None_cols = parquet_vals.shape[1]
      else:
        num_non_None_cols = np.argmax((parquet_vals[0] == None)
                                      | pd.isnull(parquet_vals[0]))
      all_vals[v] = parquet_vals[:, :num_non_None_cols]

      try:
        assert all_vals[v].shape[1] == EXPECTED_NUM_COLUMNS[v]
      except:
        # Append unknown values for the remaining columns
        assert all_vals[v].shape[1] == (EXPECTED_MISSING_TYPE_COLUMNS[v])
        num_padded_cols = EXPECTED_NUM_COLUMNS[v] - all_vals[v].shape[1]
        all_vals[v] = np.concatenate([
            all_vals[v], -999 * np.full(
                (all_vals[v].shape[0], num_padded_cols), np.nan)
        ], 1)

  # if file_id in [0, 240]:
  #   import pdb; pdb.set_trace()
  #   x=1

  # All values of 'acce', 'acce_uncali', 'gyro', 'gyro_uncali', 'magn',
  # 'magn_uncali' and 'ahrs' should have the same number of rows.
  # Combine the values into a single dataframe (same time index)
  shared_time_vals = {"time": all_vals["acce"][:, 0].astype(np.int64)}
  for k in [
      "acce",
      "acce_uncali",
      "gyro",
      "gyro_uncali",
      "magn",
      "magn_uncali",
      "ahrs",
  ]:
    try:
      assert all_vals[k].shape[0] == all_vals["acce"].shape[0]
    except:
      print(f"Failed {parquet_path}")
      return
    for n_id, n in enumerate(COL_NAMES[k]):
      shared_time_vals[n] = all_vals[k][:, n_id + 2].astype(np.float32)

  all_data = {"shared_time": pd.DataFrame(shared_time_vals)}

  if "wifi" in all_vals:
    wifi_df = pd.DataFrame({
        COL_NAMES["wifi"][0]: all_vals["wifi"][:, 0].astype(np.int64),
        COL_NAMES["wifi"][1]: all_vals["wifi"][:, 2].astype(str),
        COL_NAMES["wifi"][2]: all_vals["wifi"][:, 3].astype(str),
        COL_NAMES["wifi"][3]: all_vals["wifi"][:, 4].astype(np.int32),
        COL_NAMES["wifi"][4]: all_vals["wifi"][:, 5].astype(np.int32),
        COL_NAMES["wifi"][5]: all_vals["wifi"][:, 6].astype(np.int64),
    })
    
    wifi_last_times = wifi_df.groupby(
      't1_wifi')['t2_wifi'].transform("max").values
    wifi_df['most_recent_t2_wifi'] = wifi_last_times
    
    all_data["wifi"] = wifi_df

  if "ibeacon" in all_vals:
    if np.all(pd.isnull(all_vals["ibeacon"][:, 9])):
      t2_beac_values = all_vals["ibeacon"][:, 9].astype(np.float32)
    else:
      t2_beac_values = all_vals["ibeacon"][:, 9].astype(np.int64)

    ibeacon_df = pd.DataFrame({
        COL_NAMES["ibeacon"][0]: all_vals["ibeacon"][:, 0].astype(np.int64),
        COL_NAMES["ibeacon"][1]: all_vals["ibeacon"][:, 2].astype(str),
        COL_NAMES["ibeacon"][2]: all_vals["ibeacon"][:, 3].astype(str),
        COL_NAMES["ibeacon"][3]: all_vals["ibeacon"][:, 4].astype(str),
        COL_NAMES["ibeacon"][4]: all_vals["ibeacon"][:, 5].astype(np.int32),
        COL_NAMES["ibeacon"][5]: all_vals["ibeacon"][:, 6].astype(np.int32),
        COL_NAMES["ibeacon"][6]: all_vals["ibeacon"][:, 7].astype(np.float64),
        COL_NAMES["ibeacon"][7]: all_vals["ibeacon"][:, 8].astype(str),
        COL_NAMES["ibeacon"][8]: t2_beac_values,
    })
    all_data["ibeacon"] = ibeacon_df

  if "waypoint" in all_vals:
    waypoints_df = pd.DataFrame({
        COL_NAMES["waypoint"][0]: all_vals["waypoint"][:, 0].astype(np.int64),
        COL_NAMES["waypoint"][1]: meta_info.level,
        COL_NAMES["waypoint"][2]: all_vals["waypoint"][:, 2].astype(
          np.float64),
        COL_NAMES["waypoint"][3]: all_vals["waypoint"][:, 3].astype(
          np.float64),
    })
    all_data["waypoint"] = waypoints_df
    waypoint_times = waypoints_df.time.values
  else:
    sub_fn_ids = np.where(sample_sub_fns == meta_info.fn)[0]
    waypoint_times = sample_sub_times[sub_fn_ids]
  all_data["waypoint_times"] = waypoint_times

  if "wifi" in all_vals and meta_info["mode"] == "train":
    all_data["wifi_waypoints"] = utils.interpolate_wifi_trajectory(
        all_data, batch_interpolated=True, ignore_offset_s=2)

    if write_separate_wifi:
      wifi_storage_path = (
          data_folder / "wifi_features" /
          Path(meta_info.ext_path).with_suffix(".csv"))
      Path(wifi_storage_path).parent.mkdir(parents=True, exist_ok=True)
      all_data["wifi_waypoints"].to_csv(wifi_storage_path, index=False)

  with open(pickle_path, "wb") as handle:
    pickle.dump(all_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def run(only_process_test_sites=True, overwrite_existing_processed=False,
        write_separate_wifi=False):
  print("Reshaping raw data")
  data_folder = utils.get_data_folder()
  parquet_folder = data_folder / "reference_preprocessed"
  summary_path = data_folder / "file_summary.csv"
  df = pd.read_csv(summary_path)
  source_submission = 'submission_cost_minimization.csv'
  submission_folder = data_folder / 'submissions'
  submission = pd.read_csv(submission_folder / source_submission)
  sample_sub_fns = np.array(
      [sps.split('_')[1] for sps in (submission.site_path_timestamp)])
  sample_sub_times = np.array(
      [int(sps.split('_')[2]) for sps in (submission.site_path_timestamp)])
  
  # First check for the last file and abort if it already exists
  last_pickle_path = data_folder / (
      str(Path(df.ext_path.values[-1]).with_suffix("")) + "_reshaped.pickle")
  if last_pickle_path.exists() and (not overwrite_existing_processed):
    return
  
  # Loop over all file paths and compare the parquet and pickle files one by
  # one
  for i in range(df.shape[0]):
    # for i in np.arange(26924, 28000):
    print(f"{i+1} of {df.shape[0]}")
    if not only_process_test_sites or df.test_site[i]:
      ext_path = Path(df.ext_path[i])
      mode = ext_path.parts[0]
      pickle_path = data_folder / (
          str(ext_path.with_suffix("")) + "_reshaped.pickle")
      parquet_path = parquet_folder / mode / ext_path.with_suffix(
        ".parquet")
      pathlib.Path(parquet_path.parent).mkdir(parents=True, exist_ok=True)
      if not pickle_path.exists() or overwrite_existing_processed:
        reshape_parquet(
            pickle_path,
            parquet_path,
            df.iloc[i],
            i,
            write_separate_wifi,
            data_folder,
            sample_sub_fns,
            sample_sub_times,
        )
# 这部分代码的主要思路是将原始室内定位数据从Parquet格式转换为Pickle格式，并对数据进行预处理。该代码处理每个传感器数据（如加速度计、陀螺仪、磁场计、Wi-Fi和蓝牙信标等），并将它们整合到一个Pickle文件中。同时，该代码还支持对训练数据进行Wi-Fi轨迹插值操作。

# 首先，该代码定义了一个名为reshape_parquet的函数，用于处理单个Parquet文件。这个函数接受一个Parquet文件路径和其他参数，从Parquet文件中提取数据，将数据整理成所需的格式，并将结果保存为一个Pickle文件。具体地说，函数会处理共享时间戳的数据（如加速度计、陀螺仪、磁场计等），并将它们组合成一个单独的DataFrame。同时，Wi-Fi和蓝牙信标数据也会被提取并存储为单独的DataFrame。此外，如果存在航点数据，它们也会被提取并存储为一个DataFrame。

# 在处理完每个Parquet文件后，如果输入数据是训练数据，将会执行Wi-Fi轨迹插值。插值后的Wi-Fi轨迹可以选择保存为单独的CSV文件。最后，将所有处理后的数据保存为Pickle文件。

# run函数是整个脚本的主要入口，它会遍历所有的数据文件，为每个文件调用reshape_parquet函数。在处理数据之前，run函数会检查是否已存在处理后的数据文件，以避免重复处理。此外，还可以通过设置only_process_test_sites参数来仅处理测试数据集，以及通过设置overwrite_existing_processed参数来选择覆盖已经处理过的数据。

# 总之，这部分代码的主要思路是从原始Parquet格式数据中提取和预处理室内定位数据，并将处理后的数据保存为Pickle格式以便后续使用。同时，代码还支持对训练数据的Wi-Fi轨迹进行插值处理。