import numpy as np
import pickle

import pandas as pd
import utils

def run():
  only_process_test_sites = True
  data_folder = utils.get_data_folder()
  sensor_folder = data_folder / 'sensor_data'
  device_id_path = data_folder / 'device_ids.pickle'
  try:
    with open(device_id_path, 'rb') as f:
      device_ids = pickle.load(f)
    print("Extracting segment meta data (2/2)")
  except:
    device_ids = None
    print("Extracting segment meta data (1/2)")
  device_ext = '_no_device' if device_ids is None else ''
  save_ext = '' if only_process_test_sites else '_all_sites'
  save_path = sensor_folder / ('meta' + save_ext + device_ext + '.csv')
  if save_path.is_file():
    return device_ids is None
  summary_path = data_folder / 'file_summary.csv'
  df = pd.read_csv(summary_path)
  leaderboard_types_path = data_folder / 'leaderboard_type.csv'
  leaderboard_types = pd.read_csv(leaderboard_types_path)
  test_type_mapping = {fn: t for (fn, t) in zip(
    leaderboard_types.fn, leaderboard_types['type'])}
  
  # Combine all the sub-trajectory meta data
  all_sub_trajectories = []
  for mode in ['test', 'train', 'valid']:
    print(mode)
    load_path = sensor_folder / (mode + save_ext + '.pickle')
    with open(load_path, 'rb') as f:
      combined_mode = pickle.load(f)
      
    for fn in combined_mode:
      t = combined_mode[fn]
      
      site = t['site']
      level = t['floor']
      text_level = df.text_level.values[np.where(
        (df.site_id == site) & (df.level == level))[0][0]]
      num_waypoints = t['num_waypoints']
      waypoint_times = t['waypoint_times']
      sub_durations = np.diff(waypoint_times)
      
      waypoint_segments = t['waypoint_segments']
      waypoint_times = t['waypoint_times']
      relative_movements = t['relative_waypoint_movement_1']
      for i in range(num_waypoints-1):
        segment_time = waypoint_segments[i].time.values
        sensor_time_diff = np.diff(segment_time)
        start_time_offset = segment_time[0] - waypoint_times[i] 
        end_time_offset = segment_time[-1] - waypoint_times[i+1] 
        mean_robust_sensor_time_diff = sensor_time_diff[
          (sensor_time_diff >= 19) & (sensor_time_diff <= 21)].mean()
        
        if mode == 'test':
          distance_covered = None
          test_type = test_type_mapping[fn]
          plot_time = df.first_last_wifi_time.values[
            np.where(df.fn.values == fn)[0][0]]
        else:
          distance_covered = np.sqrt((relative_movements[i]**2).sum())
          test_type = ''
          plot_time = waypoint_times[i]
        
        all_sub_trajectories.append({
          'mode': mode,
          'site': site,
          'level': level,
          'text_level': text_level,
          'fn': fn,
          'device_id': None if device_ids is None else device_ids[fn][0],
          'device_id_merged': None if device_ids is None else (
            device_ids[fn][2]),
          'test_type': test_type,
          
          'plot_time': plot_time,
          'start_time': waypoint_times[i],
          'end_time': waypoint_times[i+1],
          'sub_trajectory_id': i,
          'num_waypoints': num_waypoints,
          'duration': sub_durations[i],
          'num_obs': segment_time.size,
          'start_time_offset': start_time_offset,
          'end_time_offset': end_time_offset,
          'mean_sensor_time_diff': sensor_time_diff.mean(),
          'mean_robust_sensor_time_diff': mean_robust_sensor_time_diff,
          'min_sensor_time_diff': sensor_time_diff.min(),
          'max_sensor_time_diff': sensor_time_diff.max(),
          'distance_covered': distance_covered,
          })
        
  combined = pd.DataFrame(all_sub_trajectories)
  combined.to_csv(save_path, index=False)
  
  return device_ids is None

# 这段代码负责从室内导航数据集中提取段落元数据。以下是代码的逐步解释：

# 导入必要的库（numpy、pickle、pandas 和 utils）。

# 定义 run() 函数。

# 将 only_process_test_sites 标志设置为 True。

# 定义数据文件夹、传感器文件夹和设备ID文件的路径。

# 尝试从pickle文件中加载设备ID。如果不存在，将 device_ids 设置为 None。

# 根据是否有设备ID设置元数据文件的文件扩展名。

# 检查元数据文件是否已经存在。如果存在，则退出函数。

# 从数据文件夹中读取文件摘要和排行榜类型。

# 创建一个字典，将文件名映射到测试类型。

# 初始化一个空列表，用于存储所有子轨迹。

# 遍历所有模式（'test'，'train'，'valid'）并处理数据：
# a. 打印当前模式。
# b. 使用pickle加载组合模式数据。
# c. 遍历组合模式数据中的所有文件名。
# d. 提取轨迹信息，如站点、层次、航点数和航点时间。
# e. 计算子轨迹持续时间。
# f. 提取航点段、航点时间和相对移动。
# g. 遍历航点并计算元数据，如传感器时间差、开始和结束时间偏移以及平均鲁棒传感器时间差。
# h. 根据模式（测试或训练），为距离覆盖、测试类型和绘图时间设置适当的值。

# 将所有子轨迹合并到一个pandas DataFrame中。

# 将组合的元数据保存到CSV文件。

# 如果设备ID为None，则返回True；否则返回False。

# 代码主要目的是从室内导航数据集中提取和处理轨迹的元数据，并将这些元数据保存到CSV文件中。