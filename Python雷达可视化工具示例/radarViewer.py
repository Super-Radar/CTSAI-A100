import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, BooleanVar, DoubleVar, IntVar
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import re
from datetime import datetime
import os
from collections import deque, defaultdict

# 第三方库依赖处理
try:
    from sklearn.cluster import DBSCAN
    from scipy.optimize import linear_sum_assignment
except ImportError:
    messagebox.showwarning("依赖缺失", "请安装依赖：pip install scikit-learn scipy")
    DBSCAN = None
    linear_sum_assignment = None

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------- 1. 数据解析器（兼容Raw/Tra格式） ----------------------
class RadarParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.frame_index = {}
        self.header_line = None
        self.all_lines = []
        self._scan_file()

    def _scan_file(self):
        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            self.all_lines = [line.rstrip('\n').rstrip(',') for line in f if line.strip()]

        for line in self.all_lines:
            if line.startswith('ObjId,') and 'range' in line:
                self.header_line = line.split(',')
                break
        if self.header_line is None:
            raise ValueError("未找到有效数据表头，文件格式不支持")

        current_frame = None
        current_timestamp = None
        frame_start = None
        data_count = 0

        for line_idx, line in enumerate(self.all_lines):
            if line.startswith('START,'):
                if current_frame is not None and frame_start is not None:
                    self.frame_index[current_frame] = {
                        'start': frame_start,
                        'end': line_idx - 1,
                        'ts': current_timestamp,
                        'count': data_count
                    }
                m = re.search(r'FrameNb:(\d+)', line)
                current_frame = int(m.group(1)) if m else len(self.frame_index)
                current_timestamp = None
                frame_start = line_idx + 1
                data_count = 0
            elif '---->' in line:
                ts_str = line.replace('---->', '').strip()
                try:
                    current_timestamp = datetime.strptime(ts_str, '%Y/%m/%d %H:%M:%S:%f')
                except:
                    current_timestamp = ts_str
            elif line.split(',')[0].isdigit():
                data_count += 1
            elif line.startswith('END,'):
                if current_frame is not None and frame_start is not None:
                    self.frame_index[current_frame] = {
                        'start': frame_start,
                        'end': line_idx - 1,
                        'ts': current_timestamp,
                        'count': data_count
                    }

        if not self.frame_index:
            df = pd.read_csv(self.file_path, on_bad_lines='skip')
            if 'FrameNb' not in df.columns:
                df['FrameNb'] = 0
            if 'Timestamp' not in df.columns:
                df['Timestamp'] = datetime.now()
            for fnb in df['FrameNb'].unique():
                self.frame_index[fnb] = {
                    'ts': df[df['FrameNb']==fnb]['Timestamp'].iloc[0],
                    'count': len(df[df['FrameNb']==fnb])
                }
            self.df_all = df

    def get_frame(self, fnb):
        if fnb not in self.frame_index:
            return pd.DataFrame()
        
        if hasattr(self, 'df_all'):
            df = self.df_all[self.df_all['FrameNb'] == fnb].copy()
        else:
            info = self.frame_index[fnb]
            start, end = info['start'], info['end']
            rows = []
            header_len = len(self.header_line)
            for idx in range(start, end+1):
                line = self.all_lines[idx]
                if line.split(',')[0].isdigit():
                    parts = line.split(',')
                    if len(parts) < header_len:
                        parts += [''] * (header_len - len(parts))
                    elif len(parts) > header_len:
                        parts = parts[:header_len]
                    rows.append(parts)
            df = pd.DataFrame(rows, columns=self.header_line)
            df['FrameNb'] = fnb
            df['Timestamp'] = info['ts']

        numeric_cols = ['ObjId', 'range', 'speed', 'angle', 'RCS', 'snr', 'X', 'Y']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df = df.dropna(subset=['ObjId']).reset_index(drop=True)
        df = df[df['range'] >= 0].reset_index(drop=True)

        if 'angle' in df.columns and 'range' in df.columns:
            df['angle_rad'] = np.deg2rad(df['angle'])
            df['X'] = df['range'] * np.cos(df['angle_rad'])
            df['Y'] = df['range'] * np.sin(df['angle_rad'])
        df['X'] = df['X'].fillna(0)
        df['Y'] = df['Y'].fillna(0)

        return df

# ---------------------- 2. 目标筛选器 ----------------------
class RadarFilter:
    def __init__(self):
        # 筛选条件默认值
        self.min_range = DoubleVar(value=0)
        self.max_range = DoubleVar(value=200)
        self.min_speed = DoubleVar(value=-100)
        self.max_speed = DoubleVar(value=100)
        self.min_rcs = DoubleVar(value=-100)
        self.max_rcs = DoubleVar(value=100)
        self.min_angle = DoubleVar(value=-180)
        self.max_angle = DoubleVar(value=180)
        self.filter_static = BooleanVar(value=True)
        self.static_speed_thresh = DoubleVar(value=0.5)
        self.enable_filter = BooleanVar(value=True)

    def apply(self, df):
        if not self.enable_filter.get() or df.empty:
            return df.copy()
        
        filtered = df.copy()
        # 距离筛选
        filtered = filtered[(filtered['range'] >= self.min_range.get()) & (filtered['range'] <= self.max_range.get())]
        # 速度筛选
        if 'speed' in filtered.columns:
            filtered = filtered[(filtered['speed'] >= self.min_speed.get()) & (filtered['speed'] <= self.max_speed.get())]
        # RCS筛选
        if 'RCS' in filtered.columns:
            filtered = filtered[(filtered['RCS'] >= self.min_rcs.get()) & (filtered['RCS'] <= self.max_rcs.get())]
        # 角度筛选
        if 'angle' in filtered.columns:
            filtered = filtered[(filtered['angle'] >= self.min_angle.get()) & (filtered['angle'] <= self.max_angle.get())]
        # 静态目标筛选
        if self.filter_static.get() and 'speed' in filtered.columns:
            filtered = filtered[abs(filtered['speed']) >= self.static_speed_thresh.get()]
        
        return filtered.reset_index(drop=True)

# ---------------------- 3. 卡尔曼目标跟踪器 ----------------------
class KalmanTracker:
    def __init__(self):
        self.enable_tracking = BooleanVar(value=True)
        self.match_thresh = DoubleVar(value=5.0)  # 匹配距离阈值
        self.max_lost_frames = IntVar(value=3)    # 最大丢失帧数
        self.next_id = 0
        self.tracked_targets = {}  # {track_id: {'state': 卡尔曼状态, 'lost_frames': 丢失帧数, 'history': 历史坐标}}
        self.dt = 0.1  # 帧间隔时间

    def _init_kalman(self, x, y, vx=0, vy=0):
        # 匀速运动模型：状态向量 [x, y, vx, vy]
        F = np.array([[1, 0, self.dt, 0],
                      [0, 1, 0, self.dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])  # 状态转移矩阵
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])  # 观测矩阵
        P = np.eye(4) * 1000  # 初始协方差矩阵
        R = np.eye(2) * 0.1  # 观测噪声协方差
        Q = np.eye(4) * 0.01  # 过程噪声协方差
        x0 = np.array([x, y, vx, vy])  # 初始状态
        return {'F': F, 'H': H, 'P': P, 'R': R, 'Q': Q, 'x': x0}

    def _predict(self, track):
        # 卡尔曼预测
        track['x'] = track['F'] @ track['x']
        track['P'] = track['F'] @ track['P'] @ track['F'].T + track['Q']
        return track

    def _update(self, track, z):
        # 卡尔曼更新
        y = z - track['H'] @ track['x']
        S = track['H'] @ track['P'] @ track['H'].T + track['R']
        K = track['P'] @ track['H'].T @ np.linalg.inv(S)
        track['x'] = track['x'] + K @ y
        track['P'] = (np.eye(4) - K @ track['H']) @ track['P']
        return track

    def track(self, df):
        if not self.enable_tracking.get() or df.empty:
            df['track_id'] = df['ObjId'] if 'ObjId' in df.columns else range(len(df))
            return df
        
        current_points = df[['X', 'Y']].values
        current_ids = []

        # 1. 预测所有已跟踪目标的位置
        for track_id in list(self.tracked_targets.keys()):
            self.tracked_targets[track_id] = self._predict(self.tracked_targets[track_id])
            self.tracked_targets[track_id]['lost_frames'] += 1

        # 2. 数据关联：匈牙利算法匹配当前点和跟踪目标
        if len(current_points) > 0 and len(self.tracked_targets) > 0:
            track_ids = list(self.tracked_targets.keys())
            predicted_points = np.array([self.tracked_targets[tid]['x'][:2] for tid in track_ids])
            # 计算距离矩阵
            dist_matrix = np.sqrt(((current_points[:, None] - predicted_points[None, :]) ** 2).sum(axis=2))
            # 匈牙利匹配
            row_ind, col_ind = linear_sum_assignment(dist_matrix)
            # 匹配结果处理
            matched_tracks = set()
            for r, c in zip(row_ind, col_ind):
                if dist_matrix[r, c] < self.match_thresh.get():
                    track_id = track_ids[c]
                    current_ids.append(track_id)
                    # 卡尔曼更新
                    self.tracked_targets[track_id] = self._update(self.tracked_targets[track_id], current_points[r])
                    self.tracked_targets[track_id]['lost_frames'] = 0
                    # 记录历史坐标（X/Y顺序不变，绘图时对调）
                    self.tracked_targets[track_id]['history'].append((current_points[r][0], current_points[r][1]))
                    matched_tracks.add(track_id)
            # 未匹配的当前点，新建跟踪
            for r in range(len(current_points)):
                if r not in row_ind or dist_matrix[r, col_ind[list(row_ind).index(r)]] >= self.match_thresh.get():
                    track_id = self.next_id
                    self.next_id += 1
                    current_ids.append(track_id)
                    x, y = current_points[r]
                    self.tracked_targets[track_id] = self._init_kalman(x, y)
                    self.tracked_targets[track_id]['lost_frames'] = 0
                    self.tracked_targets[track_id]['history'] = deque(maxlen=3000)
                    self.tracked_targets[track_id]['history'].append((x, y))
        else:
            # 无跟踪目标，全部新建
            for x, y in current_points:
                track_id = self.next_id
                self.next_id += 1
                current_ids.append(track_id)
                self.tracked_targets[track_id] = self._init_kalman(x, y)
                self.tracked_targets[track_id]['lost_frames'] = 0
                self.tracked_targets[track_id]['history'] = deque(maxlen=3000)
                self.tracked_targets[track_id]['history'].append((x, y))

        # 3. 清理丢失的跟踪目标
        for track_id in list(self.tracked_targets.keys()):
            if self.tracked_targets[track_id]['lost_frames'] > self.max_lost_frames.get():
                del self.tracked_targets[track_id]

        # 4. 给当前帧数据添加跟踪ID
        df['track_id'] = current_ids
        return df
    
    def reset(self):
        """重置跟踪器状态"""
        self.next_id = 0
        self.tracked_targets = {}

# ---------------------- 4. DBSCAN目标聚类器 ----------------------
class RadarCluster:
    def __init__(self):
        self.enable_clustering = BooleanVar(value=False)
        self.eps = DoubleVar(value=2.0)  # 聚类半径
        self.min_samples = IntVar(value=2)  # 最小样本数
        self.cluster_colors = plt.cm.get_cmap('tab10', 20)  # 聚类颜色映射

    def cluster(self, df):
        if not self.enable_clustering.get() or df.empty or DBSCAN is None:
            df['cluster_id'] = 0
            df['cluster_center_x'] = df['X']
            df['cluster_center_y'] = df['Y']
            df['cluster_size'] = 1
            return df
        
        points = df[['X', 'Y']].values
        # DBSCAN聚类
        db = DBSCAN(eps=self.eps.get(), min_samples=self.min_samples.get()).fit(points)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # 计算每个聚类的中心和大小
        cluster_info = defaultdict(dict)
        for label in set(labels):
            if label == -1:
                continue
            cluster_points = points[labels == label]
            center_x = cluster_points[:, 0].mean()
            center_y = cluster_points[:, 1].mean()
            cluster_info[label] = {
                'center_x': center_x,
                'center_y': center_y,
                'size': len(cluster_points)
            }

        # 给数据添加聚类信息
        df['cluster_id'] = labels
        df['cluster_center_x'] = df['cluster_id'].map(lambda x: cluster_info.get(x, {}).get('center_x', df['X']))
        df['cluster_center_y'] = df['cluster_id'].map(lambda x: cluster_info.get(x, {}).get('center_y', df['Y']))
        df['cluster_size'] = df['cluster_id'].map(lambda x: cluster_info.get(x, {}).get('size', 1))

        return df

# ---------------------- 5. 轨迹管理器 ----------------------
class TrajectoryManager:
    def __init__(self, tracker):
        self.tracker = tracker
        self.enable_trajectory = BooleanVar(value=False)
        self.max_history = IntVar(value=1000)  # 最大历史帧数
        self.line_width = DoubleVar(value=1.5)  # 轨迹线宽
        self.show_points = BooleanVar(value=False)  # 是否显示轨迹点

    def draw(self, ax):
        if not self.enable_trajectory.get() or not self.tracker.enable_tracking.get():
            return
        
        for track_id, track in self.tracker.tracked_targets.items():
            if len(track['history']) < 2:
                continue
            # 截取历史坐标
            history = list(track['history'])[-self.max_history.get():]
            # 关键修改1：轨迹绘制时对调X/Y
            x = [p[1] for p in history]  # 原Y
            y = [p[0] for p in history]  # 原X
            # 绘制轨迹线
            ax.plot(x, y, linewidth=self.line_width.get(), alpha=0.7, label=f'Track {track_id}')
            # 绘制轨迹点
            if self.show_points.get():
                ax.scatter(x, y, s=10, alpha=0.5)
    
    def reset(self):
        """清空所有轨迹"""
        for track in self.tracker.tracked_targets.values():
            track['history'].clear()

# ---------------------- 6. 主界面（整合所有功能） ----------------------
class RadarPlayer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("雷达目标可视化工具（增强版）")
        self.geometry("1920x1080")
        # 允许窗口缩放
        self.minsize(1200, 800)

        # 核心组件
        self.parser = None
        self.filter = RadarFilter()
        self.tracker = KalmanTracker()
        self.cluster = RadarCluster()
        self.trajectory = TrajectoryManager(self.tracker)

        # 播放控制变量
        self.fns = []
        self.current = 0
        self.playing = False
        self.timer = None

        # 绘图对象
        self.fig = None
        self.ax = None
        self.scatter = None
        self.origin = None
        self.cbar = None  # 保留colorbar引用
        self.id_texts = []
        self.cluster_centers = []

        # 先初始化界面，再打开文件
        self.setup_ui()
        # 打开初始文件
        self.open_file(initial=True)

    def open_file(self, initial=False):
        """打开CSV文件，initial=True表示初始化时打开"""
        path = filedialog.askopenfilename(filetypes=[("雷达CSV", "*.csv")])
        if not path:
            if initial:
                self.quit()
            return
        try:
            # 重置跟踪器和轨迹
            self.tracker.reset()
            self.trajectory.reset()
            
            self.parser = RadarParser(path)
            self.fns = sorted(self.parser.frame_index.keys())
            self.current = self.fns[0] if self.fns else 0
            
            # 更新界面显示当前文件路径
            file_name = os.path.basename(path)
            self.file_path_var.set(f"当前文件：{file_name}")
            
            # 重建绘图（会先清旧colorbar）
            self.init_plot()
            self.update_frame()

            messagebox.showinfo("打开成功", f"总帧数：{len(self.fns)}")
        except Exception as e:
            messagebox.showerror("错误", str(e))
            self.parser = None
            return

    def setup_ui(self):
        # 全局样式调整：增加默认间距
        style = ttk.Style()
        style.configure('TFrame', padding=2)
        style.configure('TButton', padding=2)
        style.configure('TLabel', padding=1)
        style.configure('TEntry', padding=1)

        # 顶部播放控制栏（改为可滚动，增加内边距）
        top_bar = ttk.Frame(self, padding=(10, 5))
        top_bar.pack(fill=tk.X, side=tk.TOP, pady=(5, 10))

        # 播放按钮区域
        btn_frame = ttk.Frame(top_bar)
        btn_frame.pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="播放/暂停", command=self.toggle, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="上一帧", command=self.prev, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="下一帧", command=self.next, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="首帧", command=lambda: self.jump(self.fns[0]) if self.fns else None, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="尾帧", command=lambda: self.jump(self.fns[-1]) if self.fns else None, width=6).pack(side=tk.LEFT, padx=2)
        # 新增：切换文件按钮
        ttk.Button(btn_frame, text="切换文件", command=self.open_file, width=10).pack(side=tk.LEFT, padx=2)

        # 速度设置区域
        speed_frame = ttk.Frame(top_bar)
        speed_frame.pack(side=tk.LEFT, padx=10)
        ttk.Label(speed_frame, text="播放速度:").pack(side=tk.LEFT)
        self.speed_var = IntVar(value=100)
        ttk.Combobox(speed_frame, textvariable=self.speed_var, values=[100,200,500,1000], width=5, state="readonly").pack(side=tk.LEFT, padx=3)

        # 帧号跳转区域
        jump_frame = ttk.Frame(top_bar)
        jump_frame.pack(side=tk.LEFT, padx=10)
        ttk.Label(jump_frame, text="当前帧:").pack(side=tk.LEFT)
        self.frame_var = tk.StringVar(value=str(self.current))
        frame_entry = ttk.Entry(jump_frame, textvariable=self.frame_var, width=8)
        frame_entry.pack(side=tk.LEFT, padx=3)
        frame_entry.bind("<Return>", self.on_enter)

        # 信息显示区域（自适应宽度）
        info_frame = ttk.Frame(top_bar)
        info_frame.pack(side=tk.LEFT, padx=20, fill=tk.X, expand=True)
        # 新增：文件路径显示
        self.file_path_var = tk.StringVar(value="当前文件：无")
        ttk.Label(info_frame, textvariable=self.file_path_var, font=("微软雅黑", 9)).pack(side=tk.TOP, anchor=tk.W)
        # 原有信息显示
        self.info_var = tk.StringVar(value="时间戳: - | 目标数: 0")
        ttk.Label(info_frame, textvariable=self.info_var, font=("微软雅黑", 10, "italic"), wraplength=400).pack(side=tk.TOP, anchor=tk.W)

        # 主内容区（使用PanedWindow实现可拖动调整大小）
        main_paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 左侧功能面板（可调整宽度，默认400）
        left_panel = ttk.Frame(main_paned, width=400)
        main_paned.add(left_panel, weight=1)  # weight=1 表示缩放比例

        # 左侧Notebook（增加内边距）
        notebook = ttk.Notebook(left_panel, padding=5)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 1. 筛选设置标签页（优化布局和间距）
        filter_tab = ttk.Frame(notebook, padding=10)
        notebook.add(filter_tab, text="筛选设置")
        self._setup_filter_tab(filter_tab)

        # 2. 跟踪设置标签页
        track_tab = ttk.Frame(notebook, padding=10)
        notebook.add(track_tab, text="跟踪设置")
        self._setup_track_tab(track_tab)

        # 3. 聚类设置标签页
        cluster_tab = ttk.Frame(notebook, padding=10)
        notebook.add(cluster_tab, text="聚类设置")
        self._setup_cluster_tab(cluster_tab)

        # 4. 轨迹设置标签页
        traj_tab = ttk.Frame(notebook, padding=10)
        notebook.add(traj_tab, text="轨迹设置")
        self._setup_traj_tab(traj_tab)

        # 中间+右侧面板（拆分为数据表格和绘图区的垂直Paned）
        right_paned = ttk.PanedWindow(main_paned, orient=tk.VERTICAL)
        main_paned.add(right_paned, weight=3)  # 占比更大

        # 中间数据表格（可调整高度）
        table_panel = ttk.Frame(right_paned, padding=5)
        right_paned.add(table_panel, weight=1)
        ttk.Label(table_panel, text="当前帧目标数据", font=("微软雅黑", 12, "bold")).pack(side=tk.TOP, pady=5)
        
        # 表格容器（带滚动条，自适应大小）
        table_container = ttk.Frame(table_panel)
        table_container.pack(fill=tk.BOTH, expand=True)
        
        self.tree = ttk.Treeview(table_container, show='headings', height=15)  # 减少默认高度
        tree_vscroll = ttk.Scrollbar(table_container, orient=tk.VERTICAL, command=self.tree.yview)
        tree_hscroll = ttk.Scrollbar(table_container, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=tree_vscroll.set, xscrollcommand=tree_hscroll.set)
        
        # 表格布局：滚动条自适应
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        tree_hscroll.pack(side=tk.BOTTOM, fill=tk.X)

        # 右侧绘图区
        plot_panel = ttk.Frame(right_paned, padding=5)
        right_paned.add(plot_panel, weight=2)  # 绘图区占比更高
        ttk.Label(plot_panel, text="雷达目标XY分布", font=("微软雅黑", 12, "bold")).pack(side=tk.TOP, pady=5)
        
        # 绘图容器（自适应大小）
        plot_container = ttk.Frame(plot_panel)
        plot_container.pack(fill=tk.BOTH, expand=True)

        # 初始化 fig/ax/canvas
        self.fig = plt.Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_container)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _setup_filter_tab(self, parent):
        # 筛选开关
        ttk.Checkbutton(parent, text="启用筛选", variable=self.filter.enable_filter).pack(anchor=tk.W, pady=(0, 8))

        # 距离筛选（优化网格布局，增加间距）
        range_frame = ttk.LabelFrame(parent, text="距离筛选 (m)", padding=8)
        range_frame.pack(fill=tk.X, pady=5)
        ttk.Label(range_frame, text="最小:").grid(row=0, column=0, padx=3, pady=3, sticky=tk.W)
        ttk.Entry(range_frame, textvariable=self.filter.min_range, width=10).grid(row=0, column=1, padx=3, pady=3)
        ttk.Label(range_frame, text="最大:").grid(row=0, column=2, padx=8, pady=3, sticky=tk.W)
        ttk.Entry(range_frame, textvariable=self.filter.max_range, width=10).grid(row=0, column=3, padx=3, pady=3)

        # 速度筛选
        speed_frame = ttk.LabelFrame(parent, text="速度筛选 (m/s)", padding=8)
        speed_frame.pack(fill=tk.X, pady=5)
        ttk.Label(speed_frame, text="最小:").grid(row=0, column=0, padx=3, pady=3, sticky=tk.W)
        ttk.Entry(speed_frame, textvariable=self.filter.min_speed, width=10).grid(row=0, column=1, padx=3, pady=3)
        ttk.Label(speed_frame, text="最大:").grid(row=0, column=2, padx=8, pady=3, sticky=tk.W)
        ttk.Entry(speed_frame, textvariable=self.filter.max_speed, width=10).grid(row=0, column=3, padx=3, pady=3)

        # RCS筛选
        rcs_frame = ttk.LabelFrame(parent, text="RCS筛选 (dBsm)", padding=8)
        rcs_frame.pack(fill=tk.X, pady=5)
        ttk.Label(rcs_frame, text="最小:").grid(row=0, column=0, padx=3, pady=3, sticky=tk.W)
        ttk.Entry(rcs_frame, textvariable=self.filter.min_rcs, width=10).grid(row=0, column=1, padx=3, pady=3)
        ttk.Label(rcs_frame, text="最大:").grid(row=0, column=2, padx=8, pady=3, sticky=tk.W)
        ttk.Entry(rcs_frame, textvariable=self.filter.max_rcs, width=10).grid(row=0, column=3, padx=3, pady=3)

        # 角度筛选
        angle_frame = ttk.LabelFrame(parent, text="角度筛选 (度)", padding=8)
        angle_frame.pack(fill=tk.X, pady=5)
        ttk.Label(angle_frame, text="最小:").grid(row=0, column=0, padx=3, pady=3, sticky=tk.W)
        ttk.Entry(angle_frame, textvariable=self.filter.min_angle, width=10).grid(row=0, column=1, padx=3, pady=3)
        ttk.Label(angle_frame, text="最大:").grid(row=0, column=2, padx=8, pady=3, sticky=tk.W)
        ttk.Entry(angle_frame, textvariable=self.filter.max_angle, width=10).grid(row=0, column=3, padx=3, pady=3)

        # 静态目标筛选
        static_frame = ttk.LabelFrame(parent, text="静态目标筛选", padding=8)
        static_frame.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(static_frame, text="过滤静态目标", variable=self.filter.filter_static).pack(anchor=tk.W, pady=3)
        ttk.Label(static_frame, text="静态速度阈值:").pack(anchor=tk.W, pady=3)
        ttk.Entry(static_frame, textvariable=self.filter.static_speed_thresh, width=10).pack(anchor=tk.W, pady=3)

        # 按钮区域（居中，增加间距）
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=10)
        ttk.Button(btn_frame, text="应用筛选", command=lambda: self.update_frame()).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(btn_frame, text="重置筛选", command=self._reset_filter).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

    def _reset_filter(self):
        self.filter.min_range.set(0)
        self.filter.max_range.set(200)
        self.filter.min_speed.set(-100)
        self.filter.max_speed.set(100)
        self.filter.min_rcs.set(-100)
        self.filter.max_rcs.set(100)
        self.filter.min_angle.set(-180)
        self.filter.max_angle.set(180)
        self.filter.filter_static.set(False)
        self.filter.static_speed_thresh.set(0.5)
        self.update_frame()

    def _setup_track_tab(self, parent):
        ttk.Checkbutton(parent, text="启用卡尔曼跟踪", variable=self.tracker.enable_tracking).pack(anchor=tk.W, pady=(0, 8))
        
        param_frame = ttk.LabelFrame(parent, text="跟踪参数", padding=8)
        param_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(param_frame, text="匹配距离阈值:").grid(row=0, column=0, padx=3, pady=3, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.tracker.match_thresh, width=10).grid(row=0, column=1, padx=3, pady=3)
        ttk.Label(param_frame, text="m").grid(row=0, column=2, padx=3, pady=3)
        
        ttk.Label(param_frame, text="最大丢失帧数:").grid(row=1, column=0, padx=3, pady=3, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.tracker.max_lost_frames, width=10).grid(row=1, column=1, padx=3, pady=3)
        ttk.Label(param_frame, text="帧").grid(row=1, column=2, padx=3, pady=3)

        ttk.Button(parent, text="重置跟踪器", command=self._reset_tracker).pack(fill=tk.X, pady=10)

    def _reset_tracker(self):
        self.tracker.reset()
        self.update_frame()

    def _setup_cluster_tab(self, parent):
        if DBSCAN is None:
            ttk.Label(parent, text="请安装依赖：pip install scikit-learn", foreground="red").pack(pady=10)
            return
        
        ttk.Checkbutton(parent, text="启用DBSCAN聚类", variable=self.cluster.enable_clustering).pack(anchor=tk.W, pady=(0, 8))
        
        param_frame = ttk.LabelFrame(parent, text="聚类参数", padding=8)
        param_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(param_frame, text="聚类半径:").grid(row=0, column=0, padx=3, pady=3, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.cluster.eps, width=10).grid(row=0, column=1, padx=3, pady=3)
        ttk.Label(param_frame, text="m").grid(row=0, column=2, padx=3, pady=3)
        
        ttk.Label(param_frame, text="最小样本数:").grid(row=1, column=0, padx=3, pady=3, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.cluster.min_samples, width=10).grid(row=1, column=1, padx=3, pady=3)

        ttk.Button(parent, text="应用聚类", command=lambda: self.update_frame()).pack(fill=tk.X, pady=10)

    def _setup_traj_tab(self, parent):
        ttk.Checkbutton(parent, text="启用轨迹绘制", variable=self.trajectory.enable_trajectory).pack(anchor=tk.W, pady=(0, 8))
        
        param_frame = ttk.LabelFrame(parent, text="轨迹参数", padding=8)
        param_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(param_frame, text="最大历史帧数:").grid(row=0, column=0, padx=3, pady=3, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.trajectory.max_history, width=10).grid(row=0, column=1, padx=3, pady=3)
        
        ttk.Label(param_frame, text="轨迹线宽:").grid(row=1, column=0, padx=3, pady=3, sticky=tk.W)
        ttk.Entry(param_frame, textvariable=self.trajectory.line_width, width=10).grid(row=1, column=1, padx=3, pady=3)
        
        ttk.Checkbutton(param_frame, text="显示轨迹点", variable=self.trajectory.show_points).grid(row=2, column=0, columnspan=2, pady=3, sticky=tk.W)

        ttk.Button(parent, text="清空轨迹", command=self._reset_trajectory).pack(fill=tk.X, pady=10)

    def _reset_trajectory(self):
        self.trajectory.reset()
        self.update_frame()

    def init_plot(self):
        # 【关键修复】先移除旧colorbar，防止越积越多
        if self.cbar is not None:
            try:
                self.cbar.remove()
            except Exception:
                pass
            self.cbar = None

        # 清空轴
        self.ax.clear()

        # 关键修改2：对调X/Y轴范围和标签
        self.ax.set_xlim(xmin=-50, xmax=60)  # 原Y轴范围
        self.ax.set_ylim(ymin=-10, ymax=120) # 原X轴范围
        self.ax.set_autoscale_on(False)
        self.ax.set_autoscalex_on(False)
        self.ax.set_autoscaley_on(False)
        # 关键修改3：对调坐标轴标签
        self.ax.set_xlabel('Y坐标 (m)', fontsize=10)
        self.ax.set_ylabel('X坐标 (m)', fontsize=10)
        self.ax.grid(True, alpha=0.3)
        self.title_text = self.ax.set_title(f"帧号: {self.current}", fontsize=12)
        # 绘制雷达原点
        self.origin = self.ax.scatter(0, 0, c='red', s=150, marker='o', zorder=10)
        # 初始化空散点
        self.scatter = self.ax.scatter([], [], c=[], cmap='viridis', s=80, alpha=0.8, zorder=5)
        # 只在无cbar时创建一次
        if self.cbar is None:
            self.cbar = self.fig.colorbar(self.scatter, ax=self.ax, label='目标距离 (m)')
        self.canvas.draw()

    def update_frame(self):
        if not self.parser or not self.fns:
            return
            
        # 1. 获取原始帧数据
        df_raw = self.parser.get_frame(self.current)
        info = self.parser.frame_index[self.current]

        # 2. 应用筛选
        df_filtered = self.filter.apply(df_raw)

        # 3. 应用目标跟踪
        df_tracked = self.tracker.track(df_filtered)

        # 4. 应用聚类
        df_clustered = self.cluster.cluster(df_tracked)

        # 最终使用的数据
        df = df_clustered

        # 更新信息显示
        ts = info['ts']
        if isinstance(ts, datetime):
            ts_str = ts.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        else:
            ts_str = str(ts)
        self.info_var.set(f"时间戳: {ts_str} | 原始目标数: {len(df_raw)} | 筛选后: {len(df_filtered)} | 跟踪ID: {len(self.tracker.tracked_targets)}")
        self.frame_var.set(str(self.current))

        # 更新数据表格
        self.tree.delete(*self.tree.get_children())
        display_cols = ['track_id', 'ObjId', 'range', 'speed', 'angle', 'X', 'Y', 'RCS', 'cluster_id', 'cluster_size']
        valid_cols = [c for c in display_cols if c in df.columns]
        self.tree['columns'] = valid_cols
        for c in valid_cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=70, anchor=tk.CENTER)
        if not df.empty and valid_cols:
            for _, row in df.iterrows():
                vals = []
                for c in valid_cols:
                    val = row[c]
                    vals.append(f"{val:.2f}" if isinstance(val, float) else str(val))
                self.tree.insert('', tk.END, values=vals)

        # ===================== 【终极修复】整图清空，彻底解决残影/重叠/发黑 =====================
        self.ax.clear()  # 🔥 这一句是根治！每帧彻底重画，不留下任何旧东西！
        self.cluster_centers.clear()
        self.id_texts.clear()

        # 恢复基础画布设置
        self.ax.set_xlim(-50, 60)
        self.ax.set_ylim(-10, 120)
        self.ax.grid(True)
        self.ax.scatter(0, 0, c='red', s=180, zorder=10)  # 雷达中心点

        # ===================== 散点 & 聚类绘制 =====================
        if not df.empty:
            x = df['Y'].values
            y = df['X'].values
            range_val = df['range'].values

            if self.cluster.enable_clustering.get():
                cluster_ids = df['cluster_id'].values
                self.scatter = self.ax.scatter(
                    x, y,
                    c=cluster_ids,
                    cmap=self.cluster.cluster_colors,
                    s=70,
                    zorder=2
                )
                # 绘制聚类中心（纯白，不发黑）
                for cid in df['cluster_id'].unique():
                    if cid == -1:
                        continue
                    c_df = df[df['cluster_id'] == cid]
                    cx = c_df['cluster_center_y'].iloc[0]
                    cy = c_df['cluster_center_x'].iloc[0]
                    # 干净白色聚类中心，无黑边
                    #self.ax.scatter(cx, cy, c='white', s=80, edgecolor='none', zorder=5)
                    #self.ax.text(cx + 1.0, cy + 1.0, f"C{cid}", fontsize=8, zorder=6, color='black')
            else:
                self.scatter = self.ax.scatter(
                    x, y,
                    c=range_val,
                    cmap='viridis',
                    s=70,
                    zorder=2
                )

            # ===================== 只画少量ID，绝不糊 =====================
            max_show = 10
            count = 0
            for _, row in df.iterrows():
                if count >= max_show:
                    break
                display_id = row['track_id'] if self.tracker.enable_tracking.get() else row['ObjId']
                self.ax.text(
                    row['Y'] + 0.4,
                    row['X'] + 0.4,
                    f"{int(display_id)}",
                    fontsize=7,
                    alpha=0.7,
                    zorder=3
                )
                count += 1

        # ===================== 轨迹绘制 =====================
        self.trajectory.draw(self.ax)

        # 标题 & 刷新
        self.ax.set_title(f"帧号: {self.current}")
        self.canvas.draw()
    # 播放控制方法
    def toggle(self):
        if not self.fns:
            return
        self.playing = not self.playing
        if self.playing:
            self.loop()

    def loop(self):
        if not self.playing or not self.fns:
            return
        idx = self.fns.index(self.current)
        if idx + 1 >= len(self.fns):
            self.playing = False
            return
        self.current = self.fns[idx+1]
        self.update_frame()
        delay = int(1000 / self.speed_var.get())
        self.timer = self.after(delay, self.loop)

    def prev(self):
        if not self.fns:
            return
        idx = self.fns.index(self.current)
        if idx > 0:
            self.current = self.fns[idx-1]
            self.update_frame()

    def next(self):
        if not self.fns:
            return
        idx = self.fns.index(self.current)
        if idx + 1 < len(self.fns):
            self.current = self.fns[idx+1]
            self.update_frame()

    def jump(self, fnb):
        if fnb in self.fns:
            self.current = fnb
            self.update_frame()

    def on_enter(self, event):
        try:
            fnb = int(self.frame_var.get())
            self.jump(fnb)
        except:
            self.frame_var.set(str(self.current))

    def on_close(self):
        if self.timer is not None:
            self.after_cancel(self.timer)
        # 退出前清理colorbar
        if self.cbar is not None:
            try:
                self.cbar.remove()
            except Exception:
                pass
        self.destroy()

if __name__ == "__main__":
    app = RadarPlayer()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()