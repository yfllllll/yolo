import subprocess
import os
import time
from datetime import datetime

# ====================== 配置项 ======================
# 项目根目录（所有实验结果将保存在这里）
PROJECT_ROOT = "./uav_small_experiments"
# YOLOv8命令路径（如果yolo命令不在PATH中，改为完整路径，如"/home/user/miniconda3/bin/yolo"）
YOLO_CMD = "yolo"
# 训练超参数（与实验方案完全一致）
TRAIN_ARGS = {
    "batch": 32,
    "epochs": 300,
    "patience": 50,
    "optimizer": "SGD",
    "lr0": 0.04,
    "lrf": 0.01,
    "device": [0, 1, 2, 3],
    # "momentum": 0.937,
    # "weight_decay": 0.0005,
    # "warmup_epochs": 3,
    # "cos_lr": True,
    # "exist_ok": True,
    # "save": True,
    # "save_period": -1,  # 只保存最好的模型
    # "verbose": False
}

# ====================== 实验列表（已去重） ======================
# 格式：{"name": "实验名称", "model": "模型配置文件", "data": "数据集配置文件", "imgsz": 输入分辨率}
EXPERIMENTS = [
    # ---------------------- 基线实验 ----------------------
    {"name": "B1_yolov8n_640_visdrone", "model": "yolov8n.yaml", "data": "visdrone.yaml", "imgsz": 640},
    {"name": "B2_yolov8n_640_uavdt", "model": "yolov8n.yaml", "data": "uavdt.yaml", "imgsz": 640},
    {"name": "B3_yolov8n_1280_visdrone", "model": "yolov8n.yaml", "data": "visdrone.yaml", "imgsz": 1280},
    {"name": "B4_yolov8n_1280_uavdt", "model": "yolov8n.yaml", "data": "uavdt.yaml", "imgsz": 1280},
    
    # ---------------------- 核心消融实验 ----------------------
    {"name": "A2_yolov8-A_640_visdrone", "model": "yolov8-A.yaml", "data": "visdrone.yaml", "imgsz": 640},
    {"name": "A3_yolov8-A_1280_visdrone", "model": "yolov8-A.yaml", "data": "visdrone.yaml", "imgsz": 1280},
    {"name": "A4_yolov8-A_c2flight_1280_visdrone", "model": "yolov8-A_c2flight.yaml", "data": "visdrone.yaml", "imgsz": 1280},
    {"name": "A5_yolov8-final_1280_visdrone", "model": "yolov8-final_model.yaml", "data": "visdrone.yaml", "imgsz": 1280},
    {"name": "A6_yolov8-final_1280_uavdt", "model": "yolov8-final_model.yaml", "data": "uavdt.yaml", "imgsz": 1280},
    
    # ---------------------- 多分辨率对比实验 ----------------------
    {"name": "R1_yolov8-A_320_visdrone", "model": "yolov8-A.yaml", "data": "visdrone.yaml", "imgsz": 320},
    {"name": "R2_yolov8-A_640_visdrone", "model": "yolov8-A.yaml", "data": "visdrone.yaml", "imgsz": 640},
    {"name": "R3_yolov8-A_960_visdrone", "model": "yolov8-A.yaml", "data": "visdrone.yaml", "imgsz": 960},
    {"name": "R4_yolov8-A_1280_visdrone", "model": "yolov8-A.yaml", "data": "visdrone.yaml", "imgsz": 1280},
    {"name": "R5_yolov8-A_1600_visdrone", "model": "yolov8-A.yaml", "data": "visdrone.yaml", "imgsz": 1600},
    
    # ---------------------- SOTA对比实验 ----------------------
    {"name": "S1_yolov5n_640_visdrone", "model": "yolov5n.yaml", "data": "visdrone.yaml", "imgsz": 640},
    {"name": "S2_yolov8s_640_visdrone", "model": "yolov8s.yaml", "data": "visdrone.yaml", "imgsz": 640},
    {"name": "S3_yolov8m_640_visdrone", "model": "yolov8m.yaml", "data": "visdrone.yaml", "imgsz": 640},
]

# ====================== 主函数 ======================
def main():
    # 创建项目根目录
    os.makedirs(PROJECT_ROOT, exist_ok=True)
    
    # 记录开始时间
    start_time = time.time()
    print(f"========== 批量实验开始 ==========")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"总实验数: {len(EXPERIMENTS)}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("==================================\n")
    
    # 逐个运行实验
    for i, exp in enumerate(EXPERIMENTS):
        exp_name = exp["name"]
        print(f"[{i+1}/{len(EXPERIMENTS)}] 正在运行实验: {exp_name}")
        print(f"模型: {exp['model']}, 数据集: {exp['data']}, 分辨率: {exp['imgsz']}")
        
        # 构建训练命令
        cmd_parts = [
            YOLO_CMD, "detect", "train",
            f"model={exp['model']}",
            f"data={exp['data']}",
            f"imgsz={exp['imgsz']}",
            f"project={PROJECT_ROOT}",
            f"name={exp_name}"
        ]
        
        # 添加通用训练参数
        for k, v in TRAIN_ARGS.items():
            cmd_parts.append(f"{k}={v}")
        
        cmd = " ".join(cmd_parts)
        
        # 运行命令
        try:
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            print(f"实验 {exp_name} 完成 ✓")
        except subprocess.CalledProcessError as e:
            print(f"实验 {exp_name} 失败 ✗")
            print(f"错误信息: {e.stderr}")
            # 继续运行下一个实验，不中断
            continue
        
        print("-" * 50)
    
    # 记录结束时间
    end_time = time.time()
    total_hours = (end_time - start_time) / 3600
    
    print(f"\n========== 批量实验结束 ==========")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {total_hours:.2f} 小时")
    print(f"所有结果保存在: {PROJECT_ROOT}")
    print("==================================")

if __name__ == "__main__":
    main()