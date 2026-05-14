from ultralytics import YOLO  
import os
#from ultralytics.models.yolo.detect.train import ConsistencyTrainer  

model_config = "yolov8-bifpn.yaml"
#"yolov8-starnet-C2f-Star-LSCD.yaml"
#"yolov8n-ContextGuidedDown-HFPN-GEDH-t.yaml"
model = YOLO(model_config)
                                                                                                                                                       
# 从配置文件名中提取模型名称（去掉.yaml扩展名）                                                                                h  
model_name = os.path.splitext(model_config)[0]
                                                                                                                                
# 自动生成存储路径
project_path = f"experiment/{model_name}"
                                                                                                                                                       
# 训练模型
results = model.train(
    data="VisDrone.yaml",
    batch=10,
    #patience=0,  
    momentum=0.932,
    optimizer='SGD',
    epochs=300,
    lr0=0.04,
    lrf=0.01,
    imgsz=1280,
    device=[0],
    project=project_path,  # 使用自动生成的路径
    plots=True,
    patience=50,
    amp=False,
    #trainer=ConsistencyTrainer  
) 
