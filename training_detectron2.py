# 기본설정:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# 필요한 라이브러리 import
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt

# 필요한 detectron2 utilities import
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, default_argument_parser, default_setup,launch
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.data.datasets import register_coco_instances

register_coco_instances("synthtext training", {}, "/home/xiyeon/ext_hdd/backup/jiyeon/vision_synthtext/synthtext/synthtext_train_30K.json", "/home/xiyeon/ext_hdd/backup/jiyeon/vision_synthtext/synthtext/train_images")
#학습 Dataset 등록
classes = MetadataCatalog.get("synthtext training")
dicts = DatasetCatalog.get("synthtext training")

from detectron2.engine import DefaultTrainer
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")) #yaml 파일 가져오기
cfg.DATASETS.TRAIN = ("synthtext training",) #dataset 등록
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # 초기 모델 설정
cfg.MODEL.WEIGHTS = "./output_1_50_50epoch/model_pretrained.pth" #1~50 images (50epoch) pretrained
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # 학습률 설정
cfg.SOLVER.MAX_ITER = 750000 # Iteration 설정(Detectron2에는 Epoch 지정 따로 X)
cfg.SOLVER.STEPS = [300000, 600000]
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # 본 데이터셋에 대한 속도, 성능 (default: 512)
cfg.SOLVER.CHECKPOINT_PERIOD = 20000
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 63  # class의 개수

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=True)
trainer.train() #학습 시작

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # 학습한 모델 불러오기
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # custom testing 임계값 설정
predictor = DefaultPredictor(cfg)

#register_coco_instances("icdar2013 test", {}, "./vision_homework/train_test1.json","./vision_homework/icdar2013_training") #icdar 2013 데이터
register_coco_instances("synthtext test", {}, "/home/xiyeon/ext_hdd/backup/jiyeon/vision_synthtext/synthtext/synthtext_test.json", "/home/xiyeon/ext_hdd/backup/jiyeon/vision_synthtext/synthtext/190_200")
# synth text 일부 데이터

classes_test = MetadataCatalog.get("synthtext test")
dicts_test = DatasetCatalog.get("synthtext test")


from detectron2.utils.visualizer import ColorMode
i = 1
for d in random.sample(dicts_test, 3):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=classes_test, 
                   scale=0.8
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite('./result'+str(i)+'.jpg',out.get_image()[:, :, ::-1]) #결과 이미지 저장
    i += 1

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("synthtext test", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "synthtext test")
print(inference_on_dataset(trainer.model, val_loader, evaluator)) #모델 평가