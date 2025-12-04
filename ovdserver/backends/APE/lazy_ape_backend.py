from ..base import OVDBackend
import os
from PIL import Image
import base64
from io import BytesIO
from typing import List, Union

import ape
from collections import abc
import detectron2.data.transforms as T
from detectron2.data.detection_utils import _apply_exif_orientation, convert_PIL_to_numpy
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from ape.model_zoo import get_config_file
from huggingface_hub import hf_hub_download
from detectron2.config import LazyConfig

# lazy
from .shenyunhang.APE.demo.predictor_lazy import VisualizationDemo

aug = T.ResizeShortestEdge([1024, 1024], 1024)

class APEBackend(OVDBackend):
    model_names = [
        'APE_D',
        ]
    def __init__(self, model_name=None):
        self.model = None
        if model_name is None:
            self.model_name =  self.model_names[-1]
        else:
            self.load_model(model_name)

    def load_model(self, model_name: str):
        if model_name not in self.model_names:
            raise ValueError("Invalid model name")
        if self.model is not None and model_name == self.model_name:
            return
        use_xformers = os.getenv('OVDSERVER_USE_XFORMERS', 'False').lower() in ('true', '1', 't')
        if use_xformers:
            import importlib
            xformers_spec = importlib.util.find_spec("xformers")
            if xformers_spec is not None:
                use_xformers = True
                print(f"APEServer: Using xformers: {use_xformers}")
            else:
                use_xformers = False
                print("APEServer: xformers is not installed, please pip install xformers")
        else:
            print(f"APEServer: Not using xformers: {use_xformers}, set OVDSERVER_USE_XFORMERS=1 to enable if xformers is installed")
        if model_name == "APE_D":
            self.load_APE_D(use_xformers=use_xformers)
        self.model_name = model_name

    def detect(
            self,
            image: Image.Image,
            text_prompts: Union[str, List[str]],
            confidence_threshold: float = 0.05,
            iou_threshold: float = 1.0,
            ):
        if self.model is None:
            self.load_model(self.model_name)

        if isinstance(text_prompts, str):
            input_text = text_prompts.strip()
        elif isinstance(text_prompts, list):
            input_text = ', '.join([text_prompt.strip() for text_prompt in text_prompts])
        else:
            raise ValueError("Invalid text_prompts format")
        
        self.set_confidence_threshold(confidence_threshold)

        input_image = image.copy()

        ################## from demo/app.py ##################
        
        # work around this bug: https://github.com/python-pillow/Pillow/issues/3973
        input_image = _apply_exif_orientation(input_image)
        input_image = convert_PIL_to_numpy(input_image, format="BGR")

        if input_image.shape[0] > 1024 or input_image.shape[1] > 1024:
            global aug
            transform = aug.get_transform(input_image)
            input_image = transform.apply_image(input_image)
        else:
            transform = None

        predictions, visualized_output, _, metadata = self.model.run_on_image(
            input_image,
            text_prompt=input_text,
            with_box=True,
            with_mask=False,
            with_sseg=False,
        )

        output_image = visualized_output.get_image()
        # print("output_image", output_image.shape)
        # if input_format == "RGB":
        #     output_image = output_image[:, :, ::-1]
        if transform:
            output_image = transform.inverse().apply_image(output_image)
        # print("output_image", output_image.shape)

        output_image = Image.fromarray(output_image)
        buffered = BytesIO()
        output_image.save(buffered, format="PNG")
        visualization_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        
        json_results = instances_to_coco_json(predictions["instances"].to(self.model.cpu_device), 0)
        for json_result in json_results:
            json_result["category_name"] = metadata.thing_classes[json_result["category_id"]]
            del json_result["image_id"]
            
        return {"result": json_results, "visualization": visualization_base64}
    
    def set_confidence_threshold(self, confidence_threshold: float):
        if self.model is None:
            raise ValueError("Model is not loaded")
        self.model.predictor.model.model_vision.test_score_thresh = confidence_threshold
    
    def setup_cfg(self, config_file, args_opts, conf_threshold):
        '''
        修改自 demo/demo_lazy.py
        '''
        # load config from file and command-line arguments
        cfg = LazyConfig.load(config_file)
        cfg = LazyConfig.apply_overrides(cfg, args_opts)

        # copy from demo/demo_lazy.py
        if "output_dir" in cfg.model:
            cfg.model.output_dir = cfg.train.output_dir
        if "model_vision" in cfg.model and "output_dir" in cfg.model.model_vision:
            cfg.model.model_vision.output_dir = cfg.train.output_dir
        if "train" in cfg.dataloader:
            if isinstance(cfg.dataloader.train, abc.MutableSequence):
                for i in range(len(cfg.dataloader.train)):
                    if "output_dir" in cfg.dataloader.train[i].mapper:
                        cfg.dataloader.train[i].mapper.output_dir = cfg.train.output_dir
            else:
                if "output_dir" in cfg.dataloader.train.mapper:
                    cfg.dataloader.train.mapper.output_dir = cfg.train.output_dir

        if "model_vision" in cfg.model:
            cfg.model.model_vision.test_score_thresh = conf_threshold
        else:
            cfg.model.test_score_thresh = conf_threshold

        # default_setup(cfg, args)

        # setup_logger(name="ape")
        # setup_logger(name="timm")

        return cfg
    
    def load_APE_D(
            self,
            ckpt_repo_id: str = "shenyunhang/APE",
            running_device: str = "cuda",
            use_xformers: bool = False,
            ):
        # HF_ENDPOINT=https://hf-mirror.com
        # init_checkpoint= "output2/APE/configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO_GQA_PhraseCut_Flickr30k/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024_cp_16x4_1080k_mdl_20230829_162438/model_final.pth"
        init_checkpoint = "configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO_GQA_PhraseCut_Flickr30k/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024_cp_16x4_1080k_mdl_20230829_162438/model_final.pth"
        init_checkpoint = hf_hub_download(repo_id=ckpt_repo_id, filename=init_checkpoint)

        config_file = get_config_file(
            "LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO_GQA_PhraseCut_Flickr30k/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024_cp_16x4_1080k.py"
        )
        confidence_threshold = 0.01
        # args_opts = [
        #     "train.init_checkpoint='{}'".format(init_checkpoint),
        #     "model.model_language.cache_dir=''",
        #     "model.model_vision.select_box_nums_for_evaluation=500",
        #     "model.model_vision.text_feature_bank_reset=True",
        #     "model.model_vision.backbone.net.xattn=False",
        #     "model.model_vision.transformer.encoder.pytorch_attn=True",
        #     "model.model_vision.transformer.decoder.pytorch_attn=True",
        # ]
        args_opts = [
            f"train.init_checkpoint={init_checkpoint}",
            "model.model_language.cache_dir=''",
            "model.model_vision.select_box_nums_for_evaluation=500",
            "model.model_vision.text_feature_bank_reset=True",
        ]
        if not use_xformers:
            args_opts += [
                "model.model_vision.backbone.net.xattn=False",
                "model.model_vision.transformer.encoder.pytorch_attn=True",
                "model.model_vision.transformer.decoder.pytorch_attn=True",
            ]
        cfg = self.setup_cfg(config_file=config_file, args_opts=args_opts, conf_threshold=confidence_threshold)

        cfg.model.model_vision.criterion[0].use_fed_loss = False
        cfg.model.model_vision.criterion[2].use_fed_loss = False
        cfg.train.device = running_device

        ape.modeling.text.eva02_clip.factory._MODEL_CONFIGS[cfg.model.model_language.clip_model][
            "vision_cfg"
        ]["layers"] = 1

        demo = VisualizationDemo(cfg=cfg, args=None)
        demo.predictor.model.to(running_device)

        self.model = demo
        self.model_cfg = cfg
    