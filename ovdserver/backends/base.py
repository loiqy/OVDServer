from abc import ABC, abstractmethod

class OVDBackend(ABC):
    model_name: str = None
    @abstractmethod
    def detect(
            self,
            image,
            text_prompts,
            confidence_threshold=0.1,
            iou_threshold=1.0,
        ):
        """检测图像中的目标。

        Args:
            image (PIL.Image): 输入图像。
            text_prompts (list[str]): 开放词汇提示词列表。
            confidence_threshold (float): 置信度阈值。
            iou_threshold (float): IOU 阈值。

        Returns:
            dict: 包含检测结果和标注图像的字典。
        """
        pass

    @abstractmethod
    def load_model(self, model_name):
        """加载模型。

        Args:
            model_name (str): 模型名称。
        """
        pass

    @classmethod
    def get_backend(cls, backend_name):
        """获取后端实例。

        Args:
            backend_name (str): 后端名称。

        Returns:
            OVDBackend: 后端实例。
        """
        if backend_name == "YOLOWorld":
            from .YOLOWorld.yoloworld_backend import YOLOWorldBackend
            return YOLOWorldBackend()
        elif backend_name == "APE":
            from .APE.lazy_ape_backend import APEBackend
            return APEBackend()
        elif backend_name == "OWLViT":
            from .OWLViT.owlvit_backend import OWLViTBackend
            return OWLViTBackend()
        elif backend_name == "GroundingDINO":
            from .GroundingDINO.groundingdino_backend import GroundingDINOBackend
            return GroundingDINOBackend()
        elif backend_name == "OWLv2":
            from .OWLv2.owlv2_backend import OWLv2Backend
            return OWLv2Backend()
        else:
            print(f"Invalid backend {backend_name}")
            return None