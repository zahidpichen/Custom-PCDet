import glob
from pathlib import Path
import numpy as np
import torch
import logging
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict



class PCDet:
    def __init__(self, cfg_file='config/PartA2.yaml', 
                 data_path='frames', 
                 ckpt="model/PartA2.pth"):
        """
        PCDet class for point cloud detection
        
        Args:
            cfg_file: Path to the config YAML file
            data_path: Path to the point cloud data file or directory
            ckpt: Path to the pretrained model checkpoint
        """
        self.cfg_file = cfg_file
        self.data_path = data_path
        self.ckpt = ckpt
        self.ext = '.bin'  # Hard-coded extension as requested
        
        # Setup logger
        self.logger = self._create_logger()
        self.logger.info('-----------------PCDet Object Detection-------------------------')
        
        # Load config
        self._load_config()
        
        # Initialize model and dataset
        self.demo_dataset = None
        self.model = None
        self._init_model()
        
    def _create_logger(self):
        """Create a logger for the PCDet class"""
        logger = logging.getLogger("PCDet")
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def _load_config(self):
        """Load configuration from YAML file"""
        cfg_from_yaml_file(self.cfg_file, cfg)
    
    def _init_model(self):
        """Initialize the dataset and model"""
        # Initialize dataset
        self.demo_dataset = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG, 
            class_names=cfg.CLASS_NAMES, 
            training=False,
            root_path=Path(self.data_path), 
            ext=self.ext, 
            logger=self.logger
        )
        self.logger.info(f'Total number of samples: \t{len(self.demo_dataset)}')
        
        # Build and load model
        self.model = build_network(
            model_cfg=cfg.MODEL, 
            num_class=len(cfg.CLASS_NAMES), 
            dataset=self.demo_dataset
        )
        
        if self.ckpt:
            self.model.load_params_from_file(filename=self.ckpt, logger=self.logger, to_cpu=True)
            self.model.cuda()
            self.model.eval()
        else:
            self.logger.warning("No checkpoint provided. Model is initialized but not loaded with weights.")
    
    def detect(self):
        """
        Run object detection on the point cloud data
        
        Returns:
            list: List of detection results for each frame, where each result is a dictionary
                 containing labels, scores, and bounding boxes
        """
        if not self.model or not self.demo_dataset:
            self.logger.error("Model or dataset not initialized properly")
            return []
        
        detection_results = []
        
        with torch.no_grad():
            total_objects = 0
            
            for idx, data_dict in enumerate(self.demo_dataset):
                self.logger.info(f'Processing sample index: \t{idx + 1}')
                data_dict = self.demo_dataset.collate_batch([data_dict])
                load_data_to_gpu(data_dict)
                pred_dicts, _ = self.model.forward(data_dict)
                
                # Process predictions
                batch_objects = len(pred_dicts[0]['pred_labels'])
                total_objects += batch_objects
                
                # Create a structured result for this frame
                frame_result = {
                    'frame_id': idx,
                    'num_objects': batch_objects,
                    'objects': []
                }
                
                # Add each detected object
                for obj_idx, (label, score, box) in enumerate(zip(
                        pred_dicts[0]['pred_labels'].cpu().numpy(),
                        pred_dicts[0]['pred_scores'].cpu().numpy(),
                        pred_dicts[0]['pred_boxes'].cpu().numpy()
                    )):
                    obj_data = {
                        'label': int(label),
                        'label_name': cfg.CLASS_NAMES[int(label)] if int(label) < len(cfg.CLASS_NAMES) else f"Class_{label}",
                        'score': float(score),
                        'box': box.tolist()
                    }
                    frame_result['objects'].append(obj_data)
                
                detection_results.append(frame_result)
            
            self.logger.info(f'Total objects detected across all samples: {total_objects}')
        
        return detection_results


