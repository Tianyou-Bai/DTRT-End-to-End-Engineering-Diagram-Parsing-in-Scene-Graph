# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
PID数据集处理 - 基于原始版本恢复，支持GraphML格式标注和COCO兼容性
"""
import os
import sys
import torch
import torchvision
import torchvision.transforms as T
import torchvision.datasets as datasets
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image, ImageFile
import networkx as nx
import random
import tempfile
import json
from pycocotools.coco import COCO
from .transforms import Compose, ToTensor, Normalize

# 允许加载被截断的图像
ImageFile.LOAD_TRUNCATED_IMAGES = True


class PIDGraphDatasetExtractor:
    """PID数据集提取器，处理GraphML格式的标注文件"""
    
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.data_info = []
        

    def extract(self):
        """提取所有配对的图像和标注文件"""
        paired_files = []
        for root, dirs, files in os.walk(self.base_dir):
            # 按文件名分组
            file_groups = {}
            
            for file in files:
                base_name = os.path.splitext(file)[0]
                ext = os.path.splitext(file)[1].lower()
                
                if base_name not in file_groups:
                    file_groups[base_name] = {}
                
                if ext in ['.png', '.jpg', '.jpeg']:
                    file_groups[base_name]['image'] = os.path.join(root, file)
                elif ext == '.graphml':
                    file_groups[base_name]['annotation'] = os.path.join(root, file)
            
            # 找到完整配对的文件
            for base_name, files_dict in file_groups.items():
                if 'image' in files_dict and 'annotation' in files_dict:
                    paired_files.append({
                        'id': base_name,
                        'image_path': files_dict['image'],
                        'annotation_path': files_dict['annotation'],
                        'relative_path': os.path.relpath(root, self.base_dir)
                    })

        print(f"在 {self.base_dir} 中找到 {len(paired_files)} 对配对文件")
        return paired_files
    
    def extract_annotations_from_graphml(self, graphml_path):
        """从 GraphML 文件中提取标注信息"""
        try:
            graph = nx.read_graphml(graphml_path)
            
            # 提取节点（对象）信息
            objects = []
            for node_id, node_attrs in graph.nodes(data=True):
                # 检查是否有边界框信息
                bbox_keys = ['xmin', 'ymin', 'xmax', 'ymax']
                if all(key in node_attrs for key in bbox_keys):
                    try:
                        bbox = [float(node_attrs[key]) for key in bbox_keys]
                        
                        label = node_attrs.get('label', 'unknown')
                        # 过滤背景类
                        if label == 'background':
                            continue
                            
                        obj_info = {
                            'id': node_id,
                            'bbox': bbox,  # [xmin, ymin, xmax, ymax]
                            'label': label,
                            'category': node_attrs.get('category', 'object'),
                            'attributes': {k: v for k, v in node_attrs.items() 
                                         if k not in bbox_keys + ['label', 'category']}
                        }
                        objects.append(obj_info)
                    except (ValueError, TypeError):
                        continue
            
            # 提取边（关系）信息
            relations = []
            for src, dst, edge_attrs in graph.edges(data=True):
                rel_info = {
                    'subject': src,
                    'object': dst,
                    'predicate': edge_attrs.get('edge_label', 'unknown'),
                    'attributes': dict(edge_attrs)
                }
                relations.append(rel_info)
            
            return {
                'objects': objects,
                'relations': relations,
                'num_objects': len(objects),
                'num_relations': len(relations)
            }
            
        except Exception as e:
            print(f"解析 GraphML 文件失败 {graphml_path}: {e}")
            return None


class PIDGraphDataset(Dataset):
    """PID图数据集类，支持GraphML格式的标注"""
    
    def __init__(self, complete_img=False, min_objects=1, transform=None, base_path='data/PID'):
        if complete_img == False:
            self.base_dir = os.path.join(base_path, 'patched')
        else:
            # 优先使用调整分辨率后的数据（允许递归查找，不强制要求顶层有graphml）
            resized_path = "/mnt/ShareDB_6TB/baitianyou/RelTR-main/data/pid_resized"
            if os.path.exists(resized_path):
                self.base_dir = resized_path
                print(f"🔧 使用调整分辨率后的数据: {resized_path}")
            # 其次使用提供的base_path（允许递归查找）
            elif os.path.exists(base_path):
                self.base_dir = base_path  # 直接使用包含数据的目录或其子目录
            else:
                self.base_dir = os.path.join(base_path, 'Complete')  # 使用子目录
            
        self.transform = transform
        
        # 1. 加载关系类别配置（无背景类别）
        self.rel_categories = ['solid',            # 0 - 实线连接
                               'non-solid',       # 1 - 虚线连接
                               ]
        
        # 2. 创建关系类别映射
        self.relation_label_to_idx = {rel: idx for idx, rel in enumerate(self.rel_categories)}
        self.idx_to_relation_label = {idx: rel for idx, rel in enumerate(self.rel_categories)}
        
        # 3. 加载对象类别（无背景类别）
        self.obj_categories = [
            'arrow',
            'connector',
            'crossing',
            'general',
            'instrumentation',
            'valve',
            # 'background' 被过滤
        ]
        
        self.label_to_idx = {
            obj: idx for idx, obj in enumerate(self.obj_categories)
        }
        
        self.idx_to_label = {
            idx: obj for idx, obj in enumerate(self.obj_categories)
        }
        
        # 检查数据目录是否存在
        if not os.path.exists(self.base_dir):
            print(f"⚠️ 警告: PID数据目录不存在 {self.base_dir}")
            print(f"   请确保数据路径正确，或数据已正确放置")
            self.valid_samples = []
            self._create_minimal_mappings()
            return
            
        # 提取数据集信息
        self.extractor = PIDGraphDatasetExtractor(self.base_dir)
        self.paired_files = self.extractor.extract()
        
        # 验证并过滤有效样本
        self.valid_samples = []
        print("验证样本...")
        
        for i, pair in enumerate(self.paired_files):
            annotations = self.extractor.extract_annotations_from_graphml(pair['annotation_path'])
            
            if annotations and annotations['num_objects'] >= min_objects:
                pair['annotations'] = annotations
                self.valid_samples.append(pair)
        
        print(f"\n数据集加载完成:")
        print(f"  原始配对: {len(self.paired_files)}")
        print(f"  有效样本: {len(self.valid_samples)}")
        
        if len(self.valid_samples) > 0:
            self._create_label_mappings()
        else:
            self._create_minimal_mappings()

    def _create_minimal_mappings(self):
        """创建最小标签映射（当没有有效样本时）"""
        self.object_classes = self.obj_categories
        self.relation_classes = self.rel_categories
        self.object_label_to_idx = self.label_to_idx
        self.idx_to_object_label = self.idx_to_label
        self.num_object_classes = len(self.object_classes)
        self.num_relation_classes = len(self.relation_classes)
        print(f"  使用默认类别映射")

    def _create_label_mappings(self):
        """创建标签映射"""
        all_object_labels = set()
        all_relation_labels = set()
        
        for sample in self.valid_samples:
            for obj in sample['annotations']['objects']:
                all_object_labels.add(obj['label'])
            for rel in sample['annotations']['relations']:
                all_relation_labels.add(rel['predicate'])
        
        # 对象标签映射（无背景类）
        self.object_classes = sorted(list(all_object_labels))
        self.object_label_to_idx = {label: idx for idx, label in enumerate(self.object_classes)}
        self.idx_to_object_label = {idx: label for label, idx in self.object_label_to_idx.items()}
        self.num_object_classes = len(self.object_classes)
        
        # 关系标签映射（无背景类）
        self.relation_classes = sorted(list(all_relation_labels))
        self.relation_label_to_idx = {label: idx for idx, label in enumerate(self.relation_classes)}
        self.idx_to_relation_label = {idx: label for label, idx in self.relation_label_to_idx.items()}
        self.num_relation_classes = len(self.relation_classes)
        
        print(f"  对象类别: {self.num_object_classes} 个")
        print(f"  关系类别: {self.num_relation_classes} 个")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        """返回图像、目标标注和样本信息"""
        if len(self.valid_samples) == 0:
            # 返回空样本
            dummy_image = Image.new('RGB', (800, 600), color=(128, 128, 128))
            if self.transform:
                dummy_image = self.transform(dummy_image)
            
            target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64),
                'image_id': torch.tensor([idx + 1], dtype=torch.int64),
                'orig_size': torch.tensor([600, 800]),
                'rel_annotations': torch.zeros((0, 3), dtype=torch.int64),
                'size': torch.tensor([600, 800]),
                'iscrowd': torch.zeros((0,), dtype=torch.int64)
            }
            return dummy_image, target
            
        sample = self.valid_samples[idx]
        
        # 1. 安全加载图像
        try:
            image = Image.open(sample['image_path']).convert('RGB')
        except Exception as e:
            print(f"⚠️ 警告: 无法加载图像 {sample['image_path']}: {e}")
            image = Image.new('RGB', (800, 600), color=(128, 128, 128))
            
        orig_size = torch.tensor([image.size[1], image.size[0]])  # (height, width)
        
        # 2. 处理标注
        annotations = sample['annotations']
        
        # 提取对象信息
        boxes = []
        labels = []
        object_ids = []
        
        for obj in annotations['objects']:
            boxes.append(obj['bbox'])  # [xmin, ymin, xmax, ymax]
            # 获取标签索引（跳过背景类，从1开始）
            label_idx = self.object_label_to_idx.get(obj['label'], 1)  # 默认为第一个真实类别
            labels.append(label_idx)
            object_ids.append(obj['id'])
        
        # 提取关系信息
        rel_annotations = []
        for rel in annotations['relations']:
            try:
                # 找到主体和客体在对象列表中的索引
                sub_idx = object_ids.index(rel['subject'])
                obj_idx = object_ids.index(rel['object'])
                rel_label = self.relation_label_to_idx.get(rel['predicate'], 1)
                rel_annotations.append([sub_idx, obj_idx, rel_label])
            except ValueError:
                # 跳过找不到对应对象的关系
                continue
        
        # 3. 转换为张量
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            # 处理没有对象的情况
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        
        if len(rel_annotations) > 0:
            rel_annotations = torch.as_tensor(rel_annotations, dtype=torch.int64)
        else:
            rel_annotations = torch.zeros((0, 3), dtype=torch.int64)
        
        # 4. 构建目标字典
        target = {
            'boxes': boxes,                    # [N, 4] 边界框
            'labels': labels,                  # [N] 对象标签
            'image_id': torch.tensor([idx + 1], dtype=torch.int64),   # [1] 图像ID
            'orig_size': orig_size,            # [2] 原始图像尺寸 [H, W]
            'rel_annotations': rel_annotations,  # [M, 3] - [subject_idx, object_idx, relation_label]
            'size': orig_size,              # [2] 图像尺寸 [H, W]
            'iscrowd': torch.zeros(len(boxes), dtype=torch.int64)
        }
        
        # 5. 应用图像变换
        if self.transform:
            image, target = self.transform(image, target)
        
        return image, target

    def get_dataset_info(self):
        """获取数据集信息"""
        return {
            'total_samples': len(self.valid_samples),
            'object_classes': self.object_classes,
            'relation_classes': self.relation_classes,
            'num_object_classes': self.num_object_classes,
            'num_relation_classes': self.num_relation_classes
        }


def create_datasets(complete_img=False, batch_size=2, split_ratio=0.8, base_path='data/PID', max_samples: int = None):
    """Create PID train/val datasets compatible with main.py usage.
    Returns (train_dataset, val_dataset).
    """
    # basic transforms: ToTensor + Normalize + box normalization inside Normalize
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ds = PIDGraphDataset(complete_img=complete_img, transform=transform, base_path=base_path)
    n = len(ds)
    if n == 0:
        print("⚠️ PIDGraphDataset has 0 samples; returning empty split.")
        return ds, ds
    import math
    indices = list(range(n))
    # Limit total samples for quick runs if requested
    if isinstance(max_samples, int) and max_samples > 0:
        max_samples = min(max_samples, n)
        indices = indices[:max_samples]
        n = len(indices)
    n_train = int(math.floor(n * split_ratio))
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    from torch.utils.data import Subset
    return Subset(ds, train_indices), Subset(ds, val_indices)


class CocoCompatibleDataset(torchvision.datasets.CocoDetection):
    """COCO 兼容的数据集包装类"""
    
    def __init__(self, original_dataset, indices=None):
        # 不调用父类的 __init__，因为我们要自定义行为
        self.original_dataset = original_dataset
        self.indices = indices if indices is not None else list(range(len(original_dataset)))
        
        # 🔑 创建 COCO API 对象
        self.coco = self._create_coco_api()
        
        # 复制原数据集的属性
        self.rel_categories = getattr(original_dataset, 'rel_categories', ['__background__'])
        self.obj_categories = getattr(original_dataset, 'obj_categories', ['__background__'])
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # 获取真实的索引
        real_idx = self.indices[idx]
        return self.original_dataset[real_idx]
    
    def _create_coco_api(self):
        """创建 COCO API 对象"""
        try:
            # 构建 COCO 格式数据
            images = []
            annotations = []
            categories = []
            
            # 获取类别信息
            obj_categories = getattr(self.original_dataset, 'obj_categories', ['__background__', 'object'])
            
            # 创建类别（跳过背景类）
            for i, cat_name in enumerate(obj_categories):
                if i == 0:  # 跳过背景类
                    continue
                categories.append({
                    "id": i,
                    "name": cat_name,
                    "supercategory": "thing"
                })
            
            if not categories:
                categories.append({"id": 1, "name": "object", "supercategory": "thing"})
            
            # 处理样本（限制数量避免内存问题）
            num_samples = min(len(self.indices), 50)
            ann_id = 1
            
            for i in range(num_samples):
                try:
                    real_idx = self.indices[i]
                    image, target = self.original_dataset[real_idx]
                    
                    # 图像信息
                    if 'orig_size' in target:
                        h, w = target['orig_size'].tolist()
                    else:
                        h, w = 800, 800
                    
                    image_info = {
                        "id": int(real_idx) + 1,
                        "width": w,
                        "height": h,
                        "file_name": f"image_{real_idx}.jpg"
                    }
                    images.append(image_info)
                    
                    # 标注信息
                    if 'boxes' in target and 'labels' in target:
                        boxes = target['boxes']
                        labels = target['labels']
                        
                        for box, label in zip(boxes, labels):
                            if len(box) == 4 and label.item() > 0:
                                x1, y1, x2, y2 = box.tolist()
                                annotation = {
                                    "id": ann_id,
                                    "image_id": int(real_idx) + 1,
                                    "category_id": int(label.item()),
                                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                                    "area": (x2 - x1) * (y2 - y1),
                                    "iscrowd": 0
                                }
                                annotations.append(annotation)
                                ann_id += 1
                except Exception as e:
                    print(f"处理样本 {i} 时出错: {e}")
                    continue
            
            # 确保至少有一个图像和标注
            if not images:
                images.append({"id": 1, "width": 800, "height": 800, "file_name": "dummy.jpg"})
            
            if not annotations:
                annotations.append({
                    "id": 1, "image_id": 1, "category_id": 1,
                    "bbox": [100, 100, 100, 100], "area": 10000, "iscrowd": 0
                })
            
            # 创建 COCO 数据
            coco_data = {
                "images": images,
                "annotations": annotations,
                "categories": categories,
                "info": {"description": "PID Dataset in COCO format", "version": "1.0"}
            }
            
            # 创建临时文件
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            json.dump(coco_data, temp_file, indent=2)
            temp_file.close()
            
            try:
                coco_api = COCO(temp_file.name)
                os.unlink(temp_file.name)
                print(f"✅ 创建 COCO API: {len(images)} 图像, {len(annotations)} 标注")
                return coco_api
            except Exception as e:
                print(f"❌ COCO API 创建失败: {e}")
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
                return self._create_minimal_coco_api()
                
        except Exception as e:
            print(f"❌ _create_coco_api 出错: {e}")
            return self._create_minimal_coco_api()
    
    def _create_minimal_coco_api(self):
        """创建最小的 COCO API"""
        dummy_data = {
            "images": [{"id": 1, "width": 800, "height": 800, "file_name": "dummy.jpg"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 100, 100], "area": 10000, "iscrowd": 0}],
            "categories": [{"id": 1, "name": "object", "supercategory": "thing"}]
        }
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(dummy_data, temp_file)
        temp_file.close()
        
        try:
            coco_api = COCO(temp_file.name)
            os.unlink(temp_file.name)
            return coco_api
        except:
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            return None


def create_transforms(train=True):
    """创建数据变换 - 使用固定尺寸确保边界框正确归一化"""
    import torch
    from .transforms import Compose, ToTensor, Normalize
    
    # 创建一个自定义的固定尺寸resize
    class FixedResize(object):
        def __init__(self, size):
            self.size = size  # (width, height)
        
        def __call__(self, image, target=None):
            from torchvision.transforms import functional as F
            w, h = image.size
            new_w, new_h = self.size
            
            # resize图像
            resized_image = F.resize(image, (new_h, new_w))
            
            if target is None:
                return resized_image, None
                
            # 计算缩放比例
            ratio_w = new_w / w
            ratio_h = new_h / h
            
            target = target.copy()
            if "boxes" in target:
                boxes = target["boxes"]
                # 缩放边界框坐标
                scaled_boxes = boxes * torch.as_tensor([ratio_w, ratio_h, ratio_w, ratio_h])
                target["boxes"] = scaled_boxes
                
            # 更新尺寸信息
            if "size" in target:
                target["size"] = torch.tensor([new_h, new_w])
            if "orig_size" in target:
                target["orig_size"] = torch.tensor([new_h, new_w])
                
            return resized_image, target
    
    normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    return Compose([
        FixedResize((1024, 1024)),  # 固定尺寸1024x1024
        ToTensor(),
        normalize,
    ])


def build_pid_dataset(image_set, args):
    """构建PID数据集"""
    # 获取数据路径
    if hasattr(args, 'pid_path'):
        base_path = args.pid_path
    else:
        base_path = 'data/PID'
    
    # 创建变换
    transform = create_transforms(train=(image_set == 'train'))
    
    # 创建原始数据集
    dataset = PIDGraphDataset(
        complete_img=True,   # 使用调整后的完整图像数据
        min_objects=1,
        transform=transform,
        base_path=base_path
    )
    
    # 如果没有有效样本，直接返回原数据集
    if len(dataset) == 0:
        print(f"⚠️ 警告: {image_set} 数据集为空")
        return dataset
    
    # 划分训练集和验证集
    total_size = len(dataset)
    if image_set == 'train':
        # 使用前80%作为训练集
        train_size = int(total_size * 0.8)
        indices = list(range(train_size))
    else:
        # 使用后20%作为验证集
        train_size = int(total_size * 0.8)
        indices = list(range(train_size, total_size))
    
    # 创建COCO兼容数据集
    coco_dataset = CocoCompatibleDataset(dataset, indices)
    
    print(f"✅ 构建 {image_set} 数据集: {len(coco_dataset)} 样本")
    
    return coco_dataset
