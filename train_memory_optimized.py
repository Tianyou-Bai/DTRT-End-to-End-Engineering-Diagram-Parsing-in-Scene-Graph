import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import gc
import os
import sys
import time
import logging
from contextlib import contextmanager

# 内存管理工具类
class MemoryManager:
    def __init__(self):
        self.peak_memory = 0
        
    def clear_cache(self):
        """清理GPU缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
    def get_memory_usage(self):
        """获取当前内存使用情况"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return {
                'allocated': allocated,
                'reserved': reserved,
                'total': total,
                'free': total - reserved
            }
        return None
        
    def log_memory(self, prefix=""):
        """记录内存使用情况"""
        memory = self.get_memory_usage()
        if memory:
            print(f"{prefix} 内存使用: 已分配={memory['allocated']:.2f}GB, "
                  f"已保留={memory['reserved']:.2f}GB, 可用={memory['free']:.2f}GB")
            self.peak_memory = max(self.peak_memory, memory['reserved'])
            
    @contextmanager
    def memory_checkpoint(self, name=""):
        """内存检查点上下文管理器"""
        self.log_memory(f"[{name}] 开始前")
        try:
            yield
        finally:
            self.clear_cache()
            self.log_memory(f"[{name}] 清理后")

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_memory_optimizations():
    """设置内存优化"""
    # 设置环境变量
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    # 禁用CUDA缓存分配器的内存池
    if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
        torch.cuda.set_per_process_memory_fraction(0.95)  # 限制使用95%的GPU内存
    
    # 设置数值精度以节省内存
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

def train_with_memory_management():
    """使用内存管理的训练函数"""
    memory_manager = MemoryManager()
    
    # 内存优化设置
    setup_memory_optimizations()
    
    # 导入必要的模块
    sys.path.append('/mnt/ShareDB_6TB/baitianyou/RelTR-main')
    
    with memory_manager.memory_checkpoint("初始化"):
        # 导入模型和数据集
        from models.reltrv3 import build_reltrv3
        from datasets.pid_datasets import build_pid_dataset
        import util.misc as utils
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        # 构建配置
        class Args:
            def __init__(self):
                # 模型配置
                self.backbone = 'resnet50'
                self.position_embedding = 'sine'
                self.enc_layers = 6
                self.dec_layers = 6
                self.dim_feedforward = 2048
                self.hidden_dim = 256
                self.dropout = 0.1
                self.nheads = 8
                self.num_entities = 1000  # 实体查询数量
                self.num_triplets = 2000  # 三元组查询数量
                self.pre_norm = False
                
                # 数据集配置
                self.dataset_file = 'pid'
                self.pid_path = '/mnt/ShareDB_6TB/baitianyou/RelTR-main/data/pid_resized/'
                
                # 训练配置
                self.lr = 1e-4
                self.lr_backbone = 1e-5
                self.batch_size = 1  # 使用小批次大小
                self.weight_decay = 1e-4
                self.epochs = 150
                self.lr_drop = 100
                self.clip_max_norm = 0.1
                
                # 损失权重
                self.eos_coef = 0.1
                self.rel_eos_coef = 0.1
                
                # 输出配置
                self.output_dir = './output_memory_optimized'
                self.save_every = 10
                
        args = Args()
        memory_manager.log_memory("配置完成")
        
    with memory_manager.memory_checkpoint("构建数据集"):
        # 构建数据集
        dataset_train = build_pid_dataset(image_set='train', args=args)
        dataset_val = build_pid_dataset(image_set='val', args=args)
        
        print(f"训练集大小: {len(dataset_train)}")
        print(f"验证集大小: {len(dataset_val)}")
        
        # 创建数据加载器 - 使用较小的batch_size和workers
        data_loader_train = DataLoader(
            dataset_train, 
            batch_size=args.batch_size,
            shuffle=True, 
            num_workers=2,  # 减少worker数量
            collate_fn=utils.collate_fn,
            pin_memory=False,  # 禁用pin_memory以节省内存
            drop_last=True
        )
        
        data_loader_val = DataLoader(
            dataset_val, 
            batch_size=args.batch_size,
            shuffle=False, 
            num_workers=2,
            collate_fn=utils.collate_fn,
            pin_memory=False,
            drop_last=False
        )
        
    with memory_manager.memory_checkpoint("构建模型"):
        # 构建模型
        model, criterion = build_reltrv3(args)
        model.to(device)
        
        # 计算模型参数数量
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'可训练参数数量: {n_parameters:,}')
        
        memory_manager.log_memory("模型加载完成")
        
        # 构建优化器
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
        
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("开始训练RelTRv3模型")
    print(f"总查询数量: {args.num_entities + args.num_triplets}")
    print(f"实体查询: {args.num_entities}, 三元组查询: {args.num_triplets}")
    print(f"批次大小: {args.batch_size}")
    print("="*60)
    
    # 训练循环
    for epoch in range(args.epochs):
        print(f"\n开始Epoch {epoch+1}/{args.epochs}")
        memory_manager.log_memory(f"Epoch {epoch+1} 开始")
        
        # 训练阶段
        model.train()
        criterion.train()
        
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, (samples, targets) in enumerate(data_loader_train):
            with memory_manager.memory_checkpoint(f"Batch {batch_idx+1}"):
                try:
                    # 移动数据到设备
                    samples = samples.to(device)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    
                    # 前向传播
                    outputs = model(samples)
                    loss_dict = criterion(outputs, targets)
                    
                    # 计算总损失
                    losses = sum(loss_dict[k] for k in loss_dict.keys())
                    
                    # 反向传播
                    optimizer.zero_grad()
                    losses.backward()
                    
                    # 梯度裁剪
                    if args.clip_max_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
                    
                    optimizer.step()
                    
                    epoch_loss += losses.item()
                    num_batches += 1
                    
                    # 打印进度
                    if batch_idx % 10 == 0:
                        print(f"  Batch {batch_idx+1}/{len(data_loader_train)}, "
                              f"Loss: {losses.item():.4f}")
                        memory_manager.log_memory(f"  Batch {batch_idx+1}")
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"CUDA内存不足在batch {batch_idx+1}: {str(e)}")
                        memory_manager.clear_cache()
                        continue
                    else:
                        raise e
                        
                # 每个batch后清理内存
                if batch_idx % 5 == 0:
                    memory_manager.clear_cache()
        
        # 更新学习率
        lr_scheduler.step()
        
        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1} 完成, 平均损失: {avg_loss:.4f}")
        memory_manager.log_memory(f"Epoch {epoch+1} 完成")
        
        # 保存检查点
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1:03d}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"检查点已保存: {checkpoint_path}")
            
        # 强制清理内存
        memory_manager.clear_cache()
    
    print(f"\n训练完成！峰值内存使用: {memory_manager.peak_memory:.2f}GB")

if __name__ == "__main__":
    # 清理初始内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    try:
        train_with_memory_management()
    except Exception as e:
        print(f"训练过程中出现错误: {str(e)}")
        # 清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        raise e
