#!/usr/bin/env python3
"""
训练进度监控脚本
"""

import json
import os
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def monitor_training():
    """监控训练进度"""
    log_file = "./output_optimized/training_log_optimized.jsonl"
    
    print("🔍 正在监控训练进度...")
    print("=" * 50)
    
    last_size = 0
    while True:
        try:
            if os.path.exists(log_file):
                current_size = os.path.getsize(log_file)
                if current_size > last_size:
                    # 读取训练日志
                    logs = []
                    with open(log_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                logs.append(json.loads(line))
                    
                    if logs:
                        latest = logs[-1]
                        print(f"\n📊 最新训练状态 (Epoch {latest['epoch']}):")
                        print(f"  🔥 训练损失: {latest['train_loss']:.4f}")
                        print(f"  🎯 验证损失: {latest['test_loss']:.4f}")
                        print(f"  📈 学习率: {latest['lr']:.2e}")
                        print(f"  ❌ 类别错误率: {latest.get('class_error', 0):.2f}%")
                        print(f"  🔗 关系错误率: {latest.get('rel_error', 0):.2f}%")
                        
                        # 绘制训练曲线
                        if len(logs) > 1:
                            plot_training_curves(logs)
                    
                    last_size = current_size
            else:
                print(f"⏳ 等待训练日志文件 {log_file} ...")
            
            time.sleep(30)  # 每30秒检查一次
            
        except KeyboardInterrupt:
            print("\n🛑 监控停止")
            break
        except Exception as e:
            print(f"❌ 监控错误: {e}")
            time.sleep(10)

def plot_training_curves(logs):
    """绘制训练曲线"""
    epochs = [log['epoch'] for log in logs]
    train_losses = [log['train_loss'] for log in logs]
    test_losses = [log['test_loss'] for log in logs]
    lrs = [log['lr'] for log in logs]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # 损失曲线
    ax1.plot(epochs, train_losses, 'b-', label='训练损失')
    ax1.plot(epochs, test_losses, 'r-', label='验证损失')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('损失')
    ax1.set_title('损失曲线 (优化训练)')
    ax1.legend()
    ax1.grid(True)
    
    # 学习率曲线
    ax2.plot(epochs, lrs, 'g-', label='学习率')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('学习率')
    ax2.set_title('学习率调度 (余弦退火)')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True)
    
    # 收敛速度分析
    if len(train_losses) > 5:
        recent_train_change = train_losses[-1] - train_losses[-5]
        recent_test_change = test_losses[-1] - test_losses[-5]
        
        ax3.bar(['训练损失变化', '验证损失变化'], 
                [recent_train_change, recent_test_change],
                color=['blue', 'red'])
        ax3.set_title('最近5个Epoch的损失变化')
        ax3.set_ylabel('损失变化')
        ax3.grid(True)
    
    # 优化效果对比
    if len(logs) > 10:
        early_avg = sum(train_losses[:10]) / 10
        recent_avg = sum(train_losses[-10:]) / 10
        improvement = (early_avg - recent_avg) / early_avg * 100
        
        ax4.bar(['训练改进效果'], [improvement], color='green')
        ax4.set_title('训练改进效果 (%)')
        ax4.set_ylabel('改进百分比')
        ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('./output_optimized/training_progress.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  📈 训练曲线已更新: ./output_optimized/training_progress.png")

if __name__ == "__main__":
    monitor_training()
