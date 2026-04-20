import json
import matplotlib.pyplot as plt
import numpy as np

def smooth_curve(y, window_size=10):
    """
    使用移动平均平滑曲线
    """
    if len(y) < window_size:
        return y

    # 使用numpy的卷积实现移动平均
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(y, kernel, mode='valid')

    # 为了保持原始长度，在开头补充原始数据
    padding = y[:window_size-1]
    return np.concatenate([padding, smoothed])

def plot_reward_curve(jsonl_file):
    """
    读取jsonl文件并绘制reward曲线
    """
    rewards = []
    steps = []
    
    # 读取jsonl文件
    total_lines = 0
    valid_lines = 0
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            total_lines += 1
            if total_lines > 275:
                break
            try:
                data = json.loads(line.strip())
                # 检查是否有reward字段且不为None
                if 'reward' in data and data['reward'] is not None:
                    reward_value = float(data['reward'])
                    rewards.append(reward_value)
                    # 如果有step字段就使用，否则使用行号
                    step = data.get('step', valid_lines + 1)
                    steps.append(step)
                    valid_lines += 1
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                print(f"跳过第 {i + 1} 行，错误: {e}")
                continue
    
    print(f"总行数: {total_lines}, 有效reward数据: {valid_lines}")
    
    if not rewards:
        print("没有找到有效的reward数据!")
        return
    
    # 转换为numpy数组便于处理
    rewards = np.array(rewards)
    steps = np.array(steps)
    
    # 绘制reward曲线
    plt.figure(figsize=(12, 8))
    
    # 原始曲线（半透明）
    plt.plot(steps, rewards, 'lightblue', linewidth=1, alpha=0.5, label='Original Reward')
    
    # 平滑曲线
    window_size = max(1, len(rewards) // 50)  # 动态调整窗口大小
    smoothed_rewards = smooth_curve(rewards, window_size)
    plt.plot(steps, smoothed_rewards, 'b-', linewidth=2, label=f'Smoothed Reward (window={window_size})')
    
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Training Reward Curve')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 添加统计信息 - reward通常是越高越好，所以显示最大值
    max_reward = max(rewards)
    max_step = steps[np.argmax(rewards)]
    plt.annotate(f'Max Reward: {max_reward:.4f}\nStep: {max_step}', 
                xy=(max_step, max_reward), 
                xytext=(10, 10), 
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.savefig('reward_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"最大reward: {max_reward:.6f} (step: {max_step})")
    print(f"最终reward: {rewards[-1]:.6f}")
    print(f"reward范围: {min(rewards):.6f} - {max(rewards):.6f}")

if __name__ == "__main__":
    # 使用脚本
    plot_reward_curve('./outputs/logging.jsonl')  # Set your logging.jsonl path