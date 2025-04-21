import html
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from countdown_task import CountdownTasksDataset, reward_function
from grpo_v2 import rollout, update_policy
from optimizer import MemoryEfficientAdamW
from qwen2_model import Transformer
from tokenizer import Tokenizer


def evaluate(model, tokenizer, device, dtype, config):
    """
    评估模型在倒计时任务上的表现。

    参数:
    - model: 需要评估的模型实例。
    - tokenizer: 用于将文本转换为模型可接受的输入格式的标记器。
    - device: 计算设备，例如 'cpu' 或 'cuda'。
    - dtype: 数据类型，例如 torch.float32。
    - config (dict): 配置字典，包含以下键：
        - "data": 数据相关配置，包含：
            - "path" (str): 数据集的存储路径。
            - "test_size" (int): 测试集的大小。
        - "training": 训练相关配置，包含：
            - "batch_size" (int): 每个批次的样本数量。
            - "max_gen_len" (int): 生成的最大序列长度。

    返回:
    - float: 模型在测试集上的平均答案奖励分数。
    """

    # 使用 CountdownTasksDataset 类创建测试数据集
    # 参数:
    # - data_path: 数据集的存储路径，从 config["data"]["path"] 获取
    # - tokenizer: 标记器实例
    # - split: 数据集分割方式，这里设置为 "test" 表示测试集
    # - test_size: 测试集的大小，从 config["data"]["test_size"] 获取
    test_dataset = CountdownTasksDataset(
        data_path=config["data"]["path"],
        tokenizer=tokenizer,
        split="test",
        test_size=config["data"]["test_size"],
    )

    # 创建一个生成器对象，用于生成随机数
    # device: 计算设备，从函数参数传入
    generator = torch.Generator(device=device)
    
    
    # 创建 DataLoader 对象，用于加载测试数据集
    # 参数:
    # - test_dataset: 测试数据集
    # - shuffle: 是否打乱数据，这里设置为 False 表示不打乱
    # - collate_fn: 整理批次的函数，这里使用 CountdownTasksDataset.collate_fn
    # - generator: 随机数生成器
    # - batch_size: 每个批次的样本数量，从 config["training"]["batch_size"] 获取，并除以2
    # - drop_last: 是否丢弃最后一个不完整的批次，这里设置为 False 表示不丢弃
    dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=CountdownTasksDataset.collate_fn,
        generator=generator,
        batch_size=config["training"]["batch_size"] // 2,
        drop_last=False,
    )

    # 初始化一个空列表，用于存储每个样本的答案奖励分数
    success = []

    # 遍历 DataLoader 中的每个批次
    for batch in dataloader:
        # 调用 rollout 函数进行推理生成
        # 参数:
        # - model: 模型实例
        # - tokenizer: 标记器实例
        # - batch: 当前批次的样本数据
        # - max_gen_len: 生成的最大序列长度，从 config["training"]["max_gen_len"] 获取，并乘以2
        # - num_answer_per_question: 每个问题生成的答案数量，这里设置为1
        # - reward_function: 奖励函数，这里使用 reward_function
        # - device: 计算设备
        # - dtype: 数据类型
        episodes = rollout(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=config["training"]["max_gen_len"] * 2,
            num_answer_per_question=1,
            reward_function=reward_function,
            device=device,
            dtype=dtype,
        )

        # 从生成的 episodes 中提取每个样本的答案奖励分数，并添加到 success 列表中
        # episodes 是一个包含多个 Episode 对象的列表，每个 Episode 对象包含一个 reward_info 字典
        # reward_info 字典中包含 "answer_reward" 键，表示答案的奖励分数
        success.extend([episode.reward_info["answer_reward"] for episode in episodes])

    # 计算平均答案奖励分数
    # np.mean(success) 计算 success 列表中所有奖励分数的平均值
    return np.mean(success)


def main(config_path: str):
    """
    主函数，用于加载配置、初始化模型和数据、训练模型并评估其性能。

    参数:
    - config_path (str): 配置文件的路径，配置文件通常为 YAML 格式。
    """
    # 打开配置文件并加载配置内容
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 从配置中获取预训练模型的路径，并转换为 Path 对象
    pretrained_model_path = Path(config["model"]["pretrained_model_path"])
    # 从配置中获取计算设备，并创建 torch.device 对象
    device = torch.device(config["model"]["device"])
    # 定义数据类型映射字典，将配置中的 dtype 字符串转换为 torch 的数据类型
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    # 根据配置获取数据类型，默认为 torch.bfloat16
    dtype = dtype_map.get(config["model"]["dtype"], torch.bfloat16)
    # 设置默认的计算设备
    torch.set_default_device(device)
    # 设置随机种子，以确保结果的可复现性
    torch.random.manual_seed(config["training"]["random_seed"])

    # 从配置中获取训练相关的参数
    BATCH_SIZE = config["training"]["batch_size"]
    NUM_QUESTIONS_PER_BATCH = config["training"]["num_questions_per_batch"]
    NUM_ANSWERS_PER_QUESTION = BATCH_SIZE // NUM_QUESTIONS_PER_BATCH

    # 获取当前时间，用于日志目录命名
    current_time = datetime.now().strftime(r"%Y%m%d-%H%M%S")
    # 初始化 TensorBoard 的 SummaryWriter，用于记录训练过程中的指标
    tb_writer = SummaryWriter(log_dir=f"{config['training']['log_dir']}/{current_time}")
    # 初始化 Tokenizer，加载预训练模型的 tokenizer.json 文件
    tokenizer = Tokenizer(str(pretrained_model_path / "tokenizer.json"))

    # 使用 CountdownTasksDataset 类创建训练数据集
    train_dataset = CountdownTasksDataset(
        data_path=config["data"]["path"],   # 数据集路径
        tokenizer=tokenizer,   # 标记器实例
        split="train",    # 数据集分割方式，这里是训练集
        test_size=config["data"]["test_size"],   # 测试集大小
    )

    # 创建数据加载器，用于加载训练数据
    generator = torch.Generator(device=device)   # 随机数生成器
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,    # 是否打乱数据
        collate_fn=CountdownTasksDataset.collate_fn,   # 整理批次的函数
        generator=generator,   # 随机数生成器
        batch_size=NUM_QUESTIONS_PER_BATCH,   # 每个批次的样本数量
    )

    # 从预训练模型初始化模型，并设置为训练模式
    model = Transformer.from_pretrained(pretrained_model_path, device=device).train()

    # 初始化优化器，使用 MemoryEfficientAdamW 优化器
    optimizer = MemoryEfficientAdamW(
        model.parameters(),    # 模型参数
        lr=config["training"]["learning_rate"],   # 学习率
        weight_decay=config["training"]["weight_decay"],  # 权重衰减
        betas=config["training"]["betas"],   # AdamW 的 beta 参数
        enabled=config["training"]["memory_efficient_adamw"],   # 是否启用 MemoryEfficientAdamW
    )

    # 记录开始时间
    start_time = time.time()
    # 创建检查点目录，如果不存在则创建
    ckpt_dir = Path(config["training"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 开始训练循环
    for step, batch in enumerate(train_dataloader, start=1):
        # 进行推理生成，生成模型对输入数据的响应
        episodes = rollout(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=config["training"]["max_gen_len"],   # 生成的最大序列长度
            num_answer_per_question=NUM_ANSWERS_PER_QUESTION,   # 每个问题生成的答案数量
            reward_function=reward_function,   # 奖励函数
            device=device,
            dtype=dtype,
        )

        # 如果配置中启用了跳过未完成的 episodes，则过滤掉未完成的 episodes
        if config["training"]["skip_unfinished_episodes"]:
            episodes = [episode for episode in episodes if episode.is_finished]

        # 更新模型策略
        results = update_policy(
            model=model,
            optimizer=optimizer,
            episodes=episodes,
            micro_batch_size=config["training"]["micro_batch_size"],   # 微批次大小
            pad_token_id=tokenizer.pad_token_id,     # 填充标记的 ID
            max_grad_norm=config["training"]["max_grad_norm"],   # 最大梯度范数
            device=device,
            dtype=dtype,
        )

        # 同步 CUDA，确保所有 CUDA 操作完成
        torch.cuda.synchronize()

        # 记录结束时间，计算持续时间
        end_time = time.time()
        duration = end_time - start_time
        start_time = end_time

        # 计算并记录重要的指标
        reward = [episode.reward for episode in episodes]   # 每个 episode 的奖励
        # 每个 episode 的格式奖励
        formatted_reward = [
            episode.reward_info["format_reward"] for episode in episodes
        ]
        # 每个 episode 的答案奖励
        answer_reward = [episode.reward_info["answer_reward"] for episode in episodes]
        # 已完成的 episode 数量
        num_finished_episodes = sum(episode.is_finished for episode in episodes)
        # 平均奖励
        mean_reward = np.mean(reward)
        # 奖励的标准差
        std_reward = np.std(reward)
        # 成功率
        success_rate = np.mean(answer_reward)
        # 平均格式奖励
        format_reward = np.mean(formatted_reward)
        # 梯度范数
        grad_norm = results["grad_norm"]
        # 熵
        entropy = results["entropy"]
        # 当前学习率
        lr = optimizer.param_groups[0]["lr"]
        # 损失
        loss = results["loss"]
        # 平均响应长度
        mean_response_len = np.mean(
            [len(episode.generated_token_ids) for episode in episodes]
        )

        # 打印当前训练状态
        print(
            f"\rStep {step}, mean_reward: {mean_reward:.2f}, "
            f"train success_rate: {success_rate:.2f}, "
            f"grad_norm: {grad_norm:.2f}, duration: {duration:.2f}, "
            f"num_finished_episodes: {num_finished_episodes}, "
            f"mean_response_len: {mean_response_len:.2f}, "
            f"entropy: {entropy:.2f}"
        )

        # 每隔一定的步数进行评估
        if step % config["training"]["eval_interval"] == 0:
            eval_success_rate = evaluate(model, tokenizer, device, dtype, config)
            print(f"\rEval success rate: {eval_success_rate:.2f}" + " " * 100)
            tb_writer.add_scalar("success_rate/eval", eval_success_rate, step)

        # 将指标写入 TensorBoard
        tb_writer.add_scalar("loss", loss, step)
        tb_writer.add_scalar("mean_reward", mean_reward, step)
        tb_writer.add_scalar("std_reward", std_reward, step)
        tb_writer.add_scalar("success_rate/train", success_rate, step)
        tb_writer.add_scalar("format_reward", format_reward, step)
        tb_writer.add_scalar("grad_norm", grad_norm, step)
        tb_writer.add_scalar("duration", duration, step)
        tb_writer.add_scalar("num_finished_episodes", num_finished_episodes, step)
        tb_writer.add_scalar("learning_rate", lr, step)
        tb_writer.add_scalar("mean_response_len", mean_response_len, step)
        tb_writer.add_scalar("entropy", entropy, step)

        # 将生成的文本添加到 TensorBoard 中
        for i, episode in enumerate(episodes):
            # TensorBoard 将文本视为 markdown 格式
            text = html.escape(episode.text)
            tb_writer.add_text(f"text_{i}", f"<pre>{text}</pre>", step)

        # 保存模型checkpoint
        if step % config["training"]["ckpt_save_interval"] == 0:
            output_file = ckpt_dir / f"ckpt_{step:06d}.pt"
            torch.save(model.state_dict(), output_file)
            print(f"Saved checkpoint to {output_file}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)
