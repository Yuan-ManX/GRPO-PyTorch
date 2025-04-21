import dataclasses
import gc
import math
from collections import defaultdict
from typing import Callable, List
import numpy as np
import torch

from data_types import Episode, MiniBatch
from qwen2_model import Transformer
from tokenizer import Tokenizer


@torch.no_grad()
def rollout(
    model: Transformer,
    batch: MiniBatch,
    tokenizer: Tokenizer,
    max_gen_len: int,
    num_answer_per_question: int,
    reward_function: Callable,
    device: torch.device,
    dtype: torch.dtype,
) -> List[Episode]:
    """
    执行推理过程，生成文本并计算奖励。
    
    参数:
        model (Transformer): 预训练的 Transformer 模型，用于生成文本。
        batch (MiniBatch): 一个批次的数据，包含问题和上下文等信息。
        tokenizer (Tokenizer): 分词器，用于将文本转换为 token IDs 和反之。
        max_gen_len (int): 生成文本的最大长度。
        num_answer_per_question (int): 每个问题生成多少个答案。
        reward_function (Callable): 奖励函数，用于评估生成的文本。
        device (torch.device): 计算设备，如 GPU 或 CPU。
        dtype (torch.dtype): 张量数据类型。
    
    返回:
        List[Episode]: 生成的文本及其相关信息列表。
    """
    # 获取结束标记及其 ID 和填充标记 ID
    end_token = tokenizer.eos_token
    end_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    # 获取前缀 token IDs 和批次大小
    prefix_token_ids = batch.prefix_token_ids
    # 批次大小 = 前缀数量 * 每个前缀生成的答案数量
    bsz = len(batch.prefix) * num_answer_per_question

    # 最短前缀长度
    min_prompt_len = min(len(t) for t in prefix_token_ids)
    # 最长前缀长度
    max_prompt_len = max(len(t) for t in prefix_token_ids)
    # 总长度 = 生成长度 + 最长前缀长度
    total_len = max_gen_len + max_prompt_len

    # 初始化键值缓存，用于加速推理过程中的自注意力计算
    model.init_kv_cache(
        max_batch_size=bsz,
        max_seq_len=total_len,
        device=device,
        dtype=dtype,
    )

    # 初始化 tokens 张量，填充为填充标记 ID
    tokens = torch.full((bsz, total_len), pad_token_id, dtype=torch.long, device=device)

    # 将前缀 token IDs 填充到 tokens 张量中
    for k, t in enumerate(prefix_token_ids):
        offset = k * num_answer_per_question
        for i in range(num_answer_per_question):
            tokens[offset + i, : len(t)] = torch.tensor(
                t, dtype=torch.long, device=device
            )

    # 初始化当前位置为 0
    prev_pos = 0
    # 创建输入文本掩码，标记哪些位置是填充的
    input_text_mask = tokens != pad_token_id
    # 确保前缀长度小于总长度
    assert min_prompt_len < total_len
    # 初始化完成标志张量，标记哪些生成已经完成
    is_finished = torch.zeros((bsz,), dtype=torch.bool, device=device)

    # 生成循环，从 min_prompt_len 到 total_len - 1
    for cur_pos in range(min_prompt_len, total_len):
        # 打印生成进度
        print(
            f"\r* Generating trajectories: {cur_pos-min_prompt_len:>4d}/{total_len-min_prompt_len:>4d}",
            flush=True,
            end="",
        )
        # 使用自动混合精度进行推理
        with torch.autocast(device_type=device.type, dtype=dtype):
            # 获取当前 token 序列的 logits
            logits = model.inference(tokens[:, prev_pos:cur_pos], prev_pos)

        # 计算最后一个位置的 softmax 概率分布
        probs = torch.softmax(logits[:, -1], dim=-1)
        # 多项式采样下一个 token
        next_token = torch.multinomial(probs, num_samples=1)
        next_token = next_token.reshape(-1)

        # 如果当前位置不是填充的，则使用采样得到的 token，否则保持填充
        next_token = torch.where(
            input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
        )
        # 如果生成已经完成，则填充剩余的 token 为填充标记 ID
        next_token = torch.where(is_finished, pad_token_id, next_token)
        # 更新 tokens 张量
        tokens[:, cur_pos] = next_token

        # 检查是否遇到结束标记
        if end_token_id is not None:
            is_end_token = next_token == end_token_id
            is_generated_token = ~input_text_mask[:, cur_pos]
            is_finished = is_finished | (is_end_token & is_generated_token)
        # 更新当前位置
        prev_pos = cur_pos
        # 如果所有生成都已完成，则提前退出循环
        if is_finished.all():
            break
    
    # 删除键值缓存，释放内存
    model.del_kv_cache()
    # 清理内存
    gc.collect()
    torch.cuda.empty_cache()

    # 将张量转换为列表
    is_finished_list = is_finished.tolist()
    tokens_list = tokens.tolist()

    # 准备输出 episodes
    episodes = []
    for i in range(bsz // num_answer_per_question):
        for j in range(num_answer_per_question):
            idx = i * num_answer_per_question + j
            generated_token_ids = tokens_list[idx][len(batch.prefix_token_ids[i]) :]
            # 移除填充标记
            if pad_token_id in generated_token_ids:
                generated_token_ids = generated_token_ids[
                    : generated_token_ids.index(pad_token_id)
                ]

            # 将 token IDs 转换为文本
            generated_text = tokenizer.detokenize(generated_token_ids)
            # 计算奖励
            rewards = reward_function(
                response=generated_text,
                numbers=batch.numbers[i],
                target=batch.target[i],
                end_token=end_token,
            )

            # 创建 Episode 对象
            episode = Episode(
                prefix=batch.prefix[i],
                text=batch.prefix[i] + generated_text,
                prefix_token_ids=batch.prefix_token_ids[i],
                prefix_tokens=batch.prefix_tokens[i],
                generated_token_ids=generated_token_ids,
                is_finished=is_finished_list[idx],
                reward=rewards["reward"],
                reward_info=rewards["reward_info"],
            )
            episodes.append(episode)
    # 清除输出行
    print("\r", end=" " * 100, flush=True)
    return episodes


def normalize_rewards_per_group(episodes: List[Episode]) -> List[Episode]:
    """
    按组对奖励进行归一化。一个组由前缀定义。
    
    参数:
        episodes (List[Episode]): 包含多个 Episode 对象的列表。
    
    返回:
        List[Episode]: 奖励被归一化后的 Episode 对象列表。
    """
    # 使用 defaultdict 按前缀对 Episode 进行分组
    groups = defaultdict(list)
    for episode in episodes:
        # 将前缀转换为元组作为字典的键
        groups[tuple(episode.prefix)].append(episode)
    output = []
    for group in groups.values():
        # 提取当前组中所有 Episode 的奖励
        group_rewards = [item.reward for item in group]
        # 计算当前组的平均奖励
        mean_reward = np.mean(group_rewards)
        # 计算当前组的奖励标准差
        std_reward = np.std(group_rewards)
        # 对当前组中的每个 Episode 进行奖励归一化
        for episode in group:
            normalized_reward = (episode.reward - mean_reward) / (std_reward + 1e-4)
            # 使用归一化后的奖励替换原始奖励，生成新的 Episode 对象
            episode = dataclasses.replace(episode, reward=normalized_reward)
            # 将新的 Episode 对象添加到输出列表中
            output.append(episode)
    return output


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    计算给定 logits 的熵。
    
    参数:
        logits (torch.Tensor): 模型的未归一化对数概率，形状为 (批次大小, 序列长度, 词汇表大小)。
    
    返回:
        torch.Tensor: 每个位置的熵，形状为 (批次大小, 序列长度)。
    """
    # 计算 softmax 概率分布
    probs = torch.nn.functional.softmax(logits, dim=-1)
    # 计算熵：logsumexp(logits) - sum(probs * logits)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
    return entropy


def update_policy(
    model,
    optimizer,
    episodes: List[Episode],
    micro_batch_size: int,
    pad_token_id: int,
    max_grad_norm: float,
    device: torch.device,
    dtype: torch.dtype,
):
    """
    使用 GRPO 算法更新策略。
    
    参数:
        model: 待更新的模型。
        optimizer: 优化器，用于更新模型参数。
        episodes (List[Episode]): 包含多个 Episode 对象的列表。
        micro_batch_size (int): 微批次大小，用于梯度计算。
        pad_token_id (int): 填充标记 ID，用于填充序列。
        max_grad_norm (float): 梯度裁剪的最大范数。
        device (torch.device): 计算设备，如 GPU 或 CPU。
        dtype (torch.dtype): 张量数据类型。
    
    返回:
        dict: 包含损失、梯度范数和熵的字典。
    """
    # 按组对奖励进行归一化
    episodes = normalize_rewards_per_group(episodes)
    # 按 token 长度对 Episode 进行排序，以便高效地分批
    episodes.sort(key=lambda x: len(x.prefix_token_ids) + len(x.generated_token_ids))
    # 计算微批次的数量
    num_micro_batches = math.ceil(len(episodes) / micro_batch_size)
    # 计算总的目标 token 数量
    num_target_tokens = sum(len(episode.generated_token_ids) for episode in episodes)
    # 初始化熵
    entropy = 0.0

    # 按微批次处理数据
    for i in range(0, len(episodes), micro_batch_size):
        print(
            f"\r* Computing policy gradient: {i:>2d}/{len(episodes):>2d}",
            flush=True,
            end="",
        )
        # 确定当前微批次的起始和结束索引
        j = min(i + micro_batch_size, len(episodes))
        # 获取当前微批次的 Episode 列表
        batch_episodes = episodes[i:j]
        # 计算每个 Episode 的总长度（前缀 + 生成）
        batch_lengths = [
            len(episode.prefix_token_ids) + len(episode.generated_token_ids)
            for episode in batch_episodes
        ]
        # 获取当前微批次的最长序列长度
        batch_max_length = max(batch_lengths)
        # 对每个 Episode 的 token IDs 进行填充，并创建 token IDs 列表
        batch_token_ids = [
            episode.prefix_token_ids
            + episode.generated_token_ids
            + [pad_token_id] * (batch_max_length - batch_lengths[i])
            for i, episode in enumerate(batch_episodes)
        ]
        # 创建掩码列表，标记哪些位置是生成的 token
        batch_masks = [
            [0] * len(episode.prefix_token_ids)
            + [1] * len(episode.generated_token_ids)
            + [0] * (batch_max_length - batch_lengths[i])
            for i, episode in enumerate(batch_episodes)
        ]
        # 提取奖励列表
        batch_advantages = [episode.reward for episode in batch_episodes]
        # 将 token IDs 转换为张量
        batch_token_ids = torch.tensor(batch_token_ids, device=device, dtype=torch.long)
        # 将掩码转换为布尔张量
        batch_masks = torch.tensor(batch_masks, device=device, dtype=torch.bool)
        # 将奖励转换为浮点张量
        batch_advantages = torch.tensor(
            batch_advantages, device=device, dtype=torch.float32
        )

        # 使用自动混合精度进行前向传播
        with torch.autocast(device_type=device.type, dtype=dtype):
            # 输入 token IDs，去掉最后一个 token 作为目标
            input_token_ids = batch_token_ids[:, :-1]
            # 目标 token IDs，去掉第一个 token 作为输入
            target_token_ids = batch_token_ids[:, 1:]
            # 目标掩码，去掉第一个位置
            target_masks = batch_masks[:, 1:]
            # 前向传播，获取 logits
            logits = model.forward(input_token_ids).float()

        # 计算对数概率：-cross_entropy(logits, target_token_ids, ignore_index=pad_token_id)
        log_probs = -torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_token_ids.reshape(-1),
            ignore_index=pad_token_id,
            reduction="none",
        ).reshape(input_token_ids.shape[0], -1)

        # 计算熵，不计算梯度
        with torch.no_grad():
            token_entropy = compute_entropy(logits)
            entropy = entropy + (token_entropy * target_masks).sum() / num_target_tokens

        # 计算目标函数：log_probs * advantages
        obj = log_probs * batch_advantages[:, None]
        # 计算每个 token 的目标函数
        obj = (obj * target_masks).sum() / num_target_tokens
        # 计算损失：-目标函数
        loss = -obj
        # 反向传播，计算梯度
        loss.backward()

    # 更新策略
    # 梯度裁剪
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_grad_norm
    )
    # 更新优化器
    optimizer.step()
    # 清空梯度
    optimizer.zero_grad(set_to_none=True)
    # 返回包含损失、梯度范数和熵的字典
    return {
        "loss": loss.item(),
        "grad_norm": grad_norm.item(),
        "entropy": entropy.item(),
    }
