import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from torch.utils.data import Dataset

from data_types import MiniBatch
from tokenizer import Tokenizer


# SYSTEM_MESSAGE 定义了AI助手的基本行为准则，即先思考推理过程再提供答案。
SYSTEM_MESSAGE = (
    "You are a helpful assistant. You first think about the reasoning process "
    "in your mind and then provide the user with the answer."
)

# USER_TEMPLATE 定义了用户请求的模板，用户提供一组数字和一个目标值，AI需要用这些数字和基本算术运算构造一个等于目标值的等式。
USER_TEMPLATE = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
    "Show your work in <think> </think> tags. "
    "And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."
)

# RESPONSE_PROMPT 定义了AI助手在生成响应时使用的提示符，用于引导AI进行逐步思考和解答。
RESPONSE_PROMPT = "Let me solve this step by step.\n<think>"


# CountdownTasksDataset 类用于准备倒计时任务的数据集，用于训练模型。
class CountdownTasksDataset(Dataset):
    """
    CountdownTasksDataset 类用于准备倒计时任务的数据集。

    参数:
    tokenizer (Tokenizer): 用于将文本转换为模型可接受的输入格式的标记器。
    data_path (str): 数据集的存储路径。
    split (str): 数据集的分割方式，默认为 'train'，表示训练集；也可以是 'test'，表示测试集。
    test_size (int): 测试集的大小，默认为100。
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        data_path: str,
        split: str = "train",
        test_size: int = 100,
    ):
        """
        初始化 CountdownTasksDataset 实例。

        参数:
        tokenizer (Tokenizer): 用于将文本转换为模型可接受的输入格式的标记器。
        data_path (str): 数据集的存储路径。
        split (str): 数据集的分割方式，默认为 'train'，表示训练集；也可以是 'test'，表示测试集。
        test_size (int): 测试集的大小，默认为100。
        """
        # 读取存储在指定路径的 Parquet 格式的数据集文件。
        data = pd.read_parquet(Path(data_path) / "data")

        # 根据分割方式选择数据集。如果 split 是 'train'，则选择除最后 test_size 个样本之外的所有样本作为训练集；
        # 否则，选择最后 test_size 个样本作为测试集。
        self.data = (
            data.iloc[:-test_size] if split == "train" else data.iloc[-test_size:]
        )

        # 存储标记器实例
        self.tokenizer = tokenizer

    def __len__(self):
        """
        返回数据集的大小，即样本的数量。

        返回:
        int: 数据集的大小。
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取指定索引的样本，并对其进行编码。

        参数:
        idx (int): 样本的索引。

        返回:
        dict: 编码后的样本，包括原始数据和编码后的前缀。
        """
        # 获取指定索引的样本，并将其转换为字典格式。
        item = self.data.iloc[idx].to_dict()
        # 对样本进行编码，生成模型输入的前缀。
        item.update(self.encode_prefix(item["nums"], item["target"]))
        return item

    def encode_prefix(self, numbers: List[int], target: int):
        """
        对用户输入进行编码，生成模型输入的前缀。

        参数:
        numbers (List[int]): 一组数字。
        target (int): 目标值。

        返回:
        dict: 编码后的前缀，包括前缀文本、标记列表和标记ID列表。
        """
        # 使用 USER_TEMPLATE 格式化用户请求的文本。
        user_message = USER_TEMPLATE.format(numbers=numbers, target=target)

        # 使用标记器对系统消息和用户消息进行编码，并添加响应提示符。
        prefix = self.tokenizer.encode_chat_with_response_prompt(
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_message},
            ],
            RESPONSE_PROMPT,
        )

        # 对前缀文本进行标记化，生成标记列表和标记ID列表。
        tokens = self.tokenizer.tokenize(prefix)
        return {
            "prefix": prefix,
            "prefix_tokens": tokens.tokens,
            "prefix_token_ids": tokens.ids,
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> MiniBatch:
        """
        将一批样本整理成一个批次。

        参数:
        batch (List[Dict[str, Any]]): 一批样本的列表，每个样本是一个字典。

        返回:
        MiniBatch: 整理后的批次对象，包含数字数字、目标值、前缀文本、标记列表和标记ID列表。
        """
        # 从批次中提取数字列表、目标值列表、前缀文本列表、标记列表和标记ID列表。
        numbers = [item["nums"] for item in batch]
        target = [item["target"] for item in batch]
        prefix = [item["prefix"] for item in batch]
        prefix_tokens = [item["prefix_tokens"] for item in batch]
        prefix_token_ids = [item["prefix_token_ids"] for item in batch]
        # 返回一个 MiniBatch 对象，包含整理后的数据。
        return MiniBatch(
            numbers=numbers,
            target=target,
            prefix=prefix,
            prefix_tokens=prefix_tokens,
            prefix_token_ids=prefix_token_ids,
        )


def format_reward_function(response: str, end_token: Optional[str] = None) -> float:
    """
    检查响应是否符合格式 
    <think>...</think><answer>...</answer>

    参数:
    response (str): AI生成的响应文本。
    end_token (Optional[str]): 可选的结束标记，如果提供且响应以该标记结尾，则会先移除该标记再进行匹配。

    返回:
    float: 如果响应完全符合格式，返回1.0；否则，根据部分匹配情况返回相应的奖励分数。
    """
    
    # 如果提供了 end_token 并且响应以 end_token 结尾，则移除 end_token
    if end_token and response.endswith(end_token):
        response = response[: -len(end_token)]

    # 定义用于匹配 <think>...</think> 的正则表达式
    think_regex = r"<think>.*?<\/think>"
    # 定义用于匹配 <answer>...</answer> 的正则表达式
    answer_regex = r"<answer>.*?<\/answer>"
    # 定义用于完全匹配整个响应格式的正则表达式
    full_format_regex = r"^<think>.*?<\/think>\n<answer>.*?<\/answer>$"

    # 使用正则表达式搜索 <think>...</think> 部分，re.DOTALL 使 . 匹配包括换行符在内的所有字符
    think_match = re.search(think_regex, response, re.DOTALL)
    # 使用正则表达式搜索 <answer>...</answer> 部分
    answer_match = re.search(answer_regex, response, re.DOTALL)
    # 使用正则表达式匹配整个响应的格式
    full_format_match = re.match(full_format_regex, response, re.DOTALL)

    # 如果整个响应完全符合格式，则返回最高奖励1.0
    if full_format_match:
        return 1.0

    # 初始化奖励分数为0.0
    reward = 0.0

    # 如果找到 <think>...</think> 部分，则增加0.1的奖励
    if think_match:
        reward += 0.1

    # 如果找到 <answer>...</answer> 部分，则增加0.5的奖励
    if answer_match:
        reward += 0.5

    # 返回总的奖励分数
    return reward


def answer_reward_function(
    response: str, numbers: List[int] = None, target: int = None
) -> float:
    """
    检查答案是否使用了所有提供的数字且每个数字仅使用一次，并验证答案是否等于目标值。

    参数:
    response (str): AI生成的响应文本。
    numbers (List[int], optional): 一组数字，答案中应包含这些数字且每个数字仅使用一次。
    target (int, optional): 目标值，答案计算结果应等于该值。

    返回:
    float: 如果答案符合所有条件，返回1.0；否则，返回0.0。
    """

    # 定义用于匹配 <answer>...</answer> 的正则表达式
    answer_regex = r"<answer>(.*?)<\/answer>"
    # 使用正则表达式搜索 <answer>...</answer> 部分
    answer_match = re.search(answer_regex, response, re.DOTALL)
    # 如果没有找到 <answer>...</answer> 部分，则返回0.0
    if not answer_match:
        return 0.0

    # 获取答案的内容
    answer_content = answer_match.group(1)
    # 如果答案内容为空，则返回0.0
    if not answer_content:
        return 0.0

    # 定义允许的字符集（数字、加号、减号、乘号、除号、括号和空格）
    allowed_chars = r"^[0-9+\-*/() ]+$"
    # 检查答案内容是否仅包含允许的字符
    if not re.match(allowed_chars, answer_content):
        return 0.0

    # 检查答案中使用的数字是否与提供的数字完全一致且每个数字仅使用一次
    used_numbers = [int(n) for n in re.findall(r"\d+", answer_content)]
    if sorted(used_numbers) != sorted(numbers):
        return 0.0

    # 尝试计算答案的值并与目标值进行比较
    try:
        # 使用一个受限的全局命名空间来计算表达式，防止执行恶意代码
        result = eval(answer_content, {"__builtins__": None}, {})
        # 检查计算结果是否与目标值在一定精度范围内相等
        if abs(float(result) - float(target)) < 1e-5:
            return 1.0
    except:
        # 如果计算过程中出现任何异常，则返回0.0
        pass

    # 如果答案不符合条件，则返回0.0
    return 0.0


def reward_function(
    response: str,
    numbers: List[int] = None,
    target: int = None,
    end_token: str = None,
) -> Dict[str, Any]:
    """
    针对倒计时任务的奖励函数。

    总奖励 = 0.1 * 格式奖励 + 答案奖励

    参数:
    response (str): AI生成的响应文本。
    numbers (List[int], optional): 一组数字，答案中应包含这些数字且每个数字仅使用一次。
    target (int, optional): 目标值，答案计算结果应等于该值。
    end_token (str, optional): 可选的结束标记，用于标识响应的结束。

    返回:
    Dict[str, Any]: 包含总奖励和奖励详细信息。
    """

    # 计算格式奖励，传入的 response 前面加上 "<think>" 标签
    format_reward = format_reward_function("<think>" + response, end_token)
    # 计算答案奖励
    answer_reward = answer_reward_function(response, numbers, target)
    # 返回总奖励和奖励详细信息
    return {
        "reward": format_reward * 0.1 + answer_reward,
        "reward_info": {
            "format_reward": format_reward,
            "answer_reward": answer_reward,
        },
    }
