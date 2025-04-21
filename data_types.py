from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Episode:
    """
    Episode 类用于存储与一个完整训练或推理片段（episode）相关的所有数据。
    
    属性:
        prefix (str): 
            - 片段的前缀文本，作为生成文本的起始部分或上下文。
            - 例如，在对话系统中，这可能是用户输入的初始问题或提示。
        
        text (str): 
            - 生成的完整文本内容，基于前缀文本生成的结果。
            - 例如，AI 模型根据前缀生成的回复或续写内容。
        
        prefix_token_ids (List[int]): 
            - 前缀文本对应的 token ID 列表。
            - 每个 token ID 通常对应词汇表中的一个词或子词，用于模型输入。
        
        prefix_tokens (List[str]): 
            - 前缀文本对应的 token 字符串列表。
            - 直接表示前缀文本的词汇单元，便于理解和调试。
        
        generated_token_ids (List[int]): 
            - 生成文本对应的 token ID 列表。
            - 包含模型生成的每个词汇单元的 ID，用于后续处理或评估。
            - 例如，用于计算损失或生成文本的逆向转换。
        
        is_finished (bool): 
            - 标记该片段是否已完成生成。
            - 如果为 True，表示生成过程已经结束；否则，表示仍在生成中。
        
        reward (float): 
            - 该片段的奖励值，用于强化学习中的训练。
            - 奖励值通常由某个奖励函数根据生成文本的质量或任务完成情况计算得出。
        
        reward_info (Dict[str, float]): 
            - 包含更多奖励相关信息的字典。
            - 例如，不同维度的奖励分数或详细的奖励分解。
            - 键为奖励的类别或维度，值为对应的奖励分数。
    """

    prefix: str 
    text: str
    prefix_token_ids: List[int]
    prefix_tokens: List[str]
    generated_token_ids: List[int]
    is_finished: bool
    reward: float
    reward_info: Dict[str, float]


@dataclass
class MiniBatch:
    """
    MiniBatch 类用于封装一个训练步骤中每个批次的数据。
    
    属性:
        prefix (List[str]): 
            - 当前批次中所有样本的前缀文本列表。
            - 每个元素对应一个样本的前缀文本，用于生成后续内容。
        
        prefix_tokens (List[List[str]]): 
            - 当前批次中所有样本的前缀文本对应的 token 列表。
            - 外层列表对应批次中的样本，内层列表对应每个样本的 token 字符串。
        
        prefix_token_ids (List[List[int]]): 
            - 当前批次中所有样本的前缀文本对应的 token ID 列表。
            - 外层列表对应批次中的样本，内层列表对应每个样本的 token ID。
            - 用于模型输入，表示文本的数值化表示。
        
        numbers (List[List[int]]): 
            - 当前批次中所有样本的数值特征列表。
            - 外层列表对应批次中的样本，内层列表对应每个样本的数值特征。
            - 例如，可能包含时间步长、位置编码等信息。
        
        target (List[int]): 
            - 当前批次中所有样本的目标标签列表。
            - 每个元素对应一个样本的目标标签，用于训练过程中的监督学习。
            - 例如，在语言模型中，这可能是下一个 token 的 ID。
    """

    prefix: List[str]
    prefix_tokens: List[List[str]]
    prefix_token_ids: List[List[int]]
    numbers: List[List[int]]
    target: List[int]
