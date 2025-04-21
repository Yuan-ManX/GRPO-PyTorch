import json
from pathlib import Path
from typing import Dict, List

from jinja2 import Environment
from tokenizers import Encoding
from tokenizers import Tokenizer as TokenizerBase


class Tokenizer:
    """
    使用 Jinja2 引擎支持聊天模板的分词器类。

    该类集成了自定义的分词器配置和模板，用于将聊天消息编码为模型可接受的格式。
    """

    def __init__(self, tokenizer_path: str):
        """
        初始化 Tokenizer 实例。

        参数:
            tokenizer_path (str): 分词器模型文件的路径。
        """
        super().__init__()

        # 构建 tokenizer_config.json 的路径，假设与 tokenizer_path 在同一目录下
        tokenizer_config_path = Path(tokenizer_path).parent / "tokenizer_config.json"

        # 加载 tokenizer_config.json 配置文件
        self.tokenizer_config = json.load(open(tokenizer_config_path))

        # 初始化分词器，使用指定的 tokenizer_path 加载分词器模型
        self.tokenizer = TokenizerBase.from_file(tokenizer_path)

        # 使用 Jinja2 环境加载并解析聊天模板字符串
        self.chat_template = Environment().from_string(
            self.tokenizer_config["chat_template"]
        )

        # 获取 EOS（结束）标记及其对应的 ID
        self.eos_token = self.tokenizer_config["eos_token"]
        self.eos_token_id = self.tokenizer.token_to_id(self.eos_token)

        # 获取 PAD（填充）标记及其对应的 ID
        self.pad_token = self.tokenizer_config["pad_token"]
        self.pad_token_id = self.tokenizer.token_to_id(self.pad_token)

    def encode_chat(self, messages: List[Dict[str, str]]) -> str:
        """
        将聊天消息列表编码为字符串格式，使用预定义的聊天模板。

        参数:
            messages (List[Dict[str, str]]): 聊天消息列表，每个消息是一个字典，包含 'role' 和 'content' 键。
                例如:
                [
                    {"role": "user", "content": "你好！"},
                    {"role": "assistant", "content": "你好！有什么我可以帮忙的吗？"}
                ]

        返回:
            str: 编码后的字符串，包含所有消息和生成提示（如果启用）。
        """
        # 使用 Jinja2 模板渲染消息，添加生成提示（如果配置中启用了）
        return self.chat_template.render(messages=messages, add_generation_prompt=True)

    def encode_chat_with_response_prompt(
        self, messages: List[Dict[str, str]], prompt: str
    ) -> str:
        """
        将聊天消息列表编码为字符串，并附加一个响应提示。

        参数:
            messages (List[Dict[str, str]]): 聊天消息列表，格式同上。
            prompt (str): 要附加的响应提示字符串。

        返回:
            str: 编码后的字符串，包含所有消息和附加的响应提示。
        """
        # 先编码聊天消息，然后附加响应提示
        return self.encode_chat(messages) + prompt

    def tokenize(self, text: str) -> Encoding:
        """
        将文本字符串分词并转换为 token IDs。

        参数:
            text (str): 要分词的文本。

        返回:
            Encoding: 分词后的结果，包含 token IDs 和其他相关信息。
        """
        # 使用分词器对文本进行编码，返回 Encoding 对象
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids: List[int]) -> str:
        """
        将 token IDs 列表反转换为文本字符串。

        参数:
            token_ids (List[int]): 要反转换的 token IDs 列表。

        返回:
            str: 反转换后的文本字符串。
        """
        # 使用分词器对 token IDs 进行解码，skip_special_tokens=False 表示保留特殊标记
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)
