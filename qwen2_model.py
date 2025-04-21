import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn


# 使用 @dataclass 装饰器定义 Qwen2 模型配置类
@dataclass
class Qwen2Config:
    """
    Qwen2 模型配置类，用于存储模型的各项超参数。
    
    参数:
        attention_dropout (float): 注意力机制中的 dropout 概率。默认值为 0.0。
        bos_token_id (int): 起始标记 (Beginning of Sequence) 的 ID。默认值为 151643。
        eos_token_id (int): 结束标记 (End of Sequence) 的 ID。默认值为 151645。
        hidden_act (str): 隐藏层激活函数的类型。默认值为 "silu"（Sigmoid Linear Unit）。
        hidden_size (int): 隐藏层的大小，即每个 Transformer 的维度。默认值为 2048。
        initializer_range (float): 权重初始化的标准差范围。默认值为 0.02。
        intermediate_size (int): 中间层的大小，通常用于前馈网络。默认值为 11008。
        max_position_embeddings (int): 位置编码的最大长度。默认值为 32768。
        max_window_layers (int): 最大窗口层数。默认值为 70。
        model_type (str): 模型类型。默认值为 "qwen2"。
        num_attention_heads (int): 注意力头的数量。默认值为 16。
        num_hidden_layers (int): 隐藏层的数量，即 Transformer 层数。默认值为 36。
        num_key_value_heads (int): 键值对注意力头的数量。默认值为 2。
        rms_norm_eps (float): RMS 归一化中的小常数，防止除零。默认值为 1e-6。
        rope_theta (float): RoPE（旋转位置编码）中的参数 θ。默认值为 1000000.0。
        sliding_window (int): 滑动窗口的大小。默认值为 32768。
        tie_word_embeddings (bool): 是否绑定词嵌入权重。默认值为 True。
        torch_dtype (str): PyTorch 张量的数据类型。默认值为 "bfloat16"。
        use_cache (bool): 是否使用缓存。默认值为 True。
        use_sliding_window (bool): 是否使用滑动窗口。默认值为 False。
        vocab_size (int): 词汇表的大小。默认值为 151936。
    """

    attention_dropout: float = 0.0
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    hidden_act: str = "silu"
    hidden_size: int = 2048
    initializer_range: float = 0.02
    intermediate_size: int = 11008
    max_position_embeddings: int = 32768
    max_window_layers: int = 70
    model_type: str = "qwen2"
    num_attention_heads: int = 16
    num_hidden_layers: int = 36
    num_key_value_heads: int = 2
    rms_norm_eps: float = 1e-06
    rope_theta: float = 1000000.0
    sliding_window: int = 32768
    tie_word_embeddings: bool = True
    torch_dtype: str = "bfloat16"
    use_cache: bool = True
    use_sliding_window: bool = False
    vocab_size: int = 151936


# 定义 RMSNorm 类，实现 RMS 归一化
class RMSNorm(torch.nn.Module):
    """
    RMS 归一化层，用于对输入张量进行归一化处理。
    
    参数:
        dim (int): 输入张量的维度。
        eps (float): 小常数，防止除零。默认值为 1e-6。
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 定义可学习的权重参数，初始化为全1
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        对输入张量进行 RMS 归一化。
        
        参数:
            x (torch.Tensor): 输入张量。
        
        返回:
            torch.Tensor: 归一化后的张量。
        """
        # 计算均方根值，并加上小常数防止除零
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        前向传播函数，对输入张量进行 RMS 归一化并应用权重。
        
        参数:
            x (torch.Tensor): 输入张量。
        
        返回:
            torch.Tensor: 归一化并加权后的张量。
        """
        # 记录输入张量的数据类型
        input_dtype = x.dtype
        # 将输入转换为 float32 以进行计算
        x = x.to(torch.float32)
        # 进行 RMS 归一化
        x = self._norm(x).type_as(x)
        # 应用权重并转换回原始数据类型
        x = self.weight * x.to(input_dtype)
        return x


def rotate_half(x):
    """
    将输入张量的后半部分旋转，用于旋转位置编码 (RoPE)。
    
    参数:
        x (torch.Tensor): 输入张量，形状为 (..., dim)。
    
    返回:
        torch.Tensor: 旋转后的张量，形状为 (..., dim)。
    """
    # 将输入张量分割为两部分
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    # 将后半部分取反并与前半部分拼接
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2):
    """
    应用旋转位置编码 (RoPE) 到查询 (q) 和键 (k) 张量。
    
    参数:
        q (torch.Tensor): 查询张量，形状为 (..., dim)。
        k (torch.Tensor): 键张量，形状为 (..., dim)。
        cos (torch.Tensor): 余弦编码，形状为 (..., dim)。
        sin (torch.Tensor): 正弦编码，形状为 (..., dim)。
        unsqueeze_dim (int): 在指定维度上增加维度。默认值为 2。
    
    返回:
        tuple:
            - q_embed (torch.Tensor): 应用 RoPE 后的查询张量。
            - k_embed (torch.Tensor): 应用 RoPE 后的键张量。
    """
    # 在指定维度上增加维度，以便与查询和键张量进行广播
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    # 应用 RoPE 到查询张量
    q_embed = (q * cos) + (rotate_half(q) * sin)
    # 应用 RoPE 到键张量
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    """
    Attention 类实现了多头自注意力机制，用于 Qwen2 模型。
    
    参数:
        args (Qwen2Config): 配置参数，包含模型的各种超参数。
    """
    def __init__(self, args: Qwen2Config):
        super().__init__()

        # 根据配置参数设置键值对 (KV) 头的数量
        # 如果 num_key_value_heads 为 None，则 KV 头数量等于注意力头数量
        self.n_kv_heads = (
            args.num_attention_heads
            if args.num_key_value_heads is None
            else args.num_key_value_heads
        )

        # 注意力头的数量
        self.n_heads = args.num_attention_heads
        # 键值对头的数量
        self.n_kv_heads = self.n_kv_heads
        # 每个表示头的重复次数
        self.n_rep = self.n_heads // self.n_kv_heads
        # 每个注意力头的维度大小
        self.head_dim = args.hidden_size // args.num_attention_heads

        # 定义线性层，用于将输入投影到查询 (Q)、键 (K)、值 (V) 和输出 (O)
        self.q_proj = nn.Linear(
            args.hidden_size,     # 输入特征维度
            args.num_attention_heads * self.head_dim,   # 输出特征维度
            bias=True,   # 是否使用偏置
        )
        self.k_proj = nn.Linear(
            args.hidden_size,
            args.num_key_value_heads * self.head_dim,
            bias=True,
        )
        self.v_proj = nn.Linear(
            args.hidden_size,
            args.num_key_value_heads * self.head_dim,
            bias=True,
        )
        self.o_proj = nn.Linear(
            args.num_attention_heads * self.head_dim,   # 输入特征维度
            args.hidden_size,    # 输出特征维度
            bias=False,    # 输出层通常不使用偏置
        )
        # 保存配置参数
        self.args = args

    def init_kv_cache(
        self,
        max_batch_size: int,
        max_seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        """
        初始化键 (K) 和值 (V) 的缓存，用于加速推理过程。
        
        参数:
            max_batch_size (int): 最大的批次大小。
            max_seq_len (int): 最大的序列长度。
            dtype (torch.dtype): 张量的数据类型。
            device (torch.device): 张量所在的设备。
        """
        # 定义缓存的形状 (批次大小, 最大序列长度, KV 头数量, 每个头的维度)
        cache_shape = (max_batch_size, max_seq_len, self.n_kv_heads, self.head_dim)
        # 初始化键缓存为全零张量
        cache_k = torch.zeros(cache_shape, dtype=dtype, device=device)
        # 初始化值缓存为全零张量
        cache_v = torch.zeros(cache_shape, dtype=dtype, device=device)
        # 注册缓存张量为缓冲区，不作为模型参数保存
        self.register_buffer("cache_k", cache_k, persistent=False)
        self.register_buffer("cache_v", cache_v, persistent=False)

    def del_kv_cache(self):
        """
        删除键 (K) 和值 (V) 的缓存，释放内存。
        """
        self.cache_k = None
        self.cache_v = None

    def forward(
        self,
        x: torch.Tensor,
        pos_embed: Tuple[torch.Tensor, torch.Tensor],
        start_pos: Optional[Union[int, torch.Tensor]] = None,
    ):
        """
        前向传播函数，执行多头自注意力机制。
        
        参数:
            x (torch.Tensor): 输入张量，形状为 (批次大小, 序列长度, 隐藏层维度)。
            pos_embed (Tuple[torch.Tensor, torch.Tensor]): 位置编码的余弦和正弦部分。
            start_pos (Optional[Union[int, torch.Tensor]]): 起始位置，用于推理模式。默认为 None。
        
        返回:
            torch.Tensor: 注意力机制的输出，形状为 (批次大小, 序列长度, 隐藏层维度)。
        """
        # 获取输入张量的批次大小 (bsz) 和序列长度 (seqlen)
        bsz, seqlen, _ = x.shape
        # 将输入张量投影到查询 (Q)、键 (K) 和值 (V)
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # 重塑查询张量以匹配多头注意力的形状 (批次大小, 序列长度, 注意力头数量, 每个头的维度)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        # 重塑键和值张量以匹配 KV 头的形状
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # 解包位置编码的余弦和正弦部分
        cos, sin = pos_embed
        # 应用旋转位置编码 (RoPE) 到查询和键张量
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin, unsqueeze_dim=2)
        if start_pos is not None:
            # 推理模式
            end_pos = start_pos + seqlen
            # 将键和值存储到缓存中
            self.cache_k[:bsz, start_pos:end_pos, :, :] = xk
            self.cache_v[:bsz, start_pos:end_pos, :, :] = xv
            # 执行缩放点积注意力机制，使用缓存中的键和值
            output = torch.nn.functional.scaled_dot_product_attention(
                query=xq.transpose(1, 2),   # 重塑查询张量以匹配注意力机制的要求
                key=self.cache_k[:bsz, :end_pos].transpose(1, 2),   # 重塑缓存中的键张量
                value=self.cache_v[:bsz, :end_pos].transpose(1, 2),   # 重塑缓存中的值张量
                is_causal=True if seqlen > 1 else False,   # 是否使用因果掩码
                enable_gqa=True,    # 是否启用分组查询注意力 (Grouped Query Attention)
            ).transpose(1, 2)
        else:
            # 训练模式
            # 执行缩放点积注意力机制，使用当前的键和值
            output = torch.nn.functional.scaled_dot_product_attention(
                query=xq.transpose(1, 2),
                key=xk.transpose(1, 2),
                value=xv.transpose(1, 2),
                is_causal=True,    # 在训练模式下总是使用因果掩码
                enable_gqa=True,
            ).transpose(1, 2)
        # 重塑输出张量以匹配模型的隐藏层维度
        output = output.reshape(bsz, seqlen, -1)
        # 通过输出线性层进行投影
        return self.o_proj(output)


class FeedForward(nn.Module):
    """
    FeedForward 类实现了前馈神经网络层，通常用于 Transformer 模型中的前馈部分。
    
    参数:
        dim (int): 输入和输出的维度大小。
        intermediate_size (int): 中间层的维度大小，通常比输入维度大。
    """
    def __init__(
        self,
        dim: int,
        intermediate_size: int,
    ):
        super().__init__()
        # 定义线性层，将输入维度投影到中间层维度
        self.up_proj = nn.Linear(dim, intermediate_size, bias=False)
        # 定义线性层，将中间层维度投影回输入维度
        self.down_proj = nn.Linear(intermediate_size, dim, bias=False)
        # 定义门控线性层，用于门控机制
        self.gate_proj = nn.Linear(dim, intermediate_size, bias=False)

    def forward(self, x):
        """
        前向传播函数，执行前馈计算。
        
        参数:
            x (torch.Tensor): 输入张量，形状为 (批次大小, 序列长度, 隐藏层维度)。
        
        返回:
            torch.Tensor: 前馈层的输出，形状为 (批次大小, 序列长度, 隐藏层维度)。
        """
        # 计算门控机制：先通过 gate_proj 投影，然后应用 SiLU 激活函数
        # 计算中间层的输出：先通过 up_proj 投影，然后与门控机制相乘
        # 通过 down_proj 将中间层输出投影回原始维度
        x = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return x


class TransformerBlock(nn.Module):
    """
    TransformerBlock 类实现了 Transformer 模型中的一个基本块，包括自注意力机制和前馈神经网络。
    
    参数:
        layer_id (int): Transformer 层的编号，用于标识当前层。
        args (Qwen2Config): 配置参数，包含模型的各种超参数。
    """
    def __init__(self, layer_id: int, args: Qwen2Config):
        super().__init__()
        # 获取注意力头的数量
        self.n_heads = args.num_attention_heads
        # 获取隐藏层维度大小
        self.dim = args.hidden_size
        # 计算每个注意力头的维度大小
        self.head_dim = args.hidden_size // args.num_attention_heads
        # 初始化自注意力机制
        self.self_attn = Attention(args)
        # 初始化前馈神经网络
        self.mlp = FeedForward(
            dim=args.hidden_size,   # 输入和输出维度
            intermediate_size=args.intermediate_size,  # 中间层维度
        )
        # 当前 Transformer 层的编号
        self.layer_id = layer_id
        # 定义输入层归一化
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        # 定义后注意力层归一化
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        pos_embed: Tuple[torch.Tensor, torch.Tensor],
        start_pos: Optional[Union[int, torch.Tensor]] = None,
    ):
        """
        前向传播函数，执行 Transformer 块中的计算。
        
        参数:
            x (torch.Tensor): 输入张量，形状为 (批次大小, 序列长度, 隐藏层维度)。
            pos_embed (Tuple[torch.Tensor, torch.Tensor]): 位置编码的余弦和正弦部分。
            start_pos (Optional[Union[int, torch.Tensor]]): 起始位置，用于推理模式。默认为 None。
        
        返回:
            torch.Tensor: Transformer 块的输出，形状为 (批次大小, 序列长度, 隐藏层维度)。
        """
        # 通过输入层归一化对输入进行归一化
        # 通过自注意力机制处理归一化后的输入
        # 残差连接：将注意力输出与原始输入相加
        h = x + self.self_attn(self.input_layernorm(x), pos_embed, start_pos=start_pos)

        # 通过后注意力层归一化对残差连接后的结果进行归一化
        # 通过前馈神经网络处理归一化后的结果
        # 残差连接：将前馈输出与注意力输出后的结果相加
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class Qwen2RotaryEmbedding(nn.Module):
    """
    Qwen2RotaryEmbedding 类实现了旋转位置编码 (RoPE) 用于 Qwen2 模型。
    
    参数:
        config (Qwen2Config): 配置参数，包含模型的各种超参数。
        device (torch.device): 张量所在的设备。
    """
    def __init__(self, config: Qwen2Config, device: torch.device):
        super().__init__()
        self.config = config
        # 基础参数，用于计算频率
        base = config.rope_theta
        # 每个注意力头的维度大小
        dim = config.hidden_size // config.num_attention_heads
        # 使用自动混合精度计算逆频率
        with torch.autocast(device_type=device.type, dtype=torch.float32):
            # 计算逆频率向量
            inv_freq = 1.0 / (
                base
                ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
            )
        # 注册逆频率为缓冲区，不作为模型参数保存
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, pos):
        """
        前向传播函数，应用旋转位置编码到输入张量。
        
        参数:
            x (torch.Tensor): 输入张量，形状为 (批次大小, 序列长度, 隐藏层维度)。
            pos (torch.Tensor): 位置张量，形状为 (批次大小, 序列长度)。
        
        返回:
            Tuple[torch.Tensor, torch.Tensor]: 旋转位置编码的余弦和正弦部分，形状均为 (批次大小, 序列长度, 隐藏层维度)。
        """
        # 扩展逆频率张量以匹配输入的批次大小和位置维度
        inv_freq = self.inv_freq[None, :, None].float().expand(pos.shape[0], -1, 1)
        # 扩展位置张量以匹配逆频率的维度
        pos = pos[:, None, :].float()
        device_type = x.device.type
        # 禁用自动混合精度以进行精确计算
        with torch.autocast(device_type=device_type, enabled=False):
            # 计算频率张量
            freqs = (inv_freq.float().to(x.device) @ pos.float()).transpose(1, 2)
            # 将频率张量拼接为 (cos, sin) 对
            emb = torch.cat((freqs, freqs), dim=-1)
            # 计算余弦和正弦编码
            cos = emb.cos()
            sin = emb.sin()
        # 将余弦和正弦编码转换为与输入相同的 dtype
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Transformer(nn.Module):
    """
    Transformer 类实现了 Qwen2 模型的完整 Transformer 结构，包括嵌入层、多层 Transformer 块、归一化层和输出投影层。
    
    参数:
        params (Qwen2Config): 配置参数，包含模型的各种超参数。
        device (torch.device): 模型所在的设备。
    """
    def __init__(self, params: Qwen2Config, device: torch.device):
        super().__init__()
        # 保存配置参数
        self.params = params
        # 词汇表大小
        self.vocab_size = params.vocab_size
        # Transformer 层的数量
        self.n_layers = params.num_hidden_layers

        # 定义词嵌入层，将词汇表中的 token 转换为隐藏层维度大小的向量
        self.embed_tokens = torch.nn.Embedding(params.vocab_size, params.hidden_size)
        # 将嵌入层移动到指定的设备
        with torch.device(device):
            # 初始化旋转位置编码 (RoPE)
            self.rotary_emb = Qwen2RotaryEmbedding(config=params, device=device)

        # 初始化 Transformer 层的列表
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.num_hidden_layers):
            # 为每一层创建一个 TransformerBlock 实例，并添加到列表中
            self.layers.append(TransformerBlock(layer_id, params))

        # 初始化 RMS 归一化层，用于对 Transformer 层的输出进行归一化
        self.norm = RMSNorm(params.hidden_size, eps=params.rms_norm_eps)
        # 如果不使用词嵌入权重绑定，则初始化语言模型头 (lm_head)
        if not params.tie_word_embeddings:
            self.lm_head = nn.Linear(params.hidden_size, params.vocab_size, bias=False)

    def output_proj(self, x):
        """
        输出投影函数，将 Transformer 层的输出投影到词汇表维度。
        
        参数:
            x (torch.Tensor): Transformer 层的输出，形状为 (批次大小, 序列长度, 隐藏层维度)。
        
        返回:
            torch.Tensor: 投影后的输出，形状为 (批次大小, 序列长度, 词汇表大小)。
        """
        if self.params.tie_word_embeddings:
            # 如果使用词嵌入权重绑定，则将输出与嵌入权重矩阵相乘
            return x @ self.embed_tokens.weight.T
        else:
            # 否则，使用语言模型头进行投影
            return self.lm_head(x)

    def forward(self, tokens: torch.Tensor):
        """
        前向传播函数，执行 Transformer 模型的前向计算。
        
        参数:
            tokens (torch.Tensor): 输入的 token 序列，形状为 (批次大小, 序列长度)。
        
        返回:
            torch.Tensor: Transformer 模型的输出，形状为 (批次大小, 序列长度, 词汇表大小)。
        """
        # 获取批次大小和序列长度
        _bsz, seqlen = tokens.shape
        # 通过嵌入层将 token 转换为隐藏层向量
        h = self.embed_tokens(tokens)
        # 生成位置张量，范围从 0 到序列长度 - 1
        pos = torch.arange(0, seqlen, device=tokens.device, dtype=torch.int32)
        # 生成旋转位置编码 (RoPE)
        pos_emb = self.rotary_emb(h, pos[None, :])

        # 构建前向传播管道
        pipe = []
        for layer in self.layers:
            # 为每个 Transformer 块创建一个 lambda 函数，传递位置编码
            pipe.append(lambda x, layer=layer: layer(x, pos_emb))
        # 添加归一化层和输出投影层到管道中
        pipe.append(self.norm.forward)
        pipe.append(self.output_proj)
        # 使用 checkpoint_sequential 进行顺序检查点优化
        return torch.utils.checkpoint.checkpoint_sequential(
            pipe, len(pipe), h, use_reentrant=False
        )

    def inference(self, tokens: torch.Tensor, start_pos: Union[int, torch.Tensor]):
        """
        推理函数，执行 Transformer 模型的推理过程。
        
        参数:
            tokens (torch.Tensor): 输入的 token 序列，形状为 (批次大小, 序列长度)。
            start_pos (Union[int, torch.Tensor]): 起始位置，用于指示当前序列在缓存中的位置。
        
        返回:
            torch.Tensor: Transformer 模型的输出，形状为 (批次大小, 1, 词汇表大小)。
        """
        # 获取批次大小和序列长度
        _bsz, seqlen = tokens.shape
        # 不需要批次大小变量
        del _bsz
        # 通过嵌入层将 token 转换为隐藏层向量
        h = self.embed_tokens(tokens)

        # 生成位置张量，并添加起始位置
        pos = torch.arange(0, seqlen, device=tokens.device, dtype=torch.int32)[None, :]
        if isinstance(start_pos, torch.Tensor):
            pos = pos + start_pos[:, None]
        else:  # int
            pos.add_(start_pos)
        # 生成旋转位置编码 (RoPE)
        pos_emb = self.rotary_emb(h, pos)

        # 逐层应用 Transformer 块
        for layer in self.layers:
            h = layer(h, pos_emb, start_pos=start_pos)

        # 仅保留最后一个 token 的隐藏状态，用于预测下一个 token
        h = h[:, -1:, :]
        # 对最后一个隐藏状态进行归一化
        h = self.norm(h)

        # 通过输出投影层进行投影
        output = self.output_proj(h)
        return output

    def init_kv_cache(
        self,
        max_batch_size: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """
        初始化键 (K) 和值 (V) 的缓存，用于加速推理过程中的自注意力计算。
        
        参数:
            max_batch_size (int): 最大的批次大小。
            max_seq_len (int): 最大的序列长度。
            device (torch.device): 张量所在的设备。
            dtype (torch.dtype): 张量的数据类型。
        """
        for layer in self.layers:
            layer.self_attn.init_kv_cache(
                max_batch_size, max_seq_len, dtype=dtype, device=device
            )

    def del_kv_cache(self):
        """
        删除键 (K) 和值 (V) 的缓存，释放内存。
        """
        for layer in self.layers:
            layer.self_attn.del_kv_cache()

    @classmethod
    def from_pretrained(cls, ckpt_path, device: torch.device):
        """
        从预训练模型加载权重。
        
        参数:
            ckpt_path (str): 预训练模型的路径。
            device (torch.device): 模型要加载到的设备。
        
        返回:
            Transformer: 加载了预训练权重的 Transformer 模型。
        """
        # 读取配置文件
        config_file = Path(ckpt_path) / "config.json"
        with open(config_file, "r") as f:
            config = json.load(f)
        # 初始化 Qwen2Config 对象
        args = Qwen2Config(
            attention_dropout=config["attention_dropout"],
            bos_token_id=config["bos_token_id"],
            eos_token_id=config["eos_token_id"],
            hidden_act=config["hidden_act"],
            hidden_size=config["hidden_size"],
            initializer_range=config["initializer_range"],
            intermediate_size=config["intermediate_size"],
            max_position_embeddings=config["max_position_embeddings"],
            max_window_layers=config["max_window_layers"],
            model_type=config["model_type"],
            num_hidden_layers=config["num_hidden_layers"],
            num_attention_heads=config["num_attention_heads"],
            num_key_value_heads=config["num_key_value_heads"],
            vocab_size=config["vocab_size"],
            rms_norm_eps=config["rms_norm_eps"],
            rope_theta=config["rope_theta"],
            sliding_window=config["sliding_window"],
            use_sliding_window=config["use_sliding_window"],
            use_cache=config["use_cache"],
            tie_word_embeddings=config["tie_word_embeddings"],
            torch_dtype=config["torch_dtype"],
        )
        # 在 'meta' 设备上创建一个模型实例
        with torch.device("meta"):
            model = cls(params=args, device=device)

        # 导入 safetensors 库，用于加载模型权重
        import safetensors.torch

        # 查找所有以 "model" 开头的 safetensors 格式的权重文件，并排序
        model_weight_files = sorted(Path(ckpt_path).glob("model*.safetensors"))
        weights = {}
        for file in model_weight_files:
            # 加载权重文件，并将设备设置为 "cpu" 以避免内存不足
            weights.update(safetensors.torch.load_file(file, device="cpu"))
        # 移除键名中的 "model." 前缀
        weights = {k.replace("model.", ""): v for k, v in weights.items()}
        # 将预训练权重加载到模型中，严格匹配键名
        model.load_state_dict(weights, strict=True, assign=True)
        # 将模型移动到指定的设备
        return model.to(device)
