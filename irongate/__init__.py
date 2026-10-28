"""IronGate（铁闸）—— 面向大模型的提示注入"信息流"护栏。

原创设计要点（详见 README 与 docs/DESIGN.md）：

1. **把提示注入当作权限越界**，而不是文本分类：
   低信任来源的指令 × 高危能力面 = 越界风险（`ifc.py`）。
2. **双端守恒**：输入侧做权限流分析，输出侧用金丝雀/回响/能力集合
   校验模型是否真的被说服（`ifc.verify_output`）。
3. **零依赖、可复现**：字符 n-gram 朴素贝叶斯 + 13 维线性融合器，
   训练 30 秒内完成，判决的每个分量都能翻译成人话。
"""

from .guard import Guard, OutputVerdict, Verdict
from .model import GuardModel
from .pipeline import DEFAULT_MODEL_PATH, build_dataset, load_model, train
from .policy import Policy, PolicyError, default_policy

__all__ = [
    "Guard",
    "Verdict",
    "OutputVerdict",
    "GuardModel",
    "Policy",
    "PolicyError",
    "default_policy",
    "build_dataset",
    "train",
    "load_model",
    "DEFAULT_MODEL_PATH",
]

__version__ = "1.0.0"
