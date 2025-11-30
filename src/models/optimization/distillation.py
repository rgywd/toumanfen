"""知识蒸馏模块(教学用)"""

from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class KnowledgeDistiller:
    """
    知识蒸馏器
    Teacher-Student 框架
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.5):
        """
        初始化知识蒸馏器

        Args:
            temperature: 蒸馏温度(越高越软化)
            alpha: 蒸馏损失权重(0-1之间)
                   alpha=1表示只用蒸馏损失
                   alpha=0表示只用真实标签损失
        """
        self.temperature = temperature
        self.alpha = alpha

    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算蒸馏损失

        Args:
            student_logits: 学生模型输出
            teacher_logits: 教师模型输出
            labels: 真实标签(可选)

        Returns:
            损失字典
        """
        # 软标签蒸馏损失
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        distillation_loss = F.kl_div(
            soft_student, soft_targets, reduction="batchmean"
        ) * (self.temperature**2)

        result = {"distillation_loss": distillation_loss}

        # 真实标签损失
        if labels is not None:
            hard_loss = F.cross_entropy(student_logits, labels)
            result["hard_loss"] = hard_loss

            # 组合损失
            total_loss = self.alpha * distillation_loss + (1 - self.alpha) * hard_loss
            result["total_loss"] = total_loss
        else:
            result["total_loss"] = distillation_loss

        return result


class DistillationTrainer:
    """蒸馏训练器"""

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        distiller: Optional[KnowledgeDistiller] = None,
        device: str = "cuda",
    ):
        """
        初始化蒸馏训练器

        Args:
            teacher_model: 教师模型(大模型)
            student_model: 学生模型(小模型)
            distiller: 知识蒸馏器
            device: 设备
        """
        self.teacher_model = teacher_model.to(device)
        self.student_model = student_model.to(device)
        self.distiller = distiller or KnowledgeDistiller()
        self.device = device

        # 教师模型设为评估模式并冻结
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def train_step(
        self, batch: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        单步训练

        Args:
            batch: 数据批次
            optimizer: 优化器

        Returns:
            损失字典
        """
        self.student_model.train()

        # 移动数据到设备
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch.get("labels")
        if labels is not None:
            labels = labels.to(self.device)

        # 教师模型推理
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            teacher_logits = teacher_outputs["logits"]

        # 学生模型前向传播
        student_outputs = self.student_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        student_logits = student_outputs["logits"]

        # 计算蒸馏损失
        losses = self.distiller.compute_distillation_loss(
            student_logits=student_logits, teacher_logits=teacher_logits, labels=labels
        )

        # 反向传播
        optimizer.zero_grad()
        losses["total_loss"].backward()
        optimizer.step()

        return {k: v.item() for k, v in losses.items()}

    def save_student(self, save_path: Union[str, Path]) -> None:
        """保存学生模型"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if hasattr(self.student_model, "save"):
            self.student_model.save(save_path)
        else:
            torch.save(self.student_model.state_dict(), save_path)
