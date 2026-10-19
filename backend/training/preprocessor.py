#!/usr/bin/env python3
"""
数据集预处理模块
负责验证对话、分析风格、切分训练集并导出训练数据。
仅支持用户自定义角色风格配置，为LoRA微调准备标准化的训练数据。
"""

import os
import json
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from collections import Counter

from training.chat_dataset import normalize_chat_record

logger = logging.getLogger(__name__)


def _validate_path(path_str: str, allowed_base: str = None) -> str:
    """Validate path doesn't contain traversal sequences and is within allowed base."""
    if not path_str:
        raise ValueError("Path cannot be empty")
    # Block path traversal
    if '..' in path_str or '\x00' in path_str:
        raise ValueError("Path contains invalid sequences")
    resolved = Path(path_str).resolve()
    if allowed_base:
        base = Path(allowed_base).resolve()
        if not resolved.is_relative_to(base):
            raise ValueError(f"Path must be within {allowed_base}")
    return str(resolved)


@dataclass
class CharacterStyleConfig:
    """角色风格配置数据类，定义角色的语言特征和对话行为偏好。

    包含填充词、句尾语气词、标点风格、回复长度、提问频率、共情度等参数。
    """
    name: str
    description: str = ""
    
    # 语言特征
    filler_words: List[str] = None
    sentence_enders: List[str] = None
    punctuation_style: str = "standard"
    ellipsis_frequency: float = 0.0
    capitalization: str = "sentence"
    
    # 对话特征
    response_length: str = "medium"
    question_frequency: float = 0.2
    empathy_level: float = 0.5
    formality: float = 0.5
    
    def __post_init__(self):
        if self.filler_words is None:
            self.filler_words = []
        if self.sentence_enders is None:
            self.sentence_enders = []


class DatasetPreprocessor:
    """数据集预处理器，提供对话验证、风格分析和训练集切分功能。"""

    def __init__(self, base_dir: Optional[Path] = None):
        """初始化预处理器。

        Args:
            base_dir: 基础目录路径，默认为当前文件所在目录
        """
        self.base_dir = base_dir or Path(__file__).parent
        self.data_dir = self.base_dir / "data"
        self.styles_file = self.data_dir / "user_styles.json"
        self.data_dir.mkdir(exist_ok=True)
        self.user_styles: Dict[str, CharacterStyleConfig] = {}
        if self.styles_file.exists():
            with open(self.styles_file, 'r', encoding='utf-8') as f:
                self.user_styles = {name: CharacterStyleConfig(**data) for name, data in json.load(f).items()}
    
    def get_style(self, name: str) -> Optional[CharacterStyleConfig]:
        return self.user_styles.get(name)
    
    def list_styles(self) -> Dict[str, CharacterStyleConfig]:
        return self.user_styles.copy()
    
    def analyze_style(self, data: List[List[Dict[str, str]]]) -> Dict[str, Any]:
        """Analyze the already validated messages without reparsing input formats."""
        all_text = []
        response_lengths = []
        sentence_counts = []
        question_count = 0
        total_sentences = 0
        
        for messages in data:
            response = next(message["content"] for message in messages if message["role"] == "assistant")
            all_text.append(response)
            response_lengths.append(len(response))

            sentences = re.split(r'[.!?。！？]', response)
            sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
            sentence_counts.append(len(sentences))
            total_sentences += len(sentences)
            for sentence in sentences:
                if '?' in sentence or '？' in sentence:
                    question_count += 1

        if not all_text:
            return {}
        
        avg_length = sum(response_lengths) / len(response_lengths)
        avg_sentences = sum(sentence_counts) / len(sentence_counts)
        question_ratio = question_count / total_sentences if total_sentences > 0 else 0
        
        all_words = ' '.join(all_text)
        words = re.findall(r'[\w]+', all_words)
        word_freq = Counter(words).most_common(20)
        
        return {
            'total_samples': len(data),
            'avg_response_length': round(avg_length, 1),
            'avg_sentences_per_response': round(avg_sentences, 1),
            'question_ratio': round(question_ratio, 2),
            'common_words': word_freq
        }
    
    def prepare_training_data(
        self,
        raw_data: List[Dict[str, Any]],
        style_config: Optional[CharacterStyleConfig] = None,
        output_name: str = "custom_dataset",
        train_ratio: float = 0.9,
        min_samples: int = 10,
        custom_prompt: Optional[str] = None
    ) -> Tuple[Path, Dict[str, Any]]:
        """准备训练数据：共享格式验证 -> 风格分析 -> 训练/验证集切分 -> 原样保存。

        Args:
            raw_data: 原始对话数据列表
            style_config: 角色风格配置，可选
            output_name: 输出数据集名称
            train_ratio: 训练集占比，默认0.9
            min_samples: 最少有效样本数，不满足则抛异常
            custom_prompt: 自定义系统提示词

        Returns:
            (数据集目录路径, 统计信息字典)

        Raises:
            ValueError: 有效数据不足时抛出
        """
        valid_data = []
        invalid_data = []
        conversations = []
        for item in raw_data:
            try:
                messages = normalize_chat_record(item, default_system_prompt=custom_prompt or "")
            except ValueError as error:
                invalid_data.append((item, str(error)))
            else:
                valid_data.append(item)
                conversations.append(messages)
        
        if len(valid_data) < min_samples:
            raise ValueError(f"有效数据不足，当前: {len(valid_data)}, 要求: {min_samples}")
        
        if invalid_data:
            logger.warning(f"过滤了 {len(invalid_data)} 条无效数据")
        
        style_analysis = self.analyze_style(conversations)
        
        import random
        random.shuffle(valid_data)
        
        split_idx = int(len(valid_data) * train_ratio)
        train_data = valid_data[:split_idx]
        eval_data = valid_data[split_idx:]
        
        dataset_dir = self.data_dir / output_name
        dataset_dir.mkdir(exist_ok=True)
        
        train_path = dataset_dir / "train.json"
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        
        if eval_data:
            eval_path = dataset_dir / "eval.json"
            with open(eval_path, 'w', encoding='utf-8') as f:
                json.dump(eval_data, f, indent=2, ensure_ascii=False)
        
        dataset_info = {
            "file_name": "train.json",
            "file_name_eval": "eval.json" if eval_data else None,
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "history": "history"
            },
            "stats": {
                "total": len(valid_data),
                "train": len(train_data),
                "eval": len(eval_data)
            },
            "style_analysis": style_analysis,
            "custom_prompt": custom_prompt,
            "created_at": datetime.now().isoformat()
        }
        
        info_path = dataset_dir / "dataset_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        stats = {
            "dataset_dir": str(dataset_dir),
            "total_samples": len(valid_data),
            "train_samples": len(train_data),
            "eval_samples": len(eval_data),
            "invalid_samples": len(invalid_data),
            "style_analysis": style_analysis,
            "custom_prompt": custom_prompt
        }
        
        logger.info("=" * 60)
        logger.info(f"数据集准备完成: {output_name}")
        logger.info(f"   总样本数: {len(valid_data)}")
        logger.info(f"   训练集: {len(train_data)}")
        logger.info(f"   验证集: {len(eval_data)}")
        if custom_prompt:
            logger.info(f"   自定义prompt: 已记录")
        logger.info("=" * 60)
        
        return dataset_dir, stats


# 全局实例
_dataset_preprocessor: Optional[DatasetPreprocessor] = None

# 默认数据集扫描目录（所有数据集放在此目录下，子文件夹名为数据集名称）
DEFAULT_SCAN_DIR = Path(__file__).parent / "datasets_import"


def get_dataset_preprocessor() -> DatasetPreprocessor:
    global _dataset_preprocessor
    if _dataset_preprocessor is None:
        _dataset_preprocessor = DatasetPreprocessor()
    return _dataset_preprocessor


def scan_datasets_folder(folder_path: str = None) -> list:
    """扫描指定文件夹，发现所有有效的数据集子文件夹。
    
    每个子文件夹必须包含至少一个 JSON 文件（train.json 或 data.json 等）。
    子文件夹名即为数据集名称。
    
    Args:
        folder_path: 要扫描的文件夹路径，默认使用 DEFAULT_SCAN_DIR
        
    Returns:
        [{name, path, file_count, files, valid}] 列表
    """
    scan_path = Path(folder_path) if folder_path else DEFAULT_SCAN_DIR
    if not scan_path.exists():
        return []
    
    results = []
    for subdir in sorted(scan_path.iterdir()):
        if not subdir.is_dir():
            continue
        
        json_files = list(subdir.glob("*.json"))
        if not json_files:
            continue
        
        results.append({
            "name": subdir.name,
            "path": str(subdir),
            "file_count": len(json_files),
            "files": [f.name for f in json_files],
            "valid": True
        })
    
    return results


def import_dataset_from_folder(source_path: str, dataset_name: str = None) -> dict:
    """Validate the source and read statistics before copying the dataset."""
    import shutil

    allowed_dirs = [Path(__file__).parent.parent.resolve()]
    lab_root = (os.getenv("MULTIPERSONAL_LAB_ROOT") or os.getenv("QQCHAT_LAB_ROOT", "")).strip()
    if lab_root and Path(lab_root).exists():
        allowed_dirs.append(Path(lab_root).resolve())
    src = Path(_validate_path(source_path))
    if not any(src.is_relative_to(base) for base in allowed_dirs):
        raise ValueError("源路径不在允许的目录范围内")
    if not src.is_dir():
        raise FileNotFoundError(f"数据集文件夹不存在: {source_path}")

    stats = {"total": 0, "train": 0, "eval": 0}
    info_path = src / "dataset_info.json"
    if info_path.exists():
        with open(info_path, 'r', encoding='utf-8') as f:
            stats = json.load(f).get("stats", stats)
    else:
        for fname in ["train.json", "data.json", "dataset.json"]:
            fpath = src / fname
            if fpath.exists():
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                count = len(data) if isinstance(data, list) else 0
                stats["total"] = count
                stats["train"] = count
                break

    name = dataset_name or src.name
    preprocessor = get_dataset_preprocessor()
    dest = preprocessor.data_dir / name
    counter = 1
    while dest.exists():
        dest = preprocessor.data_dir / f"{name}_{counter}"
        counter += 1
    shutil.copytree(src, dest)
    return {"success": True, "name": dest.name, "path": str(dest), "stats": stats}
