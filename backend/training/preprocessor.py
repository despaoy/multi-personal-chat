#!/usr/bin/env python3
"""
数据集预处理模块
负责加载原始对话数据（JSON/JSONL/TXT），进行清洗、验证、风格分析和训练集切分。
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
    """数据集预处理器，提供原始数据加载、文本清洗、对话验证、风格分析和训练集切分功能。"""

    def __init__(self, base_dir: Optional[Path] = None):
        """初始化预处理器。

        Args:
            base_dir: 基础目录路径，默认为当前文件所在目录
        """
        self.base_dir = base_dir or Path(__file__).parent
        self.data_dir = self.base_dir / "data"
        self.styles_file = self.data_dir / "user_styles.json"
        self.data_dir.mkdir(exist_ok=True)
        self._load_user_styles()
    
    def _load_user_styles(self):
        self.user_styles: Dict[str, CharacterStyleConfig] = {}
        if self.styles_file.exists():
            try:
                with open(self.styles_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for name, style_data in data.items():
                        self.user_styles[name] = CharacterStyleConfig(**style_data)
            except Exception as e:
                logger.warning(f"加载用户风格失败: {e}")
    
    def _save_user_styles(self):
        try:
            data = {}
            for name, style in self.user_styles.items():
                data[name] = {
                    "name": style.name,
                    "description": style.description,
                    "filler_words": style.filler_words,
                    "sentence_enders": style.sentence_enders,
                    "punctuation_style": style.punctuation_style,
                    "ellipsis_frequency": style.ellipsis_frequency,
                    "capitalization": style.capitalization,
                    "response_length": style.response_length,
                    "question_frequency": style.question_frequency,
                    "empathy_level": style.empathy_level,
                    "formality": style.formality
                }
            with open(self.styles_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存用户风格失败: {e}")
    
    def add_user_style(self, name: str, config: CharacterStyleConfig):
        self.user_styles[name] = config
        self._save_user_styles()
        logger.info(f"添加用户风格: {name}")
    
    def get_style(self, name: str) -> Optional[CharacterStyleConfig]:
        return self.user_styles.get(name)
    
    def list_styles(self) -> Dict[str, CharacterStyleConfig]:
        return self.user_styles.copy()
    
    def load_raw_data(self, file_path: Path) -> List[Dict[str, Any]]:
        """加载原始对话数据，支持JSON、JSONL和TXT格式。

        Args:
            file_path: 数据文件路径

        Returns:
            解析后的对话数据列表

        Raises:
            FileNotFoundError: 文件不存在时抛出
            ValueError: 文件格式不支持时抛出
        """
        if not file_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        data = []
        
        if file_path.suffix == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                if isinstance(content, list):
                    data = content
                elif isinstance(content, dict) and 'data' in content:
                    data = content['data']
        
        elif file_path.suffix == '.jsonl':
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
        
        elif file_path.suffix == '.txt':
            data = self._parse_txt_file(file_path)
        
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")
        
        logger.info(f"加载原始数据: {len(data)} 条")
        return data
    
    def _parse_txt_file(self, file_path: Path) -> List[Dict[str, Any]]:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            current_conv = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith(('User:', 'user:', '用户:')):
                    if current_conv:
                        data.append({
                            'conversations': current_conv
                        })
                        current_conv = []
                    content = line.split(':', 1)[1].strip()
                    current_conv.append({
                        'role': 'user',
                        'content': content
                    })
                elif line.startswith(('Assistant:', 'assistant:', '助手:')):
                    content = line.split(':', 1)[1].strip()
                    current_conv.append({
                        'role': 'assistant',
                        'content': content
                    })
            
            if current_conv:
                data.append({
                    'conversations': current_conv
                })
        
        return data
    
    def clean_text(self, text: str) -> str:
        if not text:
            return ""

        text = re.sub(r'\s+', ' ', text)
        # 保留中文标点，不替换为英文标点（中文聊天机器人需要中文标点）
        text = re.sub(r'[\x00-\x1F\x7F]', '', text)

        return text.strip()
    
    def validate_conversation(self, conv: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """校验单条对话数据的格式合法性。

        支持的格式：
        - ShareGPT多轮对话: conversations列表（from: human/gpt, value）
        - Qwen格式: conversations列表（role: user/assistant, content）
        - Alpaca格式: instruction/output
        - Prompt格式: prompt/response

        Args:
            conv: 单条对话数据

        Returns:
            (是否合法, 错误信息)，合法时错误信息为None
        """
        try:
            if 'conversations' in conv:
                conversations = conv['conversations']
                if len(conversations) < 2:
                    return False, "对话轮次不足"

                for i, msg in enumerate(conversations):
                    # ShareGPT格式: from/value
                    if 'from' in msg and 'value' in msg:
                        if msg['from'] not in ('human', 'gpt', 'user', 'assistant'):
                            return False, f"第{i}轮对话角色无效: {msg.get('from')}"
                        if not msg['value'].strip():
                            return False, f"第{i}轮对话内容为空"
                    # Qwen格式: role/content
                    elif 'role' in msg and 'content' in msg:
                        if msg['role'] not in ('user', 'assistant', 'system'):
                            return False, f"第{i}轮对话角色无效: {msg.get('role')}"
                        if not msg['content'].strip():
                            return False, f"第{i}轮对话内容为空"
                    else:
                        return False, f"第{i}轮对话格式错误: 缺少from/value或role/content"

            elif 'instruction' in conv:
                if not conv.get('output'):
                    return False, "缺少output字段"

            elif 'prompt' in conv and 'response' in conv:
                pass

            else:
                return False, "不支持的数据格式"

            return True, None

        except Exception as e:
            return False, str(e)
    
    def analyze_style(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        all_text = []
        response_lengths = []
        sentence_counts = []
        question_count = 0
        total_sentences = 0
        
        for item in data:
            response = ""
            if 'output' in item:
                response = item['output']
            elif 'response' in item:
                response = item['response']
            elif 'conversations' in item:
                convs = item['conversations']
                for conv in convs:
                    if conv.get('role') == 'assistant' or conv.get('from') == 'gpt':
                        response = conv.get('content', '') or conv.get('value', '')
                        break
            
            if response:
                all_text.append(response)
                response_lengths.append(len(response))
                
                sentences = re.split(r'[.!?。！？]', response)
                sentences = [s.strip() for s in sentences if s.strip()]
                sentence_counts.append(len(sentences))
                total_sentences += len(sentences)
                
                for s in sentences:
                    if '?' in s or '？' in s:
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
        """准备训练数据：验证 -> 清洗 -> 风格分析 -> 训练/验证集切分 -> 保存。

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
        
        for item in raw_data:
            is_valid, error = self.validate_conversation(item)
            if is_valid:
                valid_data.append(item)
            else:
                invalid_data.append((item, error))
        
        if len(valid_data) < min_samples:
            raise ValueError(f"有效数据不足，当前: {len(valid_data)}, 要求: {min_samples}")
        
        if invalid_data:
            logger.warning(f"过滤了 {len(invalid_data)} 条无效数据")
        
        style_analysis = self.analyze_style(valid_data)
        
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
    """将数据集文件夹导入到 data_dir。

    直接复制或链接整个文件夹到 preprocessor 的 data_dir 下。

    Args:
        source_path: 源数据集文件夹路径
        dataset_name: 数据集名称，默认使用源文件夹名

    Returns:
        {success, name, path, stats}
    """
    import shutil

    # Validate source_path to prevent path traversal
    backend_dir = str(Path(__file__).parent.parent.resolve())
    allowed_dirs = [backend_dir]
    # Allow a configured lab root when present. The default remains the
    # backend directory only, so local deployments are fail-closed.
    lab_root = (os.getenv("MULTIPERSONAL_LAB_ROOT") or os.getenv("QQCHAT_LAB_ROOT", "")).strip()
    if lab_root and Path(lab_root).exists():
        allowed_dirs.append(str(Path(lab_root).resolve()))
    try:
        validated_path = _validate_path(source_path)
        resolved = Path(validated_path)
        if not any(resolved.is_relative_to(Path(d)) for d in allowed_dirs):
            raise ValueError("源路径不在允许的目录范围内")
    except ValueError as e:
        raise ValueError(f"源路径验证失败: {e}")

    src = Path(source_path)
    if not src.exists() or not src.is_dir():
        raise FileNotFoundError(f"数据集文件夹不存在: {source_path}")
    
    name = dataset_name or src.name
    preprocessor = get_dataset_preprocessor()
    dest = preprocessor.data_dir / name
    
    # 如果目标已存在，追加序号
    counter = 1
    while dest.exists():
        dest = preprocessor.data_dir / f"{name}_{counter}"
        counter += 1
    
    # 复制文件夹
    shutil.copytree(src, dest)
    
    # 读取或生成 dataset_info.json
    info_path = dest / "dataset_info.json"
    stats = {"total": 0, "train": 0, "eval": 0}
    if info_path.exists():
        try:
            with open(info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
                stats = info.get("stats", stats)
        except Exception as e:
            # H3 fix: 此前静默吞噬，dataset_info.json 读取失败时无法排障
            logger.warning("读取 dataset_info.json 失败: %s", e)
    else:
        # 尝试从 JSON 文件推断统计信息
        for fname in ["train.json", "data.json", "dataset.json"]:
            fpath = dest / fname
            if fpath.exists():
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    count = len(data) if isinstance(data, list) else 0
                    stats["total"] = count
                    stats["train"] = count
                    break
                except Exception as e:
                    # H3 fix: 此前静默吞噬，数据集统计推断失败时无法排障
                    logger.warning("从 %s 推断统计信息失败: %s", fname, e)
    
    return {
        "success": True,
        "name": dest.name,
        "path": str(dest),
        "stats": stats
    }
