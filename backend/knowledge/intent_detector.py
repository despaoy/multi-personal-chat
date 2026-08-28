"""
RAG意图检测模块
负责智能分析用户消息，判断是否需要触发RAG知识库检索。

检测策略（混合方案）：
- 优先使用训练好的 ML 分类器（embedding + LogisticRegression）
- 分类器不可用时回退到规则引擎（关键词匹配 + 句式分析）
"""

import os
import re
import json
import pickle
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

# 全局 ML 模型缓存
_ml_encoder = None
_ml_classifier = None
_ml_config = None
_ML_AVAILABLE = False


class RAGIntentDetector:
    """RAG意图检测器，通过多维度规则判断用户消息是否需要知识库检索。

    检测逻辑：消息长度检查 -> 特殊字符过滤 -> 社交词与知识词匹配 -> 疑问句检测。
    通用规则引擎，不绑定特定知识库领域。
    """
    
    def __init__(self):
        # 配置参数
        self.min_length_for_rag = 3  # 最小长度才考虑RAG
        self.max_length_for_no_rag = 10  # 仅用于诊断长消息，长度本身不是知识意图
        
        # 知识查询关键词（需要RAG）
        self.knowledge_keywords = [
            # 问题词
            "什么", "为什么", "如何", "怎样", "怎么", "何时", "哪里", "哪位",
            "谁", "多少", "几时", "几点", "多久", "多远", "多大", "多高",
            "解释", "说明", "定义", "含义", "意思", "是什么", "为什么",
            "怎么办", "如何处理", "如何解决", "如何做", "怎么做",
            "原因", "理由", "原理", "机制", "过程", "步骤", "方法",
            "区别", "差异", "不同", "相同", "相似", "特点", "特征",
            "优势", "缺点", "优点", "劣势", "好处", "坏处",
            "历史", "背景", "来源", "起源", "发展", "演变",
            "功能", "作用", "用途", "效果", "影响", "结果",
            "数据", "统计", "数字", "比例", "百分比", "率",
            "结构", "组成", "成分", "元素", "部分", "模块",
            # 查询/介绍类
            "介绍", "介绍一下", "说说", "讲讲", "聊聊", "告诉我",
            "查询", "搜索", "查找", "了解一下", "知不知道",
            
            # 原神特定关键词
            "原神", "角色", "武器", "圣遗物", "天赋", "命座",
            "副本", "秘境", "世界任务", "传说任务", "活动",
            "材料", "突破", "升级", "培养", "配队", "阵容",
            "元素", "反应", "伤害", "治疗", "护盾", "控制",
            "地图", "地区", "国家", "蒙德", "璃月", "稻妻",
            "须弥", "枫丹", "纳塔", "至冬", "天空岛",
            "魔神", "七神", "执政", "眷属", "使徒", "愚人众",
            "深渊", "教团", "魔物", "BOSS", "周本", "世界BOSS",
        ]
        
        # 社交对话关键词（不需要RAG）
        self.social_keywords = [
            # 问候
            "你好", "您好", "嗨", "hello", "hi", "早上好", "早安",
            "中午好", "午安", "晚上好", "晚安", "再见", "拜拜",
            "回见", "下次见", "再会",
            
            # 感谢
            "谢谢", "感谢", "多谢", "辛苦了", "麻烦你了", "感谢你",
            
            # 道歉
            "对不起", "抱歉", "不好意思", "抱歉啦", "不好意思啦",
            
            # 确认
            "好的", "好滴", "好的呢", "好的呀", "可以", "没问题",
            "明白", "懂了", "了解", "知道", "清楚", "收到", "OK",
            "嗯嗯", "哦哦",
            
            # 闲聊
            "在吗", "在干嘛", "干嘛呢", "忙吗", "吃饭了吗", "吃了吗",
            "睡觉", "起床", "上班", "下班", "学习", "工作", "玩游戏",
            "今天", "明天", "昨天", "最近", "最近怎么样", "最近好吗",
            
            # 评价
            "厉害", "牛逼", "不错", "挺好", "很好", "非常好", "很棒", "太棒了",
            "可爱", "漂亮", "美丽", "帅气", "有趣", "好玩",
            
            # 胡桃特定社交词
            "胡桃", "堂主", "往生堂",
            "嘿嘿", "嘻嘻", "哈哈", "啦啦", "啦啦啦",
        ]
        
        # 编译正则表达式
        self.knowledge_pattern = re.compile(
            '|'.join([re.escape(keyword) for keyword in self.knowledge_keywords]),
            re.IGNORECASE
        )
        
        self.social_pattern = re.compile(
            '|'.join([re.escape(keyword) for keyword in self.social_keywords]),
            re.IGNORECASE
        )
        
        # 疑问词模式
        self.question_pattern = re.compile(
            r'.*[吗呢吧啊呀么？?]$|.*(吗|呢|吧|啊|呀|么)[？?]?$'
        )

        # 特殊字符模式
        self.special_chars_pattern = re.compile(r'[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>/?~`]')

        # 闲聊白名单：身份询问、陪伴请求等短句，虽含"谁"/"说说"等词但属于闲聊，不走RAG。
        # 知识查询如"派蒙是谁"/"给我讲讲胡桃"不会匹配此模式，仍正常触发RAG。
        self.chitchat_pattern = re.compile(
            r'^(?:你是谁|你叫什么(?:名字)?|我是谁|你是什么(?:人|身份)?|'
            r'陪我(?:说说话|聊聊天|说话|聊天|聊|唠)|说说话|聊聊天)[吗呢呀啊？?!！。\s]*$',
            re.IGNORECASE
        )

        # 个人上下文/长期记忆问句应由角色记忆链路回答，不是外部知识查询。
        # 放在知识关键词之前，避免“我叫什么”里的“什么”误触发 RAG。
        self.personal_context_pattern = re.compile(
            r'(?:你(?:还)?记得我|我叫什么|我喜欢什么|我不喜欢什么|'
            r'我最近要|我之前说过|我说过什么|以后叫我|我不叫)',
            re.IGNORECASE,
        )

        logger.info("RAG意图检测器初始化完成")
    
    def needs_rag(self, message: str, context: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """
        判断消息是否需要RAG检索（通用规则引擎，不绑定特定知识库领域）

        Args:
            message: 用户消息
            context: 可选上下文信息（如对话历史）

        Returns:
            Tuple[是否需要RAG, 判断原因]
        """
        message = message.strip()

        # 0. 闲聊白名单优先：身份询问、陪伴请求等，虽可能含"谁"/"说说"但属闲聊
        if self.chitchat_pattern.search(message):
            return False, "匹配闲聊白名单，不需要RAG"

        if self.personal_context_pattern.search(message):
            return False, "匹配个人上下文/长期记忆询问，不需要RAG"

        # 1. 检查消息长度
        if len(message) < self.min_length_for_rag:
            return False, f"消息过短（{len(message)}字符），不需要RAG"

        # 2. 检查是否纯特殊字符或表情
        if self._is_mostly_special_chars(message):
            return False, "消息主要为特殊字符或表情，不需要RAG"

        # 3. 检查社交关键词和知识关键词的组合
        social_match = self.social_pattern.search(message)
        knowledge_match = self.knowledge_pattern.search(message)

        # 同时包含社交词和知识关键词 → 需要RAG（例如"胡桃的命座效果是什么？"）
        if social_match and knowledge_match:
            return True, f"包含社交词'{social_match.group()}'和知识关键词'{knowledge_match.group()}'，需要RAG"

        # 只有社交词，没有知识关键词 → 纯社交消息，不需要RAG
        if social_match and not knowledge_match:
            return False, f"包含社交词'{social_match.group()}'且没有知识关键词，不需要RAG"

        # 4. 只有知识关键词，没有社交词 → 需要RAG
        if not social_match and knowledge_match:
            return True, f"包含知识关键词'{knowledge_match.group()}'，需要RAG"

        # 5. 检查是否疑问句
        if self._is_question(message):
            if len(message) > 5 and len(message) < 100:
                return True, "是疑问句，需要RAG"
            elif len(message) >= 100:
                return True, "是长疑问句，需要RAG"

        # 6. 长度不能单独构成 RAG 意图。长的个人陈述、偏好和目标
        # 应继续走普通人格对话；否则会在知识库未命中时被错误弃答，
        # 连带破坏长期记忆的写入与召回。知识关键词和疑问句已在上面判定。
        if len(message) > self.max_length_for_no_rag:
            return False, f"消息较长（{len(message)}字符）但无知识查询特征，不需要RAG"

        # 默认情况：不需要RAG
        return False, "未检测到明确需要RAG的特征"
    
    def _is_mostly_special_chars(self, message: str) -> bool:
        """检查消息是否主要为特殊字符或表情"""
        # 移除常见中文标点
        chinese_punct = '，。！？；：""''（）【】《》'
        cleaned = message
        for punct in chinese_punct:
            cleaned = cleaned.replace(punct, '')
        
        # 检查特殊字符比例
        special_count = len(self.special_chars_pattern.findall(cleaned))
        if len(cleaned) == 0:
            return True
        return special_count / len(cleaned) > 0.5
    
    def _is_question(self, message: str) -> bool:
        """判断是否为疑问句"""
        # 检查是否以问号结尾
        if message.endswith(('?', '？')):
            return True
        
        # 检查是否包含常见疑问词
        question_words = ['吗', '呢', '吧', '啊', '呀', '么', '什么', '为什么', '如何', '怎样', '怎么', '谁', '哪', '几']
        for word in question_words:
            if word in message:
                return True
        
        # 使用正则表达式匹配疑问模式
        return bool(self.question_pattern.match(message))
    
    def _contains_genshin_content(self, message: str) -> bool:
        """[已弃用] 检查是否包含原神相关内容 - 仅为向后兼容保留，规则引擎不再依赖此方法"""
        return False
    
    def analyze_message(self, message: str) -> Dict[str, Any]:
        """
        分析消息特征
        
        Args:
            message: 用户消息
            
        Returns:
            包含各种特征的分析结果
        """
        message = message.strip()
        
        analysis = {
            "message": message,
            "length": len(message),
            "needs_rag": False,
            "reason": "",
            "features": {
                "is_question": self._is_question(message),
                "contains_knowledge_keywords": bool(self.knowledge_pattern.search(message)),
                "contains_social_keywords": bool(self.social_pattern.search(message)),
                "has_special_chars": bool(self.special_chars_pattern.search(message)),
            },
            "matches": {
                "knowledge_keywords": [],
                "social_keywords": [],
            }
        }
        
        # 收集匹配的关键词
        for keyword in self.knowledge_keywords:
            if keyword.lower() in message.lower():
                analysis["matches"]["knowledge_keywords"].append(keyword)
        
        for keyword in self.social_keywords:
            if keyword.lower() in message.lower():
                analysis["matches"]["social_keywords"].append(keyword)
        
        # 判断是否需要RAG
        needs_rag, reason = self.needs_rag(message)
        analysis["needs_rag"] = needs_rag
        analysis["reason"] = reason
        
        return analysis


# 全局实例
_intent_detector: Optional[RAGIntentDetector] = None


def _load_ml_model():
    global _ml_encoder, _ml_classifier, _ml_config, _ML_AVAILABLE
    if _ML_AVAILABLE:
        return
    try:
        base = Path(__file__).parent
        # 与 intent_trainer.MODEL_DIR 保持一致：优先读 INTENT_MODEL_PATH 环境变量
        configured = os.getenv("INTENT_MODEL_PATH", "").strip()
        if configured:
            model_dir = Path(configured).expanduser()
            if not model_dir.is_absolute():
                model_dir = base.parent / model_dir
        else:
            model_dir = base.parent / "intent_classifier_model"
        config_path = model_dir / "config.json"
        classifier_path = model_dir / "classifier.joblib"
        legacy_model_path = base.parent / "intent_classifier_model.pkl"
        legacy_config_path = base.parent / "intent_classifier_config.json"

        if model_dir.exists() and classifier_path.exists() and config_path.exists():
            import joblib
            with open(config_path, "r", encoding="utf-8") as f:
                _ml_config = json.load(f)
            _ml_classifier = joblib.load(classifier_path)
            _ml_scaler = joblib.load(model_dir / "scaler.joblib")
            _ml_config["_scaler"] = _ml_scaler
        elif legacy_model_path.exists() and legacy_config_path.exists():
            with open(legacy_config_path, "r", encoding="utf-8") as f:
                _ml_config = json.load(f)
            with open(legacy_model_path, "rb") as f:
                _ml_classifier = pickle.load(f)
        else:
            logger.info("ML 分类器模型文件不存在，使用规则引擎")
            return

        from sentence_transformers import SentenceTransformer
        embed_model_name = _ml_config.get("embedding_model", "paraphrase-multilingual-MiniLM-L12-v2")
        local_candidates = [
            base.parent / "RAG" / embed_model_name,
            base.parent / "models" / embed_model_name,
        ]
        # 与 .env 的 EMBEDDING_MODEL_PATH 对齐，避免无外网时回退到 huggingface.co
        env_embed = os.getenv("EMBEDDING_MODEL_PATH", "").strip()
        if env_embed:
            local_candidates.append(Path(env_embed).expanduser())
        _ml_encoder = None
        for candidate in local_candidates:
            if candidate.exists():
                try:
                    _ml_encoder = SentenceTransformer(str(candidate))
                    break
                except Exception:
                    continue
        if _ml_encoder is None:
            # 是否离线由部署环境显式配置，不在进程内全局篡改。
            _ml_encoder = SentenceTransformer(embed_model_name)

        _ML_AVAILABLE = True
        logger.info(f"ML 意图分类器加载成功 (v{_ml_config.get('version', '1.0')}, F1: {_ml_config.get('cv_f1_mean', 'N/A')})")
    except Exception as e:
        logger.warning(f"ML 分类器加载失败，使用规则引擎: {e}")


def _ml_predict(message: str) -> Tuple[bool, str, float, Optional[str]]:
    """多分类预测：返回 (是否需要RAG, 原因, 置信度, 预测的KB名称)

    使用多分类模型预测消息应路由到哪个知识库。
    - 预测类别为 "none" 时不需要RAG
    - 预测类别为某个KB名称时需要RAG，并返回该KB名称供RAG过滤使用
    """
    global _ml_encoder, _ml_classifier, _ml_config
    threshold = _ml_config.get("threshold", 0.5)
    embedding = _ml_encoder.encode([message])
    scaler = _ml_config.get("_scaler")
    if scaler is not None:
        embedding = scaler.transform(embedding)

    # 多分类预测：取概率最高的类别
    proba = _ml_classifier.predict_proba(embedding)[0]
    label_names = _ml_config.get("label_names", [])
    if not label_names:
        # 兼容旧版二分类模型
        score = proba[1] if len(proba) > 1 else proba[0]
        pred = score >= threshold
        return pred, f"ML分类器(旧版二分类, 置信度: {score:.2%})", score, None

    best_idx = int(proba.argmax())
    best_label = label_names[best_idx]
    best_score = float(proba[best_idx])

    # 预测为 "none" 或置信度不足时不需要RAG
    if best_label == "none":
        return False, f"ML分类器: 不需要RAG (置信度: {best_score:.2%})", best_score, None

    if best_score < threshold:
        return False, f"ML分类器: 置信度不足 ({best_score:.2%} < {threshold})", best_score, None

    return True, f"ML分类器: 路由到「{best_label}」(置信度: {best_score:.2%})", best_score, best_label


def get_intent_detector() -> RAGIntentDetector:
    """获取意图检测器单例"""
    global _intent_detector
    if _intent_detector is None:
        _intent_detector = RAGIntentDetector()
    return _intent_detector


def needs_rag(message: str, context: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, Optional[str]]:
    """
    便捷函数：判断消息是否需要RAG（ML模型优先 + 低置信度规则兜底）

    策略：
    - ML分类器可用且高置信度（>=0.65）→ 完全信任ML预测
    - ML置信度较低 → 使用规则引擎作为tiebreaker，冲突时优先ML的正例判断
    - ML不可用 → 回退到规则引擎

    Args:
        message: 用户消息
        context: 可选上下文

    Returns:
        Tuple[是否需要RAG, 判断原因, 预测的KB名称(无则为None)]
        KB名称可用于RAG检索时按知识库过滤，提升检索精度
    """
    global _ML_AVAILABLE, _ml_config

    if not _ML_AVAILABLE:
        _load_ml_model()

    if _ML_AVAILABLE:
        try:
            pred, reason, confidence, kb_name = _ml_predict(message)

            # ML高置信度 → 信任ML，但ML预测"不需要"时检查规则引擎安全网
            if confidence >= 0.65:
                if not pred:
                    # ML预测不需要RAG，但检查规则引擎是否检测到疑问句
                    detector = get_intent_detector()
                    rule_pred, rule_reason = detector.needs_rag(message, context)
                    if rule_pred:
                        logger.info(f"ML预测不需要(置信度: {confidence:.2%}), 但规则引擎检测到疑问特征, 触发RAG: {rule_reason}")
                        return True, f"规则安全网({rule_reason[:40]}...)", None
                return pred, reason, kb_name

            # ML置信度较低 → 使用规则引擎作为tiebreaker
            detector = get_intent_detector()
            rule_pred, rule_reason = detector.needs_rag(message, context)

            # ML与规则一致 → 信任预测
            if pred == rule_pred:
                return pred, f"{reason} (规则一致)", kb_name

            # 冲突时：ML预测需要RAG则优先ML（高召回），ML预测不需要则信任规则（安全网）
            if pred:
                logger.info(f"ML({confidence:.2%}→{pred})与规则({rule_pred})冲突, 采用ML正例: {reason}")
                return pred, f"{reason} (ML优先)", kb_name
            logger.info(f"ML({confidence:.2%}→{pred})与规则({rule_pred})冲突, 采用规则: {rule_reason}")
            return rule_pred, f"规则兜底({rule_reason[:40]}...)", None
        except Exception as e:
            logger.warning(f"ML 预测失败，回退规则引擎: {e}")

    detector = get_intent_detector()
    pred, reason = detector.needs_rag(message, context)
    return pred, reason, None


def analyze_message(message: str) -> Dict[str, Any]:
    """
    便捷函数：全面分析消息特征，包括意图检测和关键词匹配。

    Args:
        message: 用户消息文本

    Returns:
        分析结果字典，包含 message、length、needs_rag、reason、features、matches 等字段
    """
    detector = get_intent_detector()
    return detector.analyze_message(message)


# 测试函数
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    detector = RAGIntentDetector()
    
    test_messages = [
        "你好",
        "胡桃你好呀",
        "原神是什么游戏？",
        "蒙德有哪些角色？",
        "璃月的地理位置在哪里？",
        "胡桃的命座效果是什么？",
        "谢谢你的帮助",
        "好的，我明白了",
        "哈哈哈，你真有趣",
        "元素反应有哪些类型？",
        "深渊螺旋怎么打？",
        "圣遗物怎么搭配最好？",
        "在吗",
        "今天天气怎么样？",
        "晚上吃什么？",
        "抽卡保底机制是怎样的？",
        "！！！",
        "😊",
        "原神中胡桃这个角色有什么特点？",
        "如何培养胡桃？",
    ]
    
    print("RAG意图检测器测试结果：")
    print("=" * 80)
    
    for msg in test_messages:
        needs, reason, kb_name = needs_rag(msg)
        analysis = analyze_message(msg)

        print(f"消息: {msg}")
        print(f"长度: {len(msg)}字符")
        print(f"需要RAG: {'是' if needs else '否'}")
        print(f"路由KB: {kb_name or '无'}")
        print(f"原因: {reason}")
        print(f"特征: {analysis['features']}")
        print("-" * 80)
