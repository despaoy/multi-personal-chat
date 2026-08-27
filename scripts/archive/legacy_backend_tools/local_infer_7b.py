"""
历史胡桃本地 4bit 推理脚本
使用Qwen3-8B基础模型加载胡桃LoRA适配器进行本地推理，适用于RTX 4060 Laptop 8GB。
通过BitsAndBytes的NF4量化将推理显存控制在~4GB。

使用方法：
1. 从AutoDL下载训练好的LoRA到 backend/loras/hutao_lora_7b/final/
2. 确保安装了 bitsandbytes
3. 运行: python local_infer_7b.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path
import os as _os

BASE = Path(__file__).parent.parent
BASE_MODEL_PATH = _os.getenv("BASE_MODEL_PATH", str(BASE / "models/Qwen3-8B-Instruct"))
LORA_PATH = str(Path(__file__).parent.parent / "loras/hutao_lora_7b/final")

SYSTEM_PROMPT = """严格按照以下角色回复用户：
你是胡桃，璃月港"往生堂"第七十七代堂主。古灵精怪、元气满满，自称"小巷派暗黑诗人"。
始终用"本堂主"自称，语气活泼俏皮，多用"呀""啦""喽""哟""嘿嘿"等语气词。
保持轻松幽默，对生死话题用豁达乐观态度。
打招呼："哟，找本堂主有何贵干呀？往生堂第七十七代堂主就是胡桃我啦！"
"""


class HutaoInferencer:
    """胡桃推理器，加载4bit量化的Qwen3-8B + 胡桃LoRA模型，
    提供单轮对话生成和交互式多轮对话功能。"""

    def __init__(self):
        """初始化推理器，自动检测CUDA设备。"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None

    def load(self):
        """加载4bit量化的基础模型和LoRA适配器。

        Returns:
            bool: 加载成功返回True，模型文件缺失或加载失败返回False
        """
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {torch.cuda.get_device_name(0)} ({vram:.0f}GB)")
        print("加载 Qwen3-8B (4bit量化)...")

        if not Path(BASE_MODEL_PATH).exists():
            print(f"基础模型不存在: {BASE_MODEL_PATH}")
            print("请先下载: huggingface-cli download Qwen/Qwen3-8B --local-dir models/Qwen3-8B-Instruct")
            return False

        if not Path(LORA_PATH).exists():
            print(f"LoRA模型不存在: {LORA_PATH}")
            print("请从AutoDL下载训练好的LoRA到该目录")
            return False

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            quantization_config=nf4_config,
            device_map="auto",
        )

        self.model = PeftModel.from_pretrained(base_model, LORA_PATH)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"模型加载完成！显存占用: {allocated:.1f}GB")
        return True

    def chat(self, user_input: str, max_new_tokens: int = 256, temperature: float = 0.8):
        """使用胡桃风格生成单轮回复。

        Args:
            user_input: 用户输入文本
            max_new_tokens: 最大生成token数，默认256
            temperature: 生成温度，默认0.8

        Returns:
            str: 胡桃风格的回复文本
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        reply = self.tokenizer.decode(
            output[0][input_ids.shape[1]:], skip_special_tokens=True
        ).strip()

        return reply


def interactive_chat():
    bot = HutaoInferencer()
    if not bot.load():
        return

    print()
    print("=" * 50)
    print("胡桃 (Qwen3-8B + LoRA) - 对话模式")
    print("输入 'quit' 退出, 'clear' 清除上下文")
    print("=" * 50)

    history = []
    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见啦~")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("胡桃: 再见啦~本堂主先走一步！")
            break
        if user_input.lower() == "clear":
            history.clear()
            print("(上下文已清除)")
            continue

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history[-6:])
        messages.append({"role": "user", "content": user_input})

        input_ids = bot.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(bot.device)

        with torch.no_grad():
            output = bot.model.generate(
                input_ids,
                max_new_tokens=256,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                pad_token_id=bot.tokenizer.eos_token_id,
            )

        reply = bot.tokenizer.decode(
            output[0][input_ids.shape[1]:], skip_special_tokens=True
        ).strip()

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})

        print(f"胡桃: {reply}")


if __name__ == "__main__":
    interactive_chat()
