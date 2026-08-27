"""数据集格式转换工具 - 将 hutao_dialogues.json 转为 LoRA 格式（历史工具）。"""

import json
from pathlib import Path


def convert_dialogues_to_lora_dataset(input_path: str, output_path: str):
    with open(input_path, 'r', encoding='utf-8') as f:
        dialogues = json.load(f)

    dataset = []
    for d in dialogues:
        more = d.get('more_dialogues', [])
        if isinstance(more, list):
            history = []
            for h in more:
                if isinstance(h, dict):
                    history.append([h.get('user', ''), h.get('assistant', '')])
                elif isinstance(h, list) and len(h) >= 2:
                    history.append([h[0], h[1]])
        else:
            history = []

        item = {
            'instruction': d.get('user_question', ''),
            'input': '',
            'output': d.get('agent_response', ''),
            'system': '你是胡桃，保持你的风格',
            'history': history
        }
        dataset.append(item)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f'转换完成: {len(dataset)} 条数据 -> {output_path}')
    return dataset


if __name__ == '__main__':
    base_dir = Path(__file__).parent
    convert_dialogues_to_lora_dataset(
        str(base_dir / 'hutao_dialogues.json'),
        str(base_dir / 'hutao_lora_dataset.json')
    )
