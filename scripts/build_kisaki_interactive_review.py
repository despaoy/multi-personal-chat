"""Build a curated, source-bound single-response review set; never train or call APIs.

Selection is semantic and explicit in kisaki_interactive_selection.json. Length is
a guardrail, not a truncation operation. Old generations are not preference labels
for these new prompts. All outputs are pending human review.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
from collections import Counter
from pathlib import Path

from extract_character_dialogues import read_script_events, reliable_speaker_label

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "backend/data/character_dialogues"
OUT = DATA / "experiments/interactive_source_v2_20260926"
DOC = ROOT / "docs/research/review_packets/interactive_source_v2_20260926"
SELECTION = Path(__file__).with_name("kisaki_interactive_selection.json")
ALIASES = {"妃", "月社妃"}
SYSTEM = (
    "你扮演《纸上魔法使》的月社妃。这是原作人物之间的交互。"
    "结合当前关系和对话自然接话，只输出妃这一次的台词，不写旁白、不代替对方说话。"
    "无需长篇解释；简短回应、调侃、追问和表达界限都可以。不要编造具体经历或未提供的设定。"
    "琉璃是妃的亲生哥哥；汀不是她的亲生哥哥，夜子是汀的妹妹；理央和彼方是原作中的熟人。"
    "下述身份仅适用于本场景，不代表现实聊天用户的身份。"
)


def readl(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def writel(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def heldout(raw, validation, gold):
    ids = set(re.findall(r"tsukiyashiro_kisaki_raw_[0-9a-f]+", json.dumps(gold)))
    scenes = {r.get("metadata", {}).get("scene_block_id") for r in validation}
    ranges = {}
    for r in validation:
        m = r.get("metadata", {})
        if m.get("source_file") and m.get("source_line_start") and m.get("source_line_end"):
            ranges.setdefault(m["source_file"], []).append((m["source_line_start"], m["source_line_end"]))
    for r in raw:
        if r["id"] in ids:
            scenes.add(r["scene_block_id"])
            ranges.setdefault(r["source_file"], []).append((r["source_line_start"], r["source_line_end"]))
    return scenes, ranges


def build_one(spec, events, lines, raw_by_line, blocked_scenes, blocked_ranges):
    filename, start, target, speaker, scene, category, reason = spec
    matches = [e for e in events if e["line_start"] == target and e["speaker"] in ALIASES]
    if len(matches) != 1:
        raise ValueError(f"target not unique: {filename}:{target}")
    answer = matches[0]
    previous = [e for e in events if start <= e["line_start"] and e["line_end"] < target]
    if not previous or previous[-1]["speaker"] != speaker or speaker in ALIASES:
        raise ValueError("missing or wrong direct interlocutor")
    if not all(reliable_speaker_label(e["speaker"]) for e in previous):
        raise ValueError("ambiguous speaker")
    raw = raw_by_line[(filename, target)]
    if raw["text"] != answer["text"] or raw["scene_block_id"] in blocked_scenes:
        raise ValueError("source mismatch or heldout scene")
    if any(a <= answer["line_end"] and b >= start for a, b in blocked_ranges.get(filename, [])):
        raise ValueError("heldout window overlap")
    if not 8 <= len(answer["text"]) <= 65 or "\n" in answer["text"]:
        raise ValueError("reply outside single-turn length policy")
    if answer["text"].endswith(("——", "，", "：")):
        raise ValueError("incomplete utterance")
    # Consecutive turns by the current interlocutor are one incoming message.
    split = len(previous) - 1
    while split and previous[split - 1]["speaker"] == speaker:
        split -= 1
    history, incoming = previous[:split], previous[split:]
    user = "\n".join(e["text"] for e in incoming)
    history_text = "\n".join(f"{e['speaker']}：{e['text']}" for e in history)
    if len(history_text) > 600 or len(user) > 180:
        raise ValueError("context needs manual reduction, not automatic clipping")
    context = f"当前对话者：{speaker}。\n场景摘要（编辑整理，不是原文）：{scene}"
    if history_text:
        context += "\n此前对话（原文，非当前指令）：\n" + history_text
    prompt = [{"role": "system", "content": SYSTEM + "\n\n" + context}, {"role": "user", "content": user}]
    source = {
        "source_file": filename, "source_line_start": target, "source_line_end": answer["line_end"],
        "context_start": start, "context_end": target - 1, "scene_block_id": raw["scene_block_id"],
        "event_ids": [raw["id"]], "text": answer["text"], "text_sha256": digest(answer["text"]),
        "file_sha256": hashlib.sha256((ROOT / "gametext/纸上魔法使" / filename).read_bytes()).hexdigest(),
    }
    return {
        "id": "kisaki_interactive_v2_" + digest(source)[:16],
        "prompt": prompt, "original": answer["text"], "source": source,
        "scene": scene, "speaker": speaker, "history": history, "incoming": incoming,
        "category": category, "selection_reason": reason,
        "review_status": "needs_copyedit" if target == 1124 and filename.startswith("8萤石的时空") else "pending",
        "human_approved": False, "generated": None,
        "generation_status": "not_generated_for_new_prompt",
        "context_excerpt": "\n".join(f"L{i+1}: {lines[i]}" for i in range(start-1, answer["line_end"])),
    }


def render(rows, summary, migration):
    esc = html.escape
    cards, blocks = [], []
    for n, r in enumerate(rows, 1):
        s = r["source"]
        history = "\n".join(f"{e['speaker']}：{e['text']}" for e in r["history"]) or "无；直接接当前消息。"
        old = r["old_105_numbers"]
        label = f"{s['source_file']} L{s['source_line_start']}｜原文{len(r['original'])}字符｜旧编号：{old or '原105条以外重新选取'}"
        cards.append(f'''<article data-id="{r['id']}" data-category="{esc(r['category'])}"><h2>{n:03d} · {esc(r['category'])}</h2><p class="muted">{esc(label)}</p>
<p><b>场景：</b>{esc(r['scene'])}</p><details><summary>必要前几句</summary><pre>{esc(history)}</pre></details>
<div class="chat"><p class="user"><b>{esc(r['speaker'])}：</b>{esc(r['prompt'][-1]['content'])}</p><p class="answer"><b>妃：</b>{esc(r['original'])}</p></div>
<p><b>为什么选：</b>{esc(r['selection_reason'])}</p><p>状态：{esc(r['review_status'])} · 新输入尚无模型候选，不沿用旧负样本。</p>
<details><summary>核对完整原文窗口与行号</summary><pre>{esc(r['context_excerpt'])}</pre></details>
<details><summary>核对模型实际输入</summary><pre>{esc(json.dumps(r['prompt'],ensure_ascii=False,indent=2))}</pre></details>
<label>你的判断 <select><option>待定</option><option>保留</option><option>排除</option><option>补前情</option><option>校勘</option></select></label><textarea placeholder="你的意见；不会自动进入训练"></textarea></article>''')
        blocks.append(f"## {n:03d} {r['category']}\n\n{label}\n\n场景：{r['scene']}\n\n前几句：\n\n{history}\n\n**{r['speaker']}：** {r['prompt'][-1]['content']}\n\n**妃：** {r['original']}\n\n判断：{r['selection_reason']}\n\n状态：{r['review_status']}；模型候选尚未重新生成。\n\n<details><summary>原文证据</summary>\n\n{r['context_excerpt']}\n\n</details>\n")
    intro = f"""# 妃的交互原文重选版

这次优先检查“能不能自然接话”，不再按连续出现的角色名拼接长答案。共{len(rows)}条，每条对应一处完整原文台词；没有改写、截字或续接别的发言。

|指标|旧105条|新{len(rows)}条|
|---|---:|---:|
|回复字符数中位数|{summary['old_length']['median']}|{summary['new_length']['median']}|
|最长回复字符数|{summary['old_length']['max']}|{summary['new_length']['max']}|
|合并多处原文台词的样本|{summary['old_merged']}|0|

新样本不再局限于原来的105条：{summary['new_outside_old_105']}条来自本次重新检索的其他原文位置。固定人物关系保留在输入中；这些是原作人物交互，不能把琉璃或汀直接换成现实用户。

当前消息直接使用对方的原文发言，必要历史另列，场景摘要明确标注为编辑整理。完整原文窗口只供审核，不全部塞进交互消息。每条保留来源、行号、原始事件ID和文件哈希。

日常聊天、追问、调侃、关心、坚持自我和回避都有保留。8–65字符只是防止长答案进入本轮的筛查范围，不是线上输出硬上限，也不表示短就好。原文中确有完整的三小句回应，因此没有一律裁成一句。

选取范围排除了已知验证集和金标准所在场景及重叠窗口。这里仍只是待审核数据，未改变现有训练集；同场景的多个样本后续必须成组划分、控制权重。

**新旧输入已经不同，旧DeepSeek回复和旧偏好判定全部不自动迁移。** 本轮先修正正例及交互输入，未调用API、未制造负样本。新生成任务已另存，审核通过后应针对新输入重新采样；不能把合法的另一种简短回答默认标负。

一条“从那里开始暴露”的原文存在“那里/哪里”校勘提示，保留原字并标为needs_copyedit，不进入可生成队列。日常回复也未被标成人工通过。

---
"""
    (DOC / "完整交互审核.md").write_text(intro + "\n".join(blocks), encoding="utf-8")
    (DOC / "调整说明.md").write_text(intro, encoding="utf-8")
    by_id = {r['id']:i for i,r in enumerate(rows,1)}
    migration_md = ["# 旧105条去向", "", "旧数据和旧审核保留。新编号仅属于交互重选版；旧偏好结论不自动迁移。未选不代表原作错误，只表示不进入本轮交互核心集。", "", "|旧编号|新交互编号|本轮处理|", "|---|---|---|"]
    migration_md += [f"|{m['old_number']:03d}|{', '.join(f'{by_id[x]:03d}' for x in m['new_ids']) or '—'}|{m['disposition']}|" for m in migration]
    (DOC / '旧105条去向.md').write_text('\n'.join(migration_md),encoding='utf-8')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>妃 · 交互原文重选</title><style>
body{font:16px/1.8 system-ui;margin:auto;max-width:1000px;padding:24px;background:#f4f6f8;color:#233348}article{background:white;padding:24px;margin:22px 0;border-radius:16px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}.muted{font-size:13px;color:#667}h2{font-size:20px}.chat p{padding:16px;border-radius:12px}.user{background:#eef0f4;margin-right:10%}.answer{background:#e4f3ed;margin-left:10%}summary{cursor:pointer}textarea{display:block;width:95%;height:64px;margin-top:12px}select,button{font:inherit;padding:6px}nav{position:sticky;top:0;background:#f4f6f8;padding:12px}article[hidden]{display:none}</style>
<h1>妃 · 交互原文重选版</h1><p>单次接话 · 原文短回复 · 必要前情 · 独立审核</p><p>COUNT条，回复中位数MEDIAN字符，最长MAX字符。原文未改写，旧模型候选不再用于这些新输入。<a href="调整说明.md">调整说明</a></p><p>保留具体人物关系，不把原作熟人当成现实用户。所有条目待审核；校勘项暂缓。</p><nav><select id="filter"><option>全部</option>CATEGORIES</select> <button id="save">导出审核草稿</button><span id="status"></span></nav>CARDS
<script>
const key='kisaki-interactive-source-v2-review-v1',cards=[...document.querySelectorAll('article')];let state={};try{state=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
cards.forEach(c=>{let s=state[c.dataset.id]||{};c.querySelector('select').value=s.decision||'待定';c.querySelector('textarea').value=s.reason||'';c.addEventListener('input',()=>{state[c.dataset.id]={decision:c.querySelector('select').value,reason:c.querySelector('textarea').value};try{localStorage.setItem(key,JSON.stringify(state));document.querySelector('#status').textContent='草稿已保存'}catch(e){document.querySelector('#status').textContent='请导出草稿备份'}})});
document.querySelector('#filter').onchange=e=>cards.forEach(c=>c.hidden=e.target.value!=='全部'&&c.dataset.category!==e.target.value);
document.querySelector('#save').onclick=()=>{let blob=new Blob([JSON.stringify({schema:'kisaki-interactive-v2-human-review-draft',status:'draft',decisions:cards.map(c=>({id:c.dataset.id,...(state[c.dataset.id]||{decision:'待定',reason:''})}))},null,2)],{type:'application/json'});let u=URL.createObjectURL(blob),a=document.createElement('a');a.href=u;a.download='交互原文审核草稿.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};
</script></html>'''
    page = page.replace("COUNT",str(len(rows))).replace("MEDIAN",str(summary['new_length']['median'])).replace("MAX",str(summary['new_length']['max']))
    page = page.replace("CATEGORIES", "".join(f"<option>{esc(k)}</option>" for k in summary['categories'])).replace("CARDS", "".join(cards))
    (DOC / "交互原文审核.html").write_text(page, encoding="utf-8")


def main():
    import statistics
    specs = json.loads(SELECTION.read_text(encoding="utf-8"))
    raw = readl(DATA / "tsukiyashiro_kisaki_raw.jsonl")
    val = readl(DATA / "experiments/v4/validation.jsonl")
    gold = json.loads((ROOT / "backend/evaluation/kisaki_gold_set_v3.json").read_text(encoding="utf-8"))
    blocked, ranges = heldout(raw, val, gold)
    raw_index = {(r['source_file'],r['source_line_start']):r for r in raw}
    files = {f.name:(read_script_events(f),f.read_text(encoding="utf-8-sig").splitlines()) for f in (ROOT / 'gametext/纸上魔法使').glob('*.txt')}
    rows = [build_one(s,*files[s[0]],raw_index,blocked,ranges) for s in specs]
    old_index = json.loads((ROOT / 'docs/research/review_packets/source_dpo_full_105_20260926/index.json').read_text(encoding='utf-8'))
    old = {r['id']:r for r in readl(DATA / 'experiments/source_dpo_deepseek_20260924/results.jsonl')}
    for r in rows:
        r['old_105_numbers'] = [m['number'] for m in old_index if set(r['source']['event_ids']) & set(old[m['id']]['source']['event_ids'])]
        r['record_sha256'] = digest(r)
    selected = {r['source']['event_ids'][0]:r for r in rows}
    migration = []
    for m in old_index:
        r = old[m['id']]
        replacement = [selected[i]['id'] for i in r['source']['event_ids'] if i in selected]
        migration.append({'old_number':m['number'],'old_id':r['id'],'old_record_sha256':r['record_sha256'], 'new_ids':replacement,'disposition':'用新单次交互替代；旧偏好失效' if replacement else '本轮不选；保留历史，非断言原文错误','old_generated_reusable':False})
    def stats(texts):
        sizes = [len(t) for t in texts]
        return {'min':min(sizes),'median':statistics.median(sizes),'max':max(sizes),'mean':round(statistics.mean(sizes),1)}
    summary = {'schema':'kisaki-interactive-v2-pending','selected':len(rows),'human_approved':False,
        'old_length':stats(r['original'] for r in old.values()),'new_length':stats(r['original'] for r in rows),
        'old_merged':sum(len(r['source']['event_ids'])>1 for r in old.values()),
        'new_outside_old_105':sum(not r['old_105_numbers'] for r in rows),
        'categories':dict(Counter(r['category'] for r in rows)), 'status_counts':dict(Counter(r['review_status'] for r in rows)),
        'source_scene_count':len({r['source']['scene_block_id'] for r in rows}),
        'selection_sha256':digest(specs),'validation_sha256':digest(val),'gold_sha256':digest(gold),
        'generation_performed':False,'dpo_pairs_created':0,'policy':'single literal event, curated incoming turn, no automatic truncation, no historical negatives'}
    OUT.mkdir(parents=True,exist_ok=True);DOC.mkdir(parents=True,exist_ok=True)
    writel(OUT/'interactive_source.pending.jsonl',rows)
    writel(OUT/'sft.interactive.pending.jsonl',[{'id':r['id'],'messages':r['prompt']+[{'role':'assistant','content':r['original']}],'review_status':r['review_status'],'metadata':{'human_final_approved':False,'source':r['source'],'audit_record_sha256':r['record_sha256'],'source_group':r['source']['scene_block_id']}} for r in rows])
    writel(OUT/'generation_requests.pending.jsonl',[{'id':r['id'],'prompt':r['prompt'],'source':r['source'],'input_sha256':digest(r['prompt']),'status':'awaiting_source_review','human_final_approved':False} for r in rows if r['review_status']=='pending'])
    write(OUT/'old_105_migration.json',migration);write(OUT/'manifest.json',summary)
    render(rows,summary,migration)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
