"""Build a complete local human review packet with full context and response texts."""

import argparse
import html
import json
from pathlib import Path

OVERRIDES = {
    "77a7de5cdd5c9b2f": (
        "建议保留；补充身份依据",
        "模型称妃为琉璃的姐姐，与已核验的妹妹身份不符。建议在训练输入补充明确的兄妹关系后使用，避免信息条件不足。",
    ),
    "dff7a40935e995b7": (
        "建议保留",
        "模型把可能知道升级为明确知道，还添加从小不擅长撒谎的输入外经历；原文保持不确定性。",
    ),
    "3c6d32f6b9b563d6": (
        "建议保留",
        "妃此前已拒绝汀追问并遮脸敷衍，原文继续掩饰，模型突然承认单相思，改变了当前人物行为。",
    ),
    "00e6d63069e349e0": (
        "建议保留",
        "夜子已经在场，争议是琉璃是否参加。模型转成夜子自己想参加，误解当前谈论对象。",
    ),
    "6b8200d369b31df1": (
        "倾向保留；需判断修辞",
        "原文强调面对自己，模型用永远失去琉璃施压。但这可能是夸张劝说，不应直接标成事实错误。请判断人物是否会这样劝告。",
    ),
    "acd5d69883675bea": (
        "暂缓；知识边界需复核",
        "模型用了该不会，可能是在猜测而非声称知道秘密。添加旁白违反本次输出要求，但主要是格式问题，不能单独证明人物失真。",
    ),
    "3a4deefb7f8874db": (
        "建议重采样",
        "短反问与长篇旁白解释差异过大，容易学成长度或格式偏好。动作变化也不能直接认定为矛盾，应换一个长度相近的候选再比较。",
    ),
    "3dfe0f04f9f8c8d1": (
        "倾向保留；需判断修辞",
        "已知规则是不能违抗夜子；模型扩大为没有追求幸福的权利。但可能是悲观修辞，需结合人物表达判断是否真在新增规则。",
    ),
    "b6ba117c6da2a30f": (
        "暂缓；关系态度需复核",
        "原文强调独立身份，模型表现对关系变化的反应。后者不能仅凭此片段判错，需结合妃对汀的态度判断。",
    ),
    "b84c79a9e9631df1": (
        "建议排除",
        "那就好可以承接琉璃否认想忘记妃。不能因为未推进原作剧情就认定为负样本；不接受初筛的无承接硬错误判断。",
    ),
    "7e388fc7fb81906b": (
        "先校勘原文",
        "模型编造具体童年经历，但原文能我提供疑似缺字。先核对底本并记录修订，再考虑入选。",
    ),
}


def assess(r):
    for suffix, value in OVERRIDES.items():
        if r["id"].endswith(suffix):
            return *value, "助手二次复核，按本次讨论修正；仍待你审核"
    j = r.get("judgment")
    if not j:
        return (
            "待判断；初评失败",
            "裁判响应解析失败，不能据此判为负样本。请直接比较下面的完整材料。",
            "无有效初评；助手未逐条二次复核",
        )
    if not j["context_sufficient"] or j["winner"] == "invalid":
        title = "暂缓；先检查上下文"
    elif j["winner"] == "tie":
        title = "倾向不构成偏好对；初评认为相当"
    elif j["winner"] == r["original_side"]:
        title = "候选保留；需核实原文优势"
    else:
        title = "倾向不采用原文为优选；初评偏向模型"
    return (
        title,
        "以下为 DeepSeek 初评理由，尚未获得助手逐条二次确认：" + j["reason"],
        "DeepSeek 初评；助手仅按结果分流，不冒充独立复核",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--independent-reviews", type=Path)
    a = p.parse_args()
    records = {
        r["id"]: r
        for r in (
            json.loads(s)
            for s in (a.root / "results.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    rows = sorted(
        records.values(),
        key=lambda r: (r["source"]["source_file"], r["context_end"], r["id"]),
    )
    if len(rows) != 105 or any(not r.get("generated") for r in rows):
        raise ValueError("expected all 105 completed generations")
    notes = {}
    if a.independent_reviews:
        for f in sorted(a.independent_reviews.glob("independent_review_*.json")):
            for note in json.loads(f.read_text(encoding="utf-8")):
                if note["n"] in notes:
                    raise ValueError("duplicate review number")
                notes[note["n"]] = note
        if set(notes) != set(range(1, 106)):
            raise ValueError("expected 105 independent reviews")
    a.output.mkdir(parents=True, exist_ok=bool(notes))
    intro = [
        "# 全部 105 条游戏原文与模型回复审核",
        "",
        "每条包含完整模型输入、原文回复、DeepSeek 回复、判断来源和理由。未截断前情或回复。",
        "",
        "105 条均已生成；104 条有有效 DeepSeek 初评，1 条初评失败。助手逐条二次复核过其中 11 条，其余保留初评并明确标注来源。初评由生成模型自评，不能当作独立人工判断。",
        "",
        "这里的建议不等于批准。既有“保留 9 对”已调整：部分转为暂缓或重采样，以本审核包的最新意见为准。不会自动改动训练数据。",
        "",
        "下面的编号是按来源文件及原文位置重新排列的全量编号，不同于聊天中那 11 条的临时编号。每条同时保留稳定 ID。",
        "",
    ]
    cards, blocks, meta = [], [], []
    if notes:
        intro[4] = "全部105条已由助手结合原文前情逐条独立复核。DeepSeek初评仅保留为历史记录，不作为本次判断依据。这是AI审核，不是人工批准。"
        intro[6] = "每条分别说明当前情景、人物回答倾向、原文分析、候选分析和处理建议。保留合理变体；不以复现原句、长度、温柔程度或道德评价代替人物一致性。不会自动改动训练数据。"
    for i, r in enumerate(rows, 1):
        title, reason, origin = assess(r)
        note = notes.get(i)
        sections = []
        if note:
            for field in ("decision", "scene", "tendency", "original", "candidate", "action"):
                if not note.get(field):
                    raise ValueError(f"missing review field: {i} {field}")
            if not note.get("lines") or any(not r["context_start"] <= line <= r["context_end"] for line in note["lines"]):
                raise ValueError(f"review evidence outside actual input: {i}")
            title = note["decision"]
            origin = "助手逐条独立审核；待用户审核，未批准入训"
            sections = [("当前情景",note["scene"]),("人物回答倾向",note["tendency"]),("原文判断",note["original"]),("模型判断",note["candidate"]),("处理建议",note["action"]),("前情证据", ", ".join(f"L{x}" for x in note["lines"]))]
            reason = "\n\n".join(f"**{k}：** {v}" for k,v in sections)
        j = r.get("judgment", {})
        source = r["source"]
        winner = j.get("winner")
        pref = (
            ("原文" if winner == r.get("original_side") else "DeepSeek")
            if winner in {"A", "B"}
            else {"tie": "相当", "invalid": "无效"}.get(winner, "初评失败")
        )
        source_label = f"{source['source_file']}｜输入 L{r['context_start']}–L{r['context_end']}｜目标事件 {', '.join(source['event_ids'])}"
        details = json.dumps(
            {
                "初评偏向": pref,
                "自报置信度_未经校准": j.get("confidence"),
                "上下文充分": j.get("context_sufficient"),
                "证据行号": j.get("evidence_lines"),
                "维度比较": {
                    k: (
                        "原文"
                        if v == r.get("original_side")
                        else "DeepSeek"
                        if v in {"A", "B"}
                        else v
                    )
                    for k, v in j.get("dimensions", {}).items()
                },
                "原文问题": j.get("hard_errors", {}).get(r.get("original_side")),
                "模型问题": j.get("hard_errors", {}).get(
                    "B" if r.get("original_side") == "A" else "A"
                ),
                "初评原始理由": j.get("reason"),
            },
            ensure_ascii=False,
            indent=2,
        )
        block = [
            f"## {i:03d}｜{r['id']}",
            "",
            source_label,
            "",
            f"**当前建议：{title}**",
            "",
            f"判断来源：{origin}",
            "",
            reason,
            "",
            "### 游戏上下文（模型实际看到的完整前文）",
            "",
            r["prompt"][-1]["content"],
            "",
            "### 原文回复",
            "",
            r["original"],
            "",
            "### DeepSeek 回复",
            "",
            r["generated"],
            "",
            "### 初评详情",
            "",
            "```json",
            details,
            "```",
            "",
            "**你的决定：** 待定 / 保留 / 排除 / 重采样 / 补上下文 / 校勘",
            "",
            "**你的理由：**",
            "",
            "---",
            "",
        ]
        blocks.append("\n".join(block))
        esc = html.escape
        reason_html = "".join(f"<p><strong>{esc(k)}：</strong>{esc(v)}</p>" for k,v in sections) if sections else f"<p>{esc(reason)}</p>"
        options = "".join(
            f"<option>{v}</option>"
            for v in ("待定", "保留", "排除", "重采样", "补上下文", "校勘")
        )
        cards.append(f'''<article id="row-{i}" data-id="{esc(r["id"])}"><h2>{i:03d} · {esc(source["source_file"])}</h2>
<p class="muted">{esc(source_label)}<br>{esc(r["id"])}</p><div class="advice"><strong>{esc(title)}</strong><p>{esc(origin)}</p>{reason_html}</div>
<details open><summary>游戏上下文 · 完整输入前文</summary><pre>{esc(r["prompt"][-1]["content"])}</pre></details>
<div class="comparison"><section><h3>游戏原文回复</h3><pre>{esc(r["original"])}</pre></section><section><h3>DeepSeek 回复</h3><pre>{esc(r["generated"])}</pre></section></div>
<details><summary>初评详情 · {esc(pref)} · 自报置信度 {j.get("confidence", "无")}</summary><pre>{esc(details)}</pre></details>
<label>你的决定 <select>{options}</select></label><label>你的理由<textarea rows="3" placeholder="填写判断依据、需要补充的前情或原文问题"></textarea></label></article>''')
        meta.append(
            {
                "number": i,
                "id": r["id"],
                "audit_record_sha256": r["record_sha256"],
                "suggestion": title,
                "judgment_source": origin,
            }
        )
    system = "\n\n".join(dict.fromkeys(r["prompt"][0]["content"] for r in rows))
    intro += ["## 所有样本共用的人物指令", "", system, ""]
    (a.output / "ALL_105.md").write_text(
        "\n".join(intro) + "\n" + "\n".join(blocks), encoding="utf-8"
    )
    for start in range(0, len(rows), 15):
        (
            a.output / f"审核_{start + 1:03d}-{min(start + 15, len(rows)):03d}.md"
        ).write_text(
            "\n".join(intro) + "\n" + "\n".join(blocks[start : start + 15]),
            encoding="utf-8",
        )
    data = json.dumps(meta, ensure_ascii=False).replace("<", "\\u003c")
    page = (
        """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>105 条原文偏好审核</title>
<style>body{font:16px/1.75 system-ui,sans-serif;margin:0;background:#f4f6f8;color:#1c2936}header,main{max-width:1180px;margin:auto;padding:24px}nav{position:sticky;top:0;background:#fff;padding:12px;z-index:2;border-bottom:1px solid #ccc;display:flex;gap:12px;flex-wrap:wrap}article{background:white;padding:24px;margin-bottom:24px;border-radius:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}h2{font-size:21px}.muted{color:#586571;font-size:13px}.advice{background:#edf3ff;padding:16px;border-radius:8px}.comparison{display:grid;grid-template-columns:1fr 1fr;gap:20px}.comparison section{padding:16px;background:#f6f8fa}summary{cursor:pointer;font-weight:600;margin-top:16px}label{display:block;margin-top:16px}textarea{width:98%;font:inherit}select,button,input{font:inherit;padding:5px 10px}button{cursor:pointer}@media(max-width:700px){.comparison{grid-template-columns:1fr}article{padding:14px}}</style>
<header><h1>全部 105 条：游戏原文 × DeepSeek 回复</h1><p>逐条完整前文、两种回复、判断依据及你的审核栏。105 条生成、104 条有效初评、1 条初评失败。</p><p>判断来源明确区分：11 条有助手二次复核，其余为 DeepSeek 初评；模型自评不代表独立人工结论。最新建议已修正早期“保留 9 对”的较宽判断。</p><p>此处编号按原文位置重新排列。记录自动保存在当前浏览器（若支持），请及时导出备份。审核操作只保存草稿，不会改动训练集。</p><details><summary>查看所有样本共用的人物指令</summary><pre>"""
        + html.escape(system)
        + """</pre></details></header>
<nav><button id="prev">上一批</button><span id="page"></span><button id="next">下一批</button><label style="margin:0">跳到编号 <input id="jump" type="number" min="1" max="105" value="1" style="width:75px"></label><button id="go">跳转</button><button id="save">导出我的审核草稿</button><span id="status"></span></nav><main>"""
        + "".join(cards)
        + """</main><script>
const metadata=DATA, cards=[...document.querySelectorAll('article')], key='source-dpo-105-human-review-v1';let state={},page=0;try{state=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
function persist(){try{localStorage.setItem(key,JSON.stringify(state));document.querySelector('#status').textContent='已本地保存'}catch(e){document.querySelector('#status').textContent='本地保存不可用，请导出备份'}}
cards.forEach(card=>{let saved=state[card.dataset.id]||{};card.querySelector('select').value=saved.decision||'待定';card.querySelector('textarea').value=saved.reason||'';card.addEventListener('input',()=>{state[card.dataset.id]={decision:card.querySelector('select').value,reason:card.querySelector('textarea').value};persist()})});
function show(){cards.forEach((c,i)=>c.hidden=Math.floor(i/15)!==page);document.querySelector('#page').textContent=`第 ${page+1}/7 批 · ${page*15+1}–${Math.min(page*15+15,105)}`;document.querySelector('#prev').disabled=page===0;document.querySelector('#next').disabled=page===6}
document.querySelector('#prev').onclick=()=>{page=Math.max(0,page-1);show();window.scrollTo(0,0)};document.querySelector('#next').onclick=()=>{page=Math.min(6,page+1);show();window.scrollTo(0,0)};
document.querySelector('#go').onclick=()=>{let n=Number(document.querySelector('#jump').value);if(!Number.isInteger(n)||n<1||n>105)return;page=Math.floor((n-1)/15);show();cards[n-1].scrollIntoView()};
document.querySelector('#save').onclick=()=>{let output={schema:'source-dpo-human-review-draft-v1',status:'draft',decisions:metadata.map(m=>({...m,...(state[m.id]||{decision:'待定',reason:''})}))};let url=URL.createObjectURL(new Blob([JSON.stringify(output,null,2)],{type:'application/json'}));let a=document.createElement('a');a.href=url;a.download='我的105条审核草稿.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};show();
</script></html>"""
    )
    if notes:
        page = page.replace("判断来源明确区分：11 条有助手二次复核，其余为 DeepSeek 初评；模型自评不代表独立人工结论。最新建议已修正早期“保留 9 对”的较宽判断。", "全部105条均有助手独立审核：当前情景 → 人物倾向 → 两种回复判断 → 处理建议。DeepSeek初评仅作历史参考。AI建议不代表你的批准。")
        bound = [{"number":i,"id":r["id"],"audit_record_sha256":r["record_sha256"],"reviewer":"assistant_independent","human_approved":False,"review":notes[i],"prompt":r["prompt"],"original":r["original"],"generated":r["generated"],"source":r["source"]} for i,r in enumerate(rows,1)]
        (a.output / "independent_review_all_105.json").write_text(json.dumps(bound,ensure_ascii=False,indent=2),encoding="utf-8")
    (a.output / "审核全部105条.html").write_text(
        page.replace("const metadata=DATA,", "const metadata=" + data + ","),
        encoding="utf-8",
    )
    (a.output / "index.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "records": len(rows),
                "batches": 7,
                "assistant_second_reviews": len(notes) if notes else len(OVERRIDES),
                "output": str(a.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
