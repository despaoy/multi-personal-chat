"""生成交互式盲评 HTML 界面，支持 4 组对比的人工评估。"""
from __future__ import annotations

import json
from pathlib import Path

BLIND_DIR = Path(r"c:\Users\13474\Desktop\qqchat-enhanced\runtime_analysis\blind_reviews")
OUTPUT_DIR = Path(r"c:\Users\13474\Desktop\qqchat-enhanced\runtime_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COMPARISONS = [
    ("e1-vs-e2-seed42", "E1(LoRA) vs E2(NEFTune)"),
    ("e1-vs-e3-seed42", "E1(LoRA) vs E3(DoRA)"),
    ("e1-vs-e4-seed42", "E1(LoRA) vs E4(RSLoRA)"),
    ("e1-vs-e5-seed42", "E1(LoRA) vs E5(SeqPacking)"),
]


def load_blind(comp_dir: Path) -> dict:
    review_path = comp_dir / "blind_review.json"
    return json.loads(review_path.read_text(encoding="utf-8"))


def generate_html() -> None:
    """生成交互式盲评界面。"""

    # 预加载所有数据
    all_data = {}
    for comp_id, comp_label in COMPARISONS:
        comp_dir = BLIND_DIR / comp_id
        if comp_dir.exists():
            all_data[comp_id] = {
                "label": comp_label,
                "data": load_blind(comp_dir),
            }

    if not all_data:
        print("无盲评数据")
        return

    # 生成单个对比组的 HTML 内容
    def render_comparison(comp_id: str, info: dict) -> str:
        data = info["data"]
        samples = data.get("samples", [])
        label = info["label"]

        sample_cards = []
        for i, s in enumerate(samples):
            sid = s.get("id", f"sample_{i}")
            category = s.get("category", "unknown")
            prompt = s.get("prompt", "")
            resp_a = s.get("response_A", "")
            resp_b = s.get("response_B", "")

            card = f'''
    <div class="sample-card" data-comp="{comp_id}" data-idx="{i}" data-category="{category}">
      <div class="sample-header">
        <span class="sample-id">{sid}</span>
        <span class="sample-cat">{category}</span>
      </div>
      <div class="prompt-box">
        <div class="prompt-label">Prompt:</div>
        <div class="prompt-text">{prompt}</div>
      </div>
      <div class="response-grid">
        <div class="response-box" id="A-{comp_id}-{i}">
          <div class="resp-label">A:</div>
          <div class="resp-text">{resp_a}</div>
        </div>
        <div class="response-box" id="B-{comp_id}-{i}">
          <div class="resp-label">B:</div>
          <div class="resp-text">{resp_b}</div>
        </div>
      </div>
      <div class="vote-row">
        <button class="vote-btn vote-a" onclick="vote('{comp_id}', {i}, 'A')">选 A</button>
        <button class="vote-btn vote-b" onclick="vote('{comp_id}', {i}, 'B')">选 B</button>
        <button class="vote-btn vote-tie" onclick="vote('{comp_id}', {i}, 'tie')">平局</button>
        <button class="vote-btn vote-skip" onclick="vote('{comp_id}', {i}, 'skip')">跳过</button>
        <span class="vote-result" id="result-{comp_id}-{i}"></span>
      </div>
    </div>'''
            sample_cards.append(card)

        return f'''
<div class="comparison-section" id="section-{comp_id}">
  <h2>{label}</h2>
  <div class="comp-stats" id="stats-{comp_id}">
    <span class="stat">已评: <b id="voted-{comp_id}">0</b>/{len(samples)}</span>
    <span class="stat">A 胜: <b id="winA-{comp_id}">0</b></span>
    <span class="stat">B 胜: <b id="winB-{comp_id}">0</b></span>
    <span class="stat">平局: <b id="tie-{comp_id}">0</b></span>
  </div>
  <div class="filter-row">
    <button class="filter-btn active" onclick="filterCategory('{comp_id}', 'all')">全部</button>
    <button class="filter-btn" onclick="filterCategory('{comp_id}', 'factual')">事实</button>
    <button class="filter-btn" onclick="filterCategory('{comp_id}', 'multiturn')">多轮</button>
    <button class="filter-btn" onclick="filterCategory('{comp_id}', 'persona')">人设</button>
    <button class="filter-btn" onclick="filterCategory('{comp_id}', 'safety')">安全</button>
  </div>
  {''.join(sample_cards)}
</div>'''

    comparisons_html = "\n".join(
        render_comparison(cid, info) for cid, info in all_data.items()
    )

    # 将数据嵌入 JavaScript
    embedded_data = {}
    for cid, info in all_data.items():
        samples = info["data"].get("samples", [])
        embedded_data[cid] = {
            "label": info["label"],
            "samples": [
                {
                    "id": s.get("id", ""),
                    "category": s.get("category", ""),
                    "prompt": s.get("prompt", ""),
                    "response_A": s.get("response_A", ""),
                    "response_B": s.get("response_B", ""),
                }
                for s in samples
            ],
        }

    # 预先构造导航按钮
    nav_tabs_html = ''.join(
        f'<button class="nav-tab {"active" if i==0 else ""}" onclick="showTab(\'{cid}\')">{info["label"]}</button>'
        for i, (cid, info) in enumerate(all_data.items())
    )
    embedded_json = json.dumps(embedded_data, ensure_ascii=False)

    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>R1 盲评界面 - 人工评估</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, "Microsoft YaHei", "PingFang SC", sans-serif; padding: 16px; background: #f7fafc; color: #2d3748; line-height: 1.6; }
  .header { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 16px; }
  .header h1 { font-size: 22px; color: #1a202c; margin-bottom: 8px; }
  .header p { font-size: 13px; color: #718096; }
  .warning-box { background: #fffbeb; border-left: 4px solid #f6ad55; padding: 12px; margin: 12px 0; border-radius: 4px; font-size: 13px; }
  .nav-tabs { display: flex; gap: 4px; margin-bottom: 16px; flex-wrap: wrap; }
  .nav-tab { padding: 10px 16px; background: #edf2f7; border: none; border-radius: 6px 6px 0 0; cursor: pointer; font-size: 13px; color: #4a5568; transition: all 0.2s; }
  .nav-tab.active { background: #4299e1; color: white; font-weight: 600; }
  .nav-tab:hover { background: #bee3f8; }
  .nav-tab.active:hover { background: #3182ce; }
  .comparison-section { display: none; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
  .comparison-section.active { display: block; }
  .comparison-section h2 { font-size: 18px; color: #2b6cb0; margin-bottom: 12px; border-left: 3px solid #4299e1; padding-left: 10px; }
  .comp-stats { background: #ebf8ff; padding: 10px 14px; border-radius: 6px; margin-bottom: 12px; font-size: 13px; }
  .stat { margin-right: 20px; color: #4a5568; }
  .stat b { color: #2b6cb0; font-size: 15px; }
  .filter-row { margin-bottom: 16px; }
  .filter-btn { padding: 5px 12px; margin-right: 6px; background: #edf2f7; border: 1px solid #cbd5e0; border-radius: 4px; cursor: pointer; font-size: 12px; color: #4a5568; }
  .filter-btn.active { background: #4299e1; color: white; border-color: #3182ce; }
  .sample-card { border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px; margin-bottom: 16px; background: #fafafa; }
  .sample-card.hidden { display: none; }
  .sample-card.voted { opacity: 0.6; border-color: #9ae6b4; }
  .sample-header { display: flex; justify-content: space-between; margin-bottom: 10px; }
  .sample-id { font-family: monospace; font-size: 12px; color: #718096; }
  .sample-cat { background: #ebf8ff; color: #2b6cb0; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }
  .prompt-box { background: white; padding: 10px; border-radius: 6px; margin-bottom: 10px; border-left: 3px solid #cbd5e0; }
  .prompt-label { font-size: 11px; color: #a0aec0; margin-bottom: 4px; font-weight: 600; }
  .prompt-text { font-size: 14px; color: #2d3748; }
  .response-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 10px; }
  .response-box { background: white; padding: 12px; border-radius: 6px; border: 2px solid #e2e8f0; cursor: pointer; transition: border-color 0.2s; }
  .response-box:hover { border-color: #90cdf4; }
  .response-box.selected-a { border-color: #48bb78; background: #f0fff4; }
  .response-box.selected-b { border-color: #ed8936; background: #fffaf0; }
  .resp-label { font-size: 12px; font-weight: bold; color: #4a5568; margin-bottom: 4px; }
  .resp-text { font-size: 14px; color: #2d3748; white-space: pre-wrap; word-break: break-word; }
  .vote-row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  .vote-btn { padding: 6px 14px; border: none; border-radius: 4px; cursor: pointer; font-size: 12px; font-weight: 600; transition: all 0.2s; }
  .vote-a { background: #48bb78; color: white; }
  .vote-a:hover { background: #38a169; }
  .vote-b { background: #ed8936; color: white; }
  .vote-b:hover { background: #dd6b20; }
  .vote-tie { background: #ecc94b; color: white; }
  .vote-tie:hover { background: #d69e2e; }
  .vote-skip { background: #a0aec0; color: white; }
  .vote-skip:hover { background: #718096; }
  .vote-result { font-size: 13px; font-weight: 600; margin-left: 8px; }
  .vote-result.a { color: #48bb78; }
  .vote-result.b { color: #ed8936; }
  .vote-result.tie { color: #d69e2e; }
  .vote-result.skip { color: #718096; }
  .actions-bar { position: sticky; bottom: 0; background: white; padding: 12px 20px; box-shadow: 0 -2px 8px rgba(0,0,0,0.1); display: flex; justify-content: space-between; align-items: center; margin: 16px -20px -20px; border-radius: 0 0 8px 8px; }
  .btn-export { background: #4299e1; color: white; padding: 8px 20px; border: none; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 600; }
  .btn-export:hover { background: #3182ce; }
  .btn-reset { background: #fed7d7; color: #c53030; padding: 8px 16px; border: 1px solid #fc8181; border-radius: 6px; cursor: pointer; font-size: 12px; }
</style>
</head>
<body>

<div class="header">
  <h1>R1 PEFT 消融实验 - 盲评界面</h1>
  <p>4 组对比 × 40 条样本 = 160 条配对评估 | A/B 身份隐藏，评完可查看 blind_key.json 揭晓</p>
  <div class="warning-box">
    <b>评估规则：</b>仅根据 prompt 和 A/B 回复判断哪个更贴近月社妃角色（简洁、直接、第一人称、无 AI 自称）。<br>
    评估维度：角色一致性 | 事实准确性 | 安全性 | 简洁度 | 对话连贯性<br>
    <b>注意：</b>评估前不要查看 blind_key.json，否则会引入偏见。
  </div>
</div>

<div class="nav-tabs">
  __NAV_TABS__
</div>

__COMPARISONS__

<div class="actions-bar">
  <span style="font-size:13px;color:#718096;">进度自动保存到浏览器本地存储</span>
  <div>
    <button class="btn-reset" onclick="resetAll()">重置全部</button>
    <button class="btn-export" onclick="exportResults()">导出结果</button>
  </div>
</div>

<script>
// 嵌入数据
const blindData = __BLIND_DATA__;

// 加载已保存的投票
let votes = JSON.parse(localStorage.getItem('r1_blind_votes') || '{}');

// 显示投票状态
function refreshVoteDisplay(compId) {
  const samples = blindData[compId].samples;
  let voted = 0, winA = 0, winB = 0, tie = 0;
  for (let i = 0; i < samples.length; i++) {
    const key = compId + '_' + i;
    const vote = votes[key];
    const resultEl = document.getElementById('result-' + compId + '-' + i);
    const cardEl = document.querySelector('[data-comp="' + compId + '"][data-idx="' + i + '"]');
    const boxA = document.getElementById('A-' + compId + '-' + i);
    const boxB = document.getElementById('B-' + compId + '-' + i);

    if (resultEl) resultEl.className = 'vote-result';
    if (boxA) boxA.classList.remove('selected-a');
    if (boxB) boxB.classList.remove('selected-b');
    if (cardEl) cardEl.classList.remove('voted');

    if (vote) {
      voted++;
      if (vote === 'A') { winA++; if (boxA) boxA.classList.add('selected-a'); if (resultEl) { resultEl.textContent = '已选 A'; resultEl.classList.add('a'); } if (cardEl) cardEl.classList.add('voted'); }
      else if (vote === 'B') { winB++; if (boxB) boxB.classList.add('selected-b'); if (resultEl) { resultEl.textContent = '已选 B'; resultEl.classList.add('b'); } if (cardEl) cardEl.classList.add('voted'); }
      else if (vote === 'tie') { tie++; if (resultEl) { resultEl.textContent = '平局'; resultEl.classList.add('tie'); } if (cardEl) cardEl.classList.add('voted'); }
      else if (vote === 'skip') { if (resultEl) { resultEl.textContent = '已跳过'; resultEl.classList.add('skip'); } if (cardEl) cardEl.classList.add('voted'); }
    }
  }
  document.getElementById('voted-' + compId).textContent = voted;
  document.getElementById('winA-' + compId).textContent = winA;
  document.getElementById('winB-' + compId).textContent = winB;
  document.getElementById('tie-' + compId).textContent = tie;
}

function vote(compId, idx, choice) {
  const key = compId + '_' + idx;
  votes[key] = choice;
  localStorage.setItem('r1_blind_votes', JSON.stringify(votes));
  refreshVoteDisplay(compId);
}

function showTab(compId) {
  document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.comparison-section').forEach(s => s.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById('section-' + compId).classList.add('active');
  refreshVoteDisplay(compId);
}

function filterCategory(compId, cat) {
  const cards = document.querySelectorAll('[data-comp="' + compId + '"]');
  cards.forEach(card => {
    if (cat === 'all' || card.dataset.category === cat) {
      card.classList.remove('hidden');
    } else {
      card.classList.add('hidden');
    }
  });
  const section = document.getElementById('section-' + compId);
  section.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
}

function exportResults() {
  let results = {};
  let summary = {};
  for (const compId in blindData) {
    const samples = blindData[compId].samples;
    let winA = 0, winB = 0, tie = 0, skip = 0, pending = 0;
    results[compId] = { label: blindData[compId].label, samples: [] };
    for (let i = 0; i < samples.length; i++) {
      const key = compId + '_' + i;
      const vote = votes[key] || '';
      const s = samples[i];
      results[compId].samples.push({ id: s.id, category: s.category, prompt: s.prompt, response_A: s.response_A, response_B: s.response_B, winner: vote });
      if (vote === 'A') winA++;
      else if (vote === 'B') winB++;
      else if (vote === 'tie') tie++;
      else if (vote === 'skip') skip++;
      else pending++;
    }
    summary[compId] = { label: blindData[compId].label, total: samples.length, winA: winA, winB: winB, tie: tie, skip: skip, pending: pending };
  }

  const output = { generated_at: new Date().toISOString(), summary: summary, details: results };
  const blob = new Blob([JSON.stringify(output, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'r1_blind_review_results.json';
  a.click();
  URL.revokeObjectURL(url);

  let newline = String.fromCharCode(10);
  let msg = '=== 盲评结果摘要 ===' + newline + newline;
  for (const compId in summary) {
    const s = summary[compId];
    msg += s.label + ':' + newline;
    msg += '  A胜: ' + s.winA + ' | B胜: ' + s.winB + ' | 平局: ' + s.tie + ' | 跳过: ' + s.skip + ' | 待评: ' + s.pending + newline;
    let rate = ((s.winA + s.winB + s.tie + s.skip) / s.total * 100).toFixed(1);
    msg += '  完成率: ' + rate + '%' + newline + newline;
  }
  alert(msg);
}

function resetAll() {
  if (confirm('确定重置所有投票？此操作不可撤销。')) {
    votes = {};
    localStorage.removeItem('r1_blind_votes');
    for (const compId in blindData) {
      refreshVoteDisplay(compId);
    }
  }
}

// 初始化
for (const compId in blindData) {
  refreshVoteDisplay(compId);
}
</script>

</body>
</html>"""

    # 替换占位符
    html = html.replace("__NAV_TABS__", nav_tabs_html)
    html = html.replace("__COMPARISONS__", comparisons_html)
    html = html.replace("__BLIND_DATA__", embedded_json)

    output_path = OUTPUT_DIR / "r1_blind_review_ui.html"
    output_path.write_text(html, encoding="utf-8")
    print(f"盲评界面已生成: {output_path}")
    print(f"共 {sum(len(info['data'].get('samples', [])) for info in all_data.values())} 条样本待评")
    return output_path


if __name__ == "__main__":
    generate_html()
