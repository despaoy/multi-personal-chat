#!/usr/bin/env python3
"""Build unified review UI HTML combining blindfix samples + V4 daily candidates."""
from __future__ import annotations

import json
from pathlib import Path

BLINDFIX = Path(
    "c:/Users/13474/Desktop/qqchat-enhanced/backend/data/character_dialogues/experiments/kisaki_v4_blindfix.jsonl"
)
CANDIDATES = Path(
    "c:/Users/13474/Desktop/qqchat-enhanced/backend/data/character_dialogues/experiments/kisaki_v4_candidates_filtered.jsonl"
)
LIFESTYLE = Path(
    "c:/Users/13474/Desktop/qqchat-enhanced/backend/data/character_dialogues/experiments/kisaki_v4_lifestyle.jsonl"
)
YORUKO = Path(
    "c:/Users/13474/Desktop/qqchat-enhanced/backend/data/character_dialogues/experiments/kisaki_v4_yoruko.jsonl"
)
OUTPUT = Path(
    "c:/Users/13474/Desktop/qqchat-enhanced/runtime_analysis/v4_unified_review_ui.html"
)


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main():
    blindfix = load_jsonl(BLINDFIX)
    candidates = load_jsonl(CANDIDATES)
    lifestyle = load_jsonl(LIFESTYLE)
    yoruko = load_jsonl(YORUKO)

    blindfix_json = json.dumps(blindfix, ensure_ascii=False)
    candidates_json = json.dumps(candidates, ensure_ascii=False)
    lifestyle_json = json.dumps(lifestyle, ensure_ascii=False)
    yoruko_json = json.dumps(yoruko, ensure_ascii=False)

    html = HTML_TEMPLATE.replace("__BLINDFIX_DATA__", blindfix_json)
    html = html.replace("__CANDIDATES_DATA__", candidates_json)
    html = html.replace("__LIFESTYLE_DATA__", lifestyle_json)
    html = html.replace("__YORUKO_DATA__", yoruko_json)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Generated unified review UI -> {OUTPUT}")
    print(f"Blindfix: {len(blindfix)} samples, Candidates: {len(candidates)} samples, Lifestyle: {len(lifestyle)} samples, Yoruko: {len(yoruko)} samples")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>V4 统一审查界面</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; background: #f5f5f5; color: #333; }
.header { background: #2c3e50; color: white; padding: 16px 24px; position: sticky; top: 0; z-index: 100; }
.header h1 { font-size: 18px; margin-bottom: 4px; }
.header .stats { font-size: 13px; opacity: 0.8; }
.tabs { display: flex; gap: 4px; padding: 0 24px; background: #34495e; }
.tab { padding: 10px 20px; color: #bdc3c7; cursor: pointer; border-radius: 6px 6px 0 0; font-size: 14px; }
.tab.active { background: #f5f5f5; color: #2c3e50; font-weight: 600; }
.tab:hover { color: white; }
.controls { padding: 12px 24px; background: white; border-bottom: 1px solid #ddd; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
.controls select, .controls button { padding: 6px 12px; border: 1px solid #ccc; border-radius: 4px; font-size: 13px; }
.controls button { cursor: pointer; background: #ecf0f1; }
.controls button:hover { background: #d5dbdb; }
.controls .counter { margin-left: auto; font-size: 13px; color: #666; }
.controls .counter span { font-weight: 600; color: #2c3e50; }
.content { padding: 16px 24px; max-width: 900px; margin: 0 auto; }
.card { background: white; border-radius: 8px; padding: 16px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); border-left: 4px solid #ccc; }
.card.approved { border-left-color: #27ae60; }
.card.rejected { border-left-color: #e74c3c; opacity: 0.6; }
.card-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.card-id { font-size: 12px; color: #888; font-family: monospace; }
.card-scene { background: #ecf0f1; padding: 2px 8px; border-radius: 10px; font-size: 11px; color: #555; }
.card-note { font-size: 11px; color: #999; margin-bottom: 8px; font-style: italic; }
.orig-prompt { background: #fce4ec; padding: 8px 10px; border-radius: 4px; margin-bottom: 8px; font-size: 13px; color: #888; }
.orig-prompt::before { content: "原题: "; font-weight: 600; color: #c0392b; }
.turn { margin-bottom: 6px; }
.turn-label { display: inline-block; width: 40px; font-size: 12px; font-weight: 600; vertical-align: top; }
.turn-label.human { color: #2980b9; }
.turn-label.assistant { color: #8e44ad; }
.turn-text { display: inline-block; width: calc(100% - 50px); font-size: 14px; line-height: 1.6; }
.turn-text.editable { background: #fffde7; padding: 4px 8px; border-radius: 4px; cursor: text; }
.turn-text.editing { background: #fff9c4; outline: 2px solid #f39c12; }
.actions { margin-top: 10px; display: flex; gap: 8px; }
.actions button { padding: 4px 14px; border: none; border-radius: 4px; cursor: pointer; font-size: 13px; }
.btn-approve { background: #27ae60; color: white; }
.btn-reject { background: #e74c3c; color: white; }
.btn-edit { background: #f39c12; color: white; }
.btn-approve:hover { background: #229954; }
.btn-reject:hover { background: #cb4335; }
.btn-edit:hover { background: #d68910; }
.hidden { display: none; }
.export-bar { position: fixed; bottom: 0; left: 0; right: 0; background: #2c3e50; color: white; padding: 12px 24px; display: flex; justify-content: space-between; align-items: center; z-index: 100; }
.export-bar button { padding: 8px 20px; background: #27ae60; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; }
.export-bar button:hover { background: #229954; }
</style>
</head>
<body>
<div class="header">
  <h1>V4 统一审查界面 — 月社妃对话质量审查</h1>
  <div class="stats" id="stats">加载中...</div>
</div>
<div class="tabs">
  <div class="tab active" data-tab="blindfix" onclick="switchTab('blindfix')">盲评修复 (55)</div>
  <div class="tab" data-tab="lifestyle" onclick="switchTab('lifestyle')">生活状态 (30)</div>
  <div class="tab" data-tab="yoruko" onclick="switchTab('yoruko')">夜子关系 (30)</div>
  <div class="tab" data-tab="candidates" onclick="switchTab('candidates')">日常候选筛选 (111)</div>
</div>
<div class="controls">
  <select id="filter" onchange="applyFilter()">
    <option value="all">全部分类</option>
  </select>
  <button onclick="approveAllFiltered()">全选通过</button>
  <button onclick="rejectAllFiltered()">全选拒绝</button>
  <div class="counter">已审: <span id="reviewed">0</span> / <span id="total">0</span> | 通过: <span id="approved" style="color:#27ae60">0</span></div>
</div>
<div class="content" id="content"></div>
<div class="export-bar">
  <span id="export-info">审查完成后点击导出</span>
  <button onclick="exportResults()">导出通过的样本 (JSONL)</button>
</div>
<script>
const blindfixData = __BLINDFIX_DATA__;
const candidatesData = __CANDIDATES_DATA__;
const lifestyleData = __LIFESTYLE_DATA__;
const yorukoData = __YORUKO_DATA__;
const reviews = {};
let currentTab = 'blindfix';

function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function getCurrentData() {
  if (currentTab === 'blindfix') return blindfixData;
  if (currentTab === 'lifestyle') return lifestyleData;
  if (currentTab === 'yoruko') return yorukoData;
  return candidatesData;
}

function switchTab(tab) {
  currentTab = tab;
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tab));
  renderFilter();
  renderContent();
}

function renderFilter() {
  const data = getCurrentData();
  const scenes = [...new Set(data.map(s => s.metadata.scene || s.metadata.category || 'unknown'))];
  const sel = document.getElementById('filter');
  sel.innerHTML = '<option value="all">全部分类</option>' + scenes.map(s => `<option value="${s}">${s}</option>`).join('');
}

function applyFilter() {
  renderContent();
}

function getFilteredData() {
  const data = getCurrentData();
  const filter = document.getElementById('filter').value;
  if (filter === 'all') return data;
  return data.filter(s => (s.metadata.scene || s.metadata.category || '') === filter);
}

function renderCard(s) {
  const convs = s.conversations || [];
  const r = reviews[s.id] || {};
  const cls = r.status ? ' ' + r.status : '';
  let turnsHtml = '';
  for (let i = 0; i < convs.length; i++) {
    const c = convs[i];
    const isAssistant = c.from === 'assistant';
    const isLast = (i === convs.length - 1);
    const editable = isAssistant && isLast ? ' editable' : '';
    turnsHtml += '<div class="turn"><span class="turn-label ' + c.from + '">' + (isAssistant ? '妃' : '用户') + '</span><div class="turn-text' + editable + '" id="text-' + s.id + '-' + i + '">' + escapeHtml(c.value) + '</div></div>';
  }
  let noteHtml = '';
  if (s.metadata.note) {
    noteHtml = '<div class="card-note">' + escapeHtml(s.metadata.note) + '</div>';
  }
  let origHtml = '';
  if (s.metadata.original_id) {
    origHtml = '<div class="orig-prompt">' + escapeHtml(s.metadata.note || s.metadata.original_id) + '</div>';
  }
  const scene = s.metadata.scene || s.metadata.category || '';
  return '<div class="card' + cls + '" id="card-' + s.id + '">'
    + '<div class="card-head"><span class="card-id">' + s.id + '</span><span class="card-scene">' + scene + '</span></div>'
    + origHtml + noteHtml + turnsHtml
    + '<div class="actions">'
    + '<button class="btn-approve" onclick="approve(\'' + s.id + '\')">通过</button>'
    + '<button class="btn-reject" onclick="reject(\'' + s.id + '\')">拒绝</button>'
    + '<button class="btn-edit" onclick="startEdit(\'' + s.id + '\')">编辑</button>'
    + '</div></div>';
}

function renderContent() {
  const data = getFilteredData();
  const container = document.getElementById('content');
  container.innerHTML = data.map(renderCard).join('');
  updateCounter();
}

function approve(id) {
  const card = document.getElementById('card-' + id);
  if (!card) return;
  const data = getCurrentData();
  const s = data.find(x => x.id === id);
  reviews[id] = { status: 'approved', data: JSON.parse(JSON.stringify(s)) };
  card.classList.remove('rejected');
  card.classList.add('approved');
  updateCounter();
}

function reject(id) {
  const card = document.getElementById('card-' + id);
  if (!card) return;
  reviews[id] = { status: 'rejected' };
  card.classList.remove('approved');
  card.classList.add('rejected');
  updateCounter();
}

function startEdit(id) {
  const data = getCurrentData();
  const s = data.find(x => x.id === id);
  if (!s) return;
  const convs = s.conversations || [];
  for (let i = convs.length - 1; i >= 0; i--) {
    if (convs[i].from === 'assistant') {
      const el = document.getElementById('text-' + id + '-' + i);
      if (el) {
        el.classList.add('editing');
        el.contentEditable = true;
        el.focus();
        el.onblur = function() {
          el.contentEditable = false;
          el.classList.remove('editing');
          convs[i].value = el.innerText;
        };
        el.onkeydown = function(e) {
          if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); el.blur(); }
        };
      }
      break;
    }
  }
}

function approveAllFiltered() {
  const data = getFilteredData();
  data.forEach(s => approve(s.id));
}

function rejectAllFiltered() {
  const data = getFilteredData();
  data.forEach(s => reject(s.id));
}

function updateCounter() {
  const data = getCurrentData();
  const total = data.length;
  const reviewed = data.filter(s => reviews[s.id]).length;
  const approved = data.filter(s => reviews[s.id] && reviews[s.id].status === 'approved').length;
  document.getElementById('total').textContent = total;
  document.getElementById('reviewed').textContent = reviewed;
  document.getElementById('approved').textContent = approved;
  document.getElementById('stats').textContent = '盲评修复: ' + blindfixData.length + ' 条 | 生活状态: ' + lifestyleData.length + ' 条 | 夜子关系: ' + yorukoData.length + ' 条 | 日常候选: ' + candidatesData.length + ' 条';
  const allApproved = (blindfixData.filter(s => reviews[s.id] && reviews[s.id].status === 'approved').length)
    + (lifestyleData.filter(s => reviews[s.id] && reviews[s.id].status === 'approved').length)
    + (yorukoData.filter(s => reviews[s.id] && reviews[s.id].status === 'approved').length)
    + (candidatesData.filter(s => reviews[s.id] && reviews[s.id].status === 'approved').length);
  document.getElementById('export-info').textContent = '当前通过: ' + allApproved + ' 条 (盲评+生活+夜子+日常合计)';
}

function exportResults() {
  const approved = [];
  for (const s of blindfixData) {
    const r = reviews[s.id];
    if (r && r.status === 'approved') approved.push(r.data || s);
  }
  for (const s of lifestyleData) {
    const r = reviews[s.id];
    if (r && r.status === 'approved') approved.push(r.data || s);
  }
  for (const s of yorukoData) {
    const r = reviews[s.id];
    if (r && r.status === 'approved') approved.push(r.data || s);
  }
  for (const s of candidatesData) {
    const r = reviews[s.id];
    if (r && r.status === 'approved') approved.push(r.data || s);
  }
  if (approved.length === 0) {
    alert('还没有通过的样本，请先审查。');
    return;
  }
  const lines = approved.map(s => JSON.stringify(s));
  const blob = new Blob([lines.join('\n')], { type: 'application/jsonl' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'v4_approved_samples.jsonl';
  a.click();
  URL.revokeObjectURL(url);
  alert('已导出 ' + approved.length + ' 条通过样本。');
}

renderFilter();
renderContent();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
