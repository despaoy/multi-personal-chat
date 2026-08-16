"""生成 V4 候选样本的交互式审查界面 HTML。"""
from __future__ import annotations

import json
from pathlib import Path

CANDIDATES_PATH = Path(r"c:\Users\13474\Desktop\qqchat-enhanced\backend\data\character_dialogues\experiments\kisaki_v4_candidates.jsonl")
OUTPUT_PATH = Path(r"c:\Users\13474\Desktop\qqchat-enhanced\runtime_analysis\v4_review_ui.html")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_candidates() -> list[dict]:
    samples = []
    with open(CANDIDATES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def main() -> int:
    samples = load_candidates()
    print(f"加载 {len(samples)} 条候选样本")

    # 按场景分组统计
    from collections import Counter
    scene_counts = Counter(s["metadata"]["scene"] for s in samples)

    # 构建嵌入数据（精简版，只含审查需要的信息）
    embedded = []
    for s in samples:
        convs = s.get("conversations", [])
        embedded.append({
            "id": s["id"],
            "scene": s["metadata"]["scene"],
            "turns": s["metadata"]["turns"],
            "conversations": convs,
        })

    embedded_json = json.dumps(embedded, ensure_ascii=False)

    # 场景列表
    scenes = sorted(scene_counts.keys())
    scene_tabs = "".join(
        f'<button class="scene-tab" data-scene="{sc}" onclick="filterScene(\'{sc}\')">{sc} ({scene_counts[sc]})</button>'
        for sc in scenes
    )

    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>V4 候选样本审查 - 月社妃</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, "Microsoft YaHei", "PingFang SC", sans-serif; padding: 16px; background: #f7fafc; color: #2d3748; line-height: 1.6; }
  .header { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 16px; }
  .header h1 { font-size: 22px; color: #1a202c; margin-bottom: 8px; }
  .header p { font-size: 13px; color: #718096; margin-bottom: 8px; }
  .stats-bar { display: flex; gap: 16px; margin-top: 12px; flex-wrap: wrap; }
  .stat-card { background: #ebf8ff; padding: 8px 16px; border-radius: 6px; font-size: 13px; }
  .stat-card b { color: #2b6cb0; font-size: 18px; }
  .info-box { background: #fffbeb; border-left: 4px solid #f6ad55; padding: 12px; margin: 12px 0; border-radius: 4px; font-size: 13px; }
  .scene-tabs { display: flex; gap: 6px; margin-bottom: 16px; flex-wrap: wrap; }
  .scene-tab { padding: 6px 14px; background: #edf2f7; border: 1px solid #cbd5e0; border-radius: 20px; cursor: pointer; font-size: 12px; color: #4a5568; transition: all 0.2s; }
  .scene-tab.active { background: #4299e1; color: white; border-color: #3182ce; }
  .scene-tab:hover { background: #bee3f8; }
  .scene-tab.active:hover { background: #3182ce; }
  .sample-list { max-width: 900px; margin: 0 auto; }
  .sample-card { background: white; padding: 16px; border-radius: 8px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); border-left: 3px solid #cbd5e0; transition: all 0.2s; }
  .sample-card.approved { border-left-color: #48bb78; background: #f0fff4; }
  .sample-card.rejected { border-left-color: #fc8181; background: #fff5f5; opacity: 0.6; }
  .sample-card.hidden { display: none; }
  .card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
  .sample-id { font-family: monospace; font-size: 11px; color: #718096; }
  .scene-badge { padding: 2px 10px; border-radius: 10px; font-size: 11px; font-weight: 600; color: white; }
  .scene-badge.问候闲聊 { background: #4299e1; }
  .scene-badge.兴趣偏好 { background: #48bb78; }
  .scene-badge.人物关系 { background: #ed8936; }
  .scene-badge.情感倾诉 { background: #9f7aea; }
  .scene-badge.请求帮助 { background: #38b2ac; }
  .scene-badge.角色人设 { background: #f56565; }
  .scene-badge.安全边界 { background: #2d3748; }
  .conversation { margin-bottom: 10px; }
  .turn { display: flex; gap: 8px; margin-bottom: 6px; align-items: flex-start; }
  .turn-label { padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; min-width: 60px; text-align: center; }
  .turn-label.human { background: #ebf8ff; color: #2b6cb0; }
  .turn-label.assistant { background: #f0fff4; color: #2f855a; }
  .turn-text { flex: 1; font-size: 14px; padding: 6px 10px; background: #f7fafc; border-radius: 4px; }
  .turn-text.editable { outline: 2px dashed #f6ad55; background: #fffbeb; }
  .turn-text.editing { outline: 2px solid #4299e1; background: #ebf8ff; }
  .actions { display: flex; gap: 8px; margin-top: 10px; align-items: center; }
  .btn { padding: 6px 16px; border: none; border-radius: 4px; cursor: pointer; font-size: 12px; font-weight: 600; transition: all 0.2s; }
  .btn-approve { background: #48bb78; color: white; }
  .btn-approve:hover { background: #38a169; }
  .btn-reject { background: #fc8181; color: white; }
  .btn-reject:hover { background: #f56565; }
  .btn-edit { background: #f6ad55; color: white; }
  .btn-edit:hover { background: #ed8936; }
  .btn-save { background: #4299e1; color: white; }
  .btn-save:hover { background: #3182ce; }
  .btn-cancel { background: #a0aec0; color: white; }
  .btn-cancel:hover { background: #718096; }
  .status-tag { font-size: 12px; font-weight: 600; margin-left: auto; }
  .status-tag.approved { color: #2f855a; }
  .status-tag.rejected { color: #c53030; }
  .actions-bar { position: sticky; bottom: 0; background: white; padding: 12px 20px; box-shadow: 0 -2px 8px rgba(0,0,0,0.1); display: flex; justify-content: space-between; align-items: center; margin: 16px -16px -16px; border-radius: 0 0 8px 8px; z-index: 100; }
  .btn-export { background: #4299e1; color: white; padding: 8px 24px; border: none; border-radius: 6px; cursor: pointer; font-size: 14px; font-weight: 600; }
  .btn-export:hover { background: #3182ce; }
  .btn-batch { background: #48bb78; color: white; padding: 6px 14px; border: none; border-radius: 4px; cursor: pointer; font-size: 12px; margin-right: 8px; }
  .btn-batch:hover { background: #38a169; }
  .btn-batch.reject { background: #fc8181; }
  .btn-batch.reject:hover { background: #f56565; }
  .edit-area { width: 100%; min-height: 60px; padding: 8px; border: 2px solid #4299e1; border-radius: 4px; font-size: 14px; font-family: inherit; resize: vertical; }
</style>
</head>
<body>

<div class="header">
  <h1>V4 候选样本审查界面</h1>
  <p>月社妃角色对话 | 共 <b id="totalCount">0</b> 条样本 | 审查通过后加入训练集</p>
  <div class="stats-bar">
    <div class="stat-card">已审核: <b id="reviewedCount">0</b></div>
    <div class="stat-card" style="background:#f0fff4;">通过: <b id="approvedCount" style="color:#2f855a;">0</b></div>
    <div class="stat-card" style="background:#fff5f5;">拒绝: <b id="rejectedCount" style="color:#c53030;">0</b></div>
    <div class="stat-card" style="background:#fffbeb;">待审: <b id="pendingCount" style="color:#d69e2e;">0</b></div>
  </div>
  <div class="info-box">
    <b>审查规则：</b>逐条审查对话，判断回复是否贴合月社妃角色（简洁、直接、第一人称、克制、敏锐、反问、适度挖苦）。<br>
    <b>操作：</b>通过（绿色）/ 拒绝（红色）/ 编辑（修改回复内容后通过）。进度自动保存。<br>
    <b>导出：</b>点击导出按钮，生成通过审查的JSONL文件，加入训练集。
  </div>
</div>

<div class="scene-tabs">
  <button class="scene-tab active" data-scene="all" onclick="filterScene('all')">全部 (<span id="count-all">0</span>)</button>
  __SCENE_TABS__
</div>

<div class="sample-list" id="sampleList">
  __SAMPLE_CARDS__
</div>

<div class="actions-bar">
  <div>
    <button class="btn-batch" onclick="batchApprove()">一键通过当前页</button>
    <button class="btn-batch reject" onclick="batchReject()">一键拒绝当前页</button>
    <span style="font-size:12px;color:#718096;margin-left:12px;">进度自动保存</span>
  </div>
  <button class="btn-export" onclick="exportResults()">导出通过的样本</button>
</div>

<script>
const allData = __EMBEDDED_DATA__;
let reviews = JSON.parse(localStorage.getItem('v4_reviews') || '{}');
let editingId = null;

function renderCard(s) {
  const convs = s.conversations || [];
  let turnsHtml = '';
  for (let i = 0; i < convs.length; i++) {
    const c = convs[i];
    const isAssistant = c.from === 'assistant';
    turnsHtml += '<div class="turn"><span class="turn-label ' + c.from + '">' + (isAssistant ? '妃' : '用户') + '</span><div class="turn-text' + (isAssistant ? ' editable' : '') + '" id="text-' + s.id + '-' + i + '">' + escapeHtml(c.value) + '</div></div>';
  }

  const review = reviews[s.id] || { status: 'pending', edited: null };
  const statusClass = review.status === 'approved' ? 'approved' : (review.status === 'rejected' ? 'rejected' : '');
  const statusTag = review.status === 'approved' ? '<span class="status-tag approved">已通过</span>' : (review.status === 'rejected' ? '<span class="status-tag rejected">已拒绝</span>' : '');

  let actionsHtml = '';
  if (editingId === s.id) {
    actionsHtml = '<button class="btn btn-save" onclick="saveEdit(\'' + s.id + '\')">保存修改</button><button class="btn btn-cancel" onclick="cancelEdit()">取消</button>';
  } else {
    actionsHtml = '<button class="btn btn-approve" onclick="approve(\'' + s.id + '\')">通过</button><button class="btn btn-reject" onclick="reject(\'' + s.id + '\')">拒绝</button><button class="btn btn-edit" onclick="startEdit(\'' + s.id + '\')">编辑回复</button>' + statusTag;
  }

  return '<div class="sample-card ' + statusClass + '" data-scene="' + s.scene + '" data-id="' + s.id + '">' +
    '<div class="card-header"><span class="sample-id">' + s.id + ' (' + s.turns + '轮)</span><span class="scene-badge ' + s.scene + '">' + s.scene + '</span></div>' +
    '<div class="conversation">' + turnsHtml + '</div>' +
    '<div class="actions">' + actionsHtml + '</div>' +
    '</div>';
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function renderAll() {
  const listEl = document.getElementById('sampleList');
  listEl.innerHTML = allData.map(renderCard).join('');
  updateStats();
}

function updateStats() {
  let approved = 0, rejected = 0, pending = 0;
  for (const s of allData) {
    const r = reviews[s.id];
    if (r && r.status === 'approved') approved++;
    else if (r && r.status === 'rejected') rejected++;
    else pending++;
  }
  document.getElementById('totalCount').textContent = allData.length;
  document.getElementById('reviewedCount').textContent = approved + rejected;
  document.getElementById('approvedCount').textContent = approved;
  document.getElementById('rejectedCount').textContent = rejected;
  document.getElementById('pendingCount').textContent = pending;
}

function approve(id) {
  reviews[id] = { status: 'approved', edited: reviews[id] && reviews[id].edited ? reviews[id].edited : null };
  localStorage.setItem('v4_reviews', JSON.stringify(reviews));
  renderAll();
}

function reject(id) {
  reviews[id] = { status: 'rejected', edited: null };
  localStorage.setItem('v4_reviews', JSON.stringify(reviews));
  renderAll();
}

function startEdit(id) {
  editingId = id;
  renderAll();
  // 将assistant回复变为可编辑textarea
  const sample = allData.find(s => s.id === id);
  if (!sample) return;
  for (let i = 0; i < sample.conversations.length; i++) {
    const c = sample.conversations[i];
    if (c.from === 'assistant') {
      const el = document.getElementById('text-' + id + '-' + i);
      if (el) {
        const text = (reviews[id] && reviews[id].edited) ? reviews[id].edited : c.value;
        el.innerHTML = '<textarea class="edit-area" id="editarea-' + id + '-' + i + '">' + escapeHtml(text) + '</textarea>';
        el.classList.add('editing');
      }
    }
  }
}

function saveEdit(id) {
  const sample = allData.find(s => s.id === id);
  if (!sample) return;
  // 收集所有assistant编辑框的内容
  let editedText = '';
  for (let i = 0; i < sample.conversations.length; i++) {
    const c = sample.conversations[i];
    if (c.from === 'assistant') {
      const area = document.getElementById('editarea-' + id + '-' + i);
      if (area) {
        editedText = area.value;
      }
    }
  }
  reviews[id] = { status: 'approved', edited: editedText };
  localStorage.setItem('v4_reviews', JSON.stringify(reviews));
  editingId = null;
  renderAll();
}

function cancelEdit() {
  editingId = null;
  renderAll();
}

function filterScene(scene) {
  document.querySelectorAll('.scene-tab').forEach(t => t.classList.remove('active'));
  event.target.classList.add('active');
  document.querySelectorAll('.sample-card').forEach(card => {
    if (scene === 'all' || card.dataset.scene === scene) {
      card.classList.remove('hidden');
    } else {
      card.classList.add('hidden');
    }
  });
}

function batchApprove() {
  const visible = document.querySelectorAll('.sample-card:not(.hidden)');
  visible.forEach(card => {
    const id = card.dataset.id;
    if (!reviews[id] || reviews[id].status !== 'rejected') {
      reviews[id] = { status: 'approved', edited: reviews[id] && reviews[id].edited ? reviews[id].edited : null };
    }
  });
  localStorage.setItem('v4_reviews', JSON.stringify(reviews));
  renderAll();
}

function batchReject() {
  const visible = document.querySelectorAll('.sample-card:not(.hidden)');
  visible.forEach(card => {
    const id = card.dataset.id;
    if (!reviews[id] || reviews[id].status !== 'approved') {
      reviews[id] = { status: 'rejected', edited: null };
    }
  });
  localStorage.setItem('v4_reviews', JSON.stringify(reviews));
  renderAll();
}

function exportResults() {
  const approved = [];
  for (const s of allData) {
    const r = reviews[s.id];
    if (r && r.status === 'approved') {
      const edited = JSON.parse(JSON.stringify(s));
      if (r.edited) {
        // 应用编辑后的回复
        for (let i = edited.conversations.length - 1; i >= 0; i--) {
          if (edited.conversations[i].from === 'assistant') {
            edited.conversations[i].value = r.edited;
            break;
          }
        }
      }
      edited.review_status = 'approved';
      approved.push(edited);
    }
  }

  const output = {
    exported_at: new Date().toISOString(),
    total_approved: approved.length,
    samples: approved,
  };
  const blob = new Blob([JSON.stringify(output, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'v4_approved_samples.json';
  a.click();
  URL.revokeObjectURL(url);

  let msg = '审查结果摘要\\n\\n';
  msg += '通过: ' + approved.length + ' / ' + allData.length + '\\n';
  let rate = (approved.length / allData.length * 100).toFixed(1);
  msg += '通过率: ' + rate + '%\\n\\n';
  msg += '通过样本已导出为 v4_approved_samples.json\\n';
  msg += '请将文件发给我，我会将其加入训练集。';
  alert(msg);
}

// 初始化
renderAll();
</script>

</body>
</html>"""

    # 生成样本卡片
    cards_html = "\n".join(
        f'<div class="sample-card" data-scene="{s["metadata"]["scene"]}" data-id="{s["id"]}">'
        f'<div class="card-header"><span class="sample-id">{s["id"]} ({s["metadata"]["turns"]}轮)</span>'
        f'<span class="scene-badge {s["metadata"]["scene"]}">{s["metadata"]["scene"]}</span></div>'
        f'<div class="conversation">'
        + "".join(
            f'<div class="turn"><span class="turn-label {c["from"]}">{"妃" if c["from"]=="assistant" else "用户"}</span>'
            f'<div class="turn-text{" editable" if c["from"]=="assistant" else ""}" id="text-{s["id"]}-{i}">{c["value"]}</div></div>'
            for i, c in enumerate(s["conversations"])
        )
        + f'</div><div class="actions">'
        f'<button class="btn btn-approve" onclick="approve(\'{s["id"]}\')">通过</button>'
        f'<button class="btn btn-reject" onclick="reject(\'{s["id"]}\')">拒绝</button>'
        f'<button class="btn btn-edit" onclick="startEdit(\'{s["id"]}\')">编辑回复</button>'
        f'</div></div>'
        for s in samples
    )

    # 替换占位符
    html = html.replace("__SCENE_TABS__", scene_tabs)
    html = html.replace("__SAMPLE_CARDS__", cards_html)
    html = html.replace("__EMBEDDED_DATA__", embedded_json)

    # 更新计数
    for sc in scenes:
        html = html.replace(f'id="count-{sc}"', f'id="count-{sc}">{scene_counts[sc]}')
    html = html.replace('id="count-all"', f'id="count-all">{len(samples)}')

    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"审查界面已生成: {OUTPUT_PATH}")
    print(f"共 {len(samples)} 条样本待审查")
    print(f"场景分布: {dict(scene_counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
