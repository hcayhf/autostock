'use strict';

// 符号に意味があるか（= 表示に + を付けて色分けするか）は
// サーバ側の fields.py が signed フラグとして持っている。
const isSigned = (key) => Boolean(state.fieldsByKey[key] && state.fieldsByKey[key].signed);

const state = {
  meta: null,
  fieldsByKey: {},
  ranges: [],          // {field, min, max}
  offset: 0,
  limit: 50,
  sortBy: 'turnover_20d',
  sortDesc: true,
  activePreset: null,
  lastResult: null,
};

const $ = (id) => document.getElementById(id);

// --- 表示整形 ---------------------------------------------------------------

function formatValue(key, value) {
  if (value === null || value === undefined || value === '') return '—';
  const field = state.fieldsByKey[key];
  if (!field || field.dtype === 'text') return String(value);

  const n = Number(value);
  if (!Number.isFinite(n)) return '—';

  switch (field.unit) {
    case '%':
      return `${isSigned(key) && n >= 0 ? '+' : ''}${n.toFixed(2)}%`;
    case '倍':
      return `${n.toFixed(2)}倍`;
    case '円':
      return Math.abs(n) < 100 ? n.toFixed(1) : Math.round(n).toLocaleString('ja-JP');
    case '億円':
    case '百万円':
      return Math.round(n).toLocaleString('ja-JP');
    case '株':
      return Math.round(n).toLocaleString('ja-JP');
    default:
      return n.toFixed(1);
  }
}

function cellClass(key, value) {
  const field = state.fieldsByKey[key];
  if (!field) return '';
  if (value === null || value === undefined) return 'null';
  if (field.dtype === 'text') return 'text';
  if (field.signed) return `num ${Number(value) >= 0 ? 'pos' : 'neg'}`;
  return 'num';
}

// --- 初期化 -----------------------------------------------------------------

async function init() {
  const res = await fetch('/api/meta');
  if (!res.ok) {
    $('freshness').textContent = 'メタ情報の取得に失敗しました';
    return;
  }
  state.meta = await res.json();
  state.meta.fields.forEach((f) => { state.fieldsByKey[f.key] = f; });

  renderFreshness();
  renderPresets();
  renderCategories();
  renderSortOptions();

  $('empty-notice').hidden = !state.meta.empty;

  $('add-range').addEventListener('click', () => addRange());
  $('run').addEventListener('click', () => { state.offset = 0; runScreen(); });
  $('reset').addEventListener('click', resetForm);
  $('prev').addEventListener('click', () => { state.offset = Math.max(0, state.offset - state.limit); runScreen(); });
  $('next').addEventListener('click', () => { state.offset += state.limit; runScreen(); });
  $('text').addEventListener('keydown', (e) => { if (e.key === 'Enter') { state.offset = 0; runScreen(); } });
  $('sort-by').addEventListener('change', (e) => { state.sortBy = e.target.value; });
  $('sort-dir').addEventListener('change', (e) => { state.sortDesc = e.target.value === 'desc'; });
  $('detail-close').addEventListener('click', closeDetail);
  $('detail').addEventListener('click', (e) => { if (e.target === $('detail')) closeDetail(); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeDetail(); });

  if (!state.meta.empty) runScreen();
}

function renderFreshness() {
  const s = state.meta.status || {};
  if (!s.stocks) { $('freshness').textContent = 'データ未取り込み'; return; }
  const parts = [`${s.stocks.toLocaleString('ja-JP')} 銘柄`];
  if (s.price_date) parts.push(`株価 ${s.price_date} 時点`);
  if (s.fundamental_date) parts.push(`財務 ${s.fundamental_date} 時点`);
  $('freshness').textContent = parts.join(' / ');
}

function renderPresets() {
  const box = $('presets');
  box.innerHTML = '';
  state.meta.presets.forEach((preset) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = preset.name;
    btn.title = preset.description;
    btn.setAttribute('aria-pressed', 'false');
    btn.addEventListener('click', () => applyPreset(preset));
    box.appendChild(btn);
  });
}

function applyPreset(preset) {
  state.ranges = preset.ranges.map((r) => ({
    field: r.field,
    min: r.min ?? null,
    max: r.max ?? null,
  }));
  state.sortBy = preset.sort_by;
  state.sortDesc = preset.sort_desc;
  state.activePreset = preset.id;
  state.offset = 0;

  $('sort-by').value = state.sortBy;
  $('sort-dir').value = state.sortDesc ? 'desc' : 'asc';
  [...$('presets').children].forEach((b) => {
    b.setAttribute('aria-pressed', String(b.textContent === preset.name));
  });

  renderRanges();
  runScreen();
}

function renderCategories() {
  const box = $('categories');
  box.innerHTML = '';
  Object.entries(state.meta.categories).forEach(([key, values]) => {
    if (!values.length) return;
    const field = state.fieldsByKey[key];
    const wrap = document.createElement('div');

    const label = document.createElement('label');
    label.textContent = `${field ? field.label : key}（複数選択可）`;
    label.htmlFor = `cat-${key}`;

    const select = document.createElement('select');
    select.id = `cat-${key}`;
    select.multiple = true;
    select.dataset.categoryKey = key;
    values.forEach((v) => select.appendChild(new Option(v, v)));

    wrap.append(label, select);
    box.appendChild(wrap);
  });
}

function renderSortOptions() {
  const select = $('sort-by');
  select.innerHTML = '';
  const groups = {};
  state.meta.fields.filter((f) => f.sortable).forEach((f) => {
    (groups[f.group] ||= []).push(f);
  });
  Object.entries(groups).forEach(([group, fields]) => {
    const og = document.createElement('optgroup');
    og.label = state.meta.groups[group] || group;
    fields.forEach((f) => og.appendChild(new Option(f.label, f.key)));
    select.appendChild(og);
  });
  select.value = state.sortBy;
}

// --- 条件行 -----------------------------------------------------------------

function numericFields() {
  return state.meta.fields.filter((f) => f.filterable && f.dtype === 'real');
}

function addRange(field = null) {
  const candidates = numericFields();
  const used = new Set(state.ranges.map((r) => r.field));
  const next = field || (candidates.find((f) => f.primary && !used.has(f.key)) || candidates[0]).key;
  state.ranges.push({ field: next, min: null, max: null });
  renderRanges();
}

function renderRanges() {
  const box = $('ranges');
  box.innerHTML = '';

  state.ranges.forEach((range, index) => {
    const field = state.fieldsByKey[range.field];
    const row = document.createElement('div');
    row.className = 'range-row';

    const select = document.createElement('select');
    const groups = {};
    numericFields().forEach((f) => { (groups[f.group] ||= []).push(f); });
    Object.entries(groups).forEach(([group, fields]) => {
      const og = document.createElement('optgroup');
      og.label = state.meta.groups[group] || group;
      fields.forEach((f) => og.appendChild(new Option(f.label, f.key)));
      select.appendChild(og);
    });
    select.value = range.field;
    select.title = field ? field.desc : '';
    select.addEventListener('change', (e) => {
      state.ranges[index].field = e.target.value;
      renderRanges();
    });

    const drop = document.createElement('button');
    drop.type = 'button';
    drop.className = 'drop';
    drop.textContent = '×';
    drop.title = 'この条件を削除';
    drop.addEventListener('click', () => {
      state.ranges.splice(index, 1);
      renderRanges();
    });

    const inputs = document.createElement('div');
    inputs.className = 'inputs';

    const min = numberInput('以上', range.min, (v) => { state.ranges[index].min = v; });
    const sep = document.createElement('span');
    sep.className = 'sep';
    sep.textContent = '〜';
    const max = numberInput('以下', range.max, (v) => { state.ranges[index].max = v; });
    const unit = document.createElement('span');
    unit.className = 'unit';
    unit.textContent = field ? field.unit : '';

    inputs.append(min, sep, max, unit);
    row.append(select, drop, inputs);
    box.appendChild(row);
  });

  $('range-hint').hidden = state.ranges.length === 0;
}

function numberInput(placeholder, value, onChange) {
  const input = document.createElement('input');
  input.type = 'number';
  input.step = 'any';
  input.placeholder = placeholder;
  if (value !== null && value !== undefined) input.value = value;
  input.addEventListener('input', (e) => {
    const raw = e.target.value;
    onChange(raw === '' ? null : Number(raw));
  });
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { state.offset = 0; runScreen(); }
  });
  return input;
}

function resetForm() {
  state.ranges = [];
  state.offset = 0;
  state.activePreset = null;
  state.sortBy = 'turnover_20d';
  state.sortDesc = true;
  $('text').value = '';
  $('sort-by').value = state.sortBy;
  $('sort-dir').value = 'desc';
  [...$('presets').children].forEach((b) => b.setAttribute('aria-pressed', 'false'));
  document.querySelectorAll('[data-category-key]').forEach((sel) => {
    [...sel.options].forEach((o) => { o.selected = false; });
  });
  renderRanges();
  runScreen();
}

// --- 検索 -------------------------------------------------------------------

function collectQuery() {
  const categories = {};
  document.querySelectorAll('[data-category-key]').forEach((select) => {
    const chosen = [...select.selectedOptions].map((o) => o.value);
    if (chosen.length) categories[select.dataset.categoryKey] = chosen;
  });

  return {
    text: $('text').value.trim() || null,
    categories,
    ranges: state.ranges.filter((r) => r.min !== null || r.max !== null),
    sort_by: state.sortBy,
    sort_desc: state.sortDesc,
    limit: state.limit,
    offset: state.offset,
  };
}

async function runScreen() {
  $('summary').textContent = '検索中…';
  let res;
  try {
    res = await fetch('/api/screen', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(collectQuery()),
    });
  } catch (e) {
    $('summary').textContent = `通信に失敗しました: ${e.message}`;
    return;
  }
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    $('summary').textContent = `検索エラー: ${detail.detail || res.status}`;
    return;
  }
  state.lastResult = await res.json();
  renderResults();
}

function renderResults() {
  const result = state.lastResult;
  const from = result.total === 0 ? 0 : result.offset + 1;
  const to = result.offset + result.count;

  $('summary').innerHTML = result.total
    ? `<strong>${result.total.toLocaleString('ja-JP')}</strong> 件が該当（${from}〜${to} 件目を表示）`
    : '該当する銘柄がありません';
  $('page-info').textContent = result.total
    ? `${Math.floor(result.offset / result.limit) + 1} / ${Math.max(1, Math.ceil(result.total / result.limit))}`
    : '';
  $('prev').disabled = result.offset === 0;
  $('next').disabled = to >= result.total;

  const head = $('thead-row');
  head.innerHTML = '';
  result.columns.forEach((key) => {
    const field = state.fieldsByKey[key];
    const th = document.createElement('th');
    th.textContent = field ? field.label : key;
    th.title = field && field.desc ? field.desc : '';
    if (field && field.dtype === 'text') th.classList.add('text');
    if (field && field.sortable) {
      if (key === state.sortBy) th.textContent += state.sortDesc ? ' ▼' : ' ▲';
      th.addEventListener('click', () => {
        if (state.sortBy === key) state.sortDesc = !state.sortDesc;
        else { state.sortBy = key; state.sortDesc = true; }
        $('sort-by').value = state.sortBy;
        $('sort-dir').value = state.sortDesc ? 'desc' : 'asc';
        state.offset = 0;
        runScreen();
      });
    }
    head.appendChild(th);
  });

  const body = $('tbody');
  body.innerHTML = '';
  if (!result.rows.length) {
    const tr = document.createElement('tr');
    const td = document.createElement('td');
    td.colSpan = Math.max(1, result.columns.length);
    td.className = 'empty-row';
    td.textContent = '条件を緩めるか、指標を欠損している銘柄が多くないか確認してください。';
    tr.appendChild(td);
    body.appendChild(tr);
    return;
  }

  result.rows.forEach((row) => {
    const tr = document.createElement('tr');
    tr.addEventListener('click', () => openDetail(row.code));
    result.columns.forEach((key) => {
      const td = document.createElement('td');
      td.className = cellClass(key, row[key]);
      td.textContent = formatValue(key, row[key]);
      tr.appendChild(td);
    });
    body.appendChild(tr);
  });
}

// --- 個別銘柄 ---------------------------------------------------------------

async function openDetail(code) {
  const res = await fetch(`/api/stocks/${encodeURIComponent(code)}`);
  if (!res.ok) return;
  const data = await res.json();
  const snap = data.snapshot || {};
  const stock = data.stock;

  const metrics = [
    'close', 'market_cap', 'per', 'pbr', 'dividend_yield', 'roe',
    'equity_ratio', 'rsi_14', 'ret_250d', 'sma_dev_200', 'volatility_60d', 'turnover_20d',
  ];

  const body = $('detail-body');
  body.innerHTML = '';

  const title = document.createElement('h3');
  title.textContent = `${stock.code} ${stock.name}`;
  const sub = document.createElement('p');
  sub.className = 'sub';
  sub.textContent = [stock.market, stock.sector33, stock.scale, snap.price_date && `${snap.price_date} 時点`]
    .filter(Boolean).join(' ・ ');
  body.append(title, sub);

  body.appendChild(sparkline(data.prices));

  const dl = document.createElement('div');
  dl.className = 'metrics';
  metrics.forEach((key) => {
    const field = state.fieldsByKey[key];
    if (!field) return;
    const item = document.createElement('dl');
    item.className = 'metric';
    const dt = document.createElement('dt');
    dt.textContent = field.label;
    const dd = document.createElement('dd');
    dd.textContent = formatValue(key, snap[key]);
    dd.className = cellClass(key, snap[key]).replace('num', '').trim();
    item.append(dt, dd);
    dl.appendChild(item);
  });
  body.appendChild(dl);

  $('detail').hidden = false;
}

function closeDetail() { $('detail').hidden = true; }

/** 終値の推移を SVG の折れ線で描く（外部ライブラリ不要）。 */
function sparkline(prices) {
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('class', 'chart');
  svg.setAttribute('preserveAspectRatio', 'none');

  const points = prices.filter((p) => p.close !== null);
  const W = 700, H = 190, PAD_X = 8, PAD_Y = 20;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  if (points.length < 2) return svg;

  const values = points.map((p) => p.close);
  const min = Math.min(...values), max = Math.max(...values);
  const span = max - min || 1;
  const x = (i) => PAD_X + (i / (points.length - 1)) * (W - PAD_X * 2);
  const y = (v) => PAD_Y + (1 - (v - min) / span) * (H - PAD_Y * 2);

  const line = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(1)},${y(p.close).toFixed(1)}`).join(' ');

  const area = document.createElementNS(ns, 'path');
  area.setAttribute('class', 'area');
  area.setAttribute('d', `${line} L${x(points.length - 1).toFixed(1)},${H - PAD_Y} L${x(0).toFixed(1)},${H - PAD_Y} Z`);

  const path = document.createElementNS(ns, 'path');
  path.setAttribute('class', 'line');
  path.setAttribute('d', line);

  svg.append(area, path);

  // 最小値ラベルを下端に置くと折れ線と重なるので、レンジは左上に 1 行でまとめる
  const range = document.createElementNS(ns, 'text');
  range.setAttribute('x', PAD_X);
  range.setAttribute('y', PAD_Y - 2);
  range.textContent =
    `${Math.round(min).toLocaleString('ja-JP')} 〜 ${Math.round(max).toLocaleString('ja-JP')} 円`;
  svg.appendChild(range);

  const label = document.createElementNS(ns, 'text');
  label.setAttribute('x', W - PAD_X);
  label.setAttribute('y', PAD_Y - 2);
  label.setAttribute('text-anchor', 'end');
  label.textContent = `${points[0].date} 〜 ${points[points.length - 1].date}`;
  svg.appendChild(label);

  return svg;
}

init();
