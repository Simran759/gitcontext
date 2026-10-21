"""The browser chat UI, served as a single self-contained HTML page.

Kept as one string with inline CSS + vanilla JS (no CDN, works offline). The
page POSTs the question to /query and renders the answer plus clickable
citations. FastAPI serves it at GET /  (see api.py).
"""

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GitContext</title>
<style>
  :root {
    --bg: #ffffff; --fg: #1a1a1a; --muted: #6b7280; --line: #e5e7eb;
    --card: #f7f7f8; --accent: #4f46e5; --accent-fg: #ffffff; --code: #f3f4f6;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0f1115; --fg: #e6e6e6; --muted: #9aa0aa; --line: #262a33;
      --card: #171a21; --accent: #7c83ff; --accent-fg: #0f1115; --code: #1b1f27;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--fg);
         font: 15px/1.6 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
  .wrap { max-width: 820px; margin: 0 auto; padding: 24px 16px 64px; }
  h1 { font-size: 22px; margin: 8px 0 2px; }
  .sub { color: var(--muted); margin: 0 0 20px; font-size: 13px; }
  form { display: flex; gap: 8px; }
  input[type=text] { flex: 1; padding: 12px 14px; border: 1px solid var(--line);
    border-radius: 10px; background: var(--card); color: var(--fg); font-size: 15px; }
  select { padding: 12px 10px; border: 1px solid var(--line); border-radius: 10px;
    background: var(--card); color: var(--fg); font-size: 14px; }
  button { padding: 12px 18px; border: 0; border-radius: 10px; background: var(--accent);
    color: var(--accent-fg); font-weight: 600; cursor: pointer; font-size: 15px; }
  button:disabled { opacity: .55; cursor: default; }
  .answer { margin-top: 24px; padding: 16px 18px; background: var(--card);
    border: 1px solid var(--line); border-radius: 12px; white-space: pre-wrap; }
  .answer pre { background: var(--code); padding: 12px; border-radius: 8px; overflow-x: auto; }
  .backend { color: var(--muted); font-size: 12px; margin-bottom: 8px; }
  h3 { margin: 24px 0 8px; font-size: 13px; text-transform: uppercase;
       letter-spacing: .05em; color: var(--muted); }
  .cites { display: flex; flex-direction: column; gap: 6px; }
  .cite { padding: 8px 12px; background: var(--card); border: 1px solid var(--line);
    border-radius: 8px; font-size: 13px; display: flex; gap: 8px; flex-wrap: wrap; }
  .cite .path { font-family: ui-monospace, Menlo, Consolas, monospace; color: var(--accent); }
  .cite .meta { color: var(--muted); }
  .hint { color: var(--muted); font-size: 13px; margin-top: 16px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>GitContext</h1>
  <p class="sub">Ask a question about the indexed repository. Answers are grounded in retrieved code and cited.</p>

  <form id="f">
    <input id="q" type="text" placeholder="e.g. How does a user log in?" autocomplete="off" autofocus>
    <select id="src" title="Filter by source">
      <option value="">All</option>
      <option value="code">Code</option>
      <option value="docs">Docs</option>
      <option value="commit">Commits</option>
      <option value="pr">PRs</option>
    </select>
    <button id="btn" type="submit">Ask</button>
  </form>

  <div id="out"></div>
  <p class="hint">Tip: try an exact string like <code>cannot divide by zero</code> to see keyword (BM25) retrieval in action.</p>
</div>

<script>
const f = document.getElementById('f');
const q = document.getElementById('q');
const btn = document.getElementById('btn');
const out = document.getElementById('out');

function esc(s){ return s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

f.addEventListener('submit', async (e) => {
  e.preventDefault();
  const question = q.value.trim();
  if (!question) return;
  btn.disabled = true; btn.textContent = 'Thinking...';
  out.innerHTML = '<p class="hint">Retrieving and generating...</p>';
  const src = document.getElementById('src').value;
  const body = { question, top_k: 8 };
  if (src) body.filter = { source: src };
  try {
    const r = await fetch('/query', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)
    });
    const d = await r.json();
    // Split the "[backend] answer" prefix for display.
    let backend = '', answer = d.answer || '';
    const m = answer.match(/^\[([^\]]+)\]\s*/);
    if (m) { backend = m[1]; answer = answer.slice(m[0].length); }

    let html = '';
    if (backend) html += '<div class="backend">model: ' + esc(backend) + '</div>';
    html += '<div class="answer">' + esc(answer) + '</div>';
    html += '<h3>Citations</h3><div class="cites">';
    (d.citations || []).forEach(c => {
      const sym = c.symbol ? ':' + c.symbol : '';
      const bits = [];
      if (c.pr) bits.push('PR #' + c.pr);
      if (c.author) bits.push(c.author);
      if (c.last_modified) bits.push(c.last_modified);
      if (c.commit) bits.push(c.commit);
      html += '<div class="cite"><span class="path">' + esc((c.file_path||'?') + sym) + '</span>'
            + (bits.length ? '<span class="meta">' + esc(bits.join(' · ')) + '</span>' : '')
            + '</div>';
    });
    html += '</div>';
    out.innerHTML = html;
  } catch (err) {
    out.innerHTML = '<div class="answer">Error: ' + esc(String(err)) + '</div>';
  } finally {
    btn.disabled = false; btn.textContent = 'Ask';
  }
});
</script>
</body>
</html>"""
