// ── DOM References ──
const dropZone    = document.getElementById('dropZone');
const fileInput   = document.getElementById('fileInput');
const fileSelected = document.getElementById('fileSelected');
const btnAnalyze  = document.getElementById('btnAnalyze');

let selectedFile = null;

// ── Drag & Drop / File Selection ──
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});

// ── Set Selected File ──
function setFile(file) {
  selectedFile = file;
  fileSelected.textContent = `▸ ${file.name}  (${(file.size / 1024).toFixed(1)} KB)`;
  fileSelected.classList.add('visible');
  btnAnalyze.classList.add('visible');
}

// ── Analyze File ──
function analyzeFile() {
  if (!selectedFile) return;

  const formData = new FormData();
  formData.append('file', selectedFile);

  document.getElementById('uploadSection').style.display = 'none';
  document.getElementById('loader').classList.add('visible');

  fetch('/analyze', { method: 'POST', body: formData })
    .then(r => r.json())
    .then(data => {
      document.getElementById('loader').classList.remove('visible');
      if (data.error) {
        alert('Error: ' + data.error);
        resetApp();
        return;
      }
      renderResults(data);
    })
    .catch(err => {
      document.getElementById('loader').classList.remove('visible');
      alert('Error: ' + err);
      resetApp();
    });
}

// ── Render Results ──
function renderResults(data) {
  const { profile, charts, top_correlations, insights } = data;

  // Stats cards
  const statsGrid = document.getElementById('statsGrid');
  const missingPct = profile.shape.rows > 0
    ? ((profile.missing_total / (profile.shape.rows * profile.shape.cols)) * 100).toFixed(1)
    : 0;
  statsGrid.innerHTML = `
    <div class="stat-card"><div class="stat-value">${profile.shape.rows.toLocaleString()}</div><div class="stat-label">Rows</div></div>
    <div class="stat-card"><div class="stat-value">${profile.shape.cols}</div><div class="stat-label">Columns</div></div>
    <div class="stat-card"><div class="stat-value">${missingPct}%</div><div class="stat-label">Missing</div></div>
    <div class="stat-card"><div class="stat-value">${profile.duplicate_rows}</div><div class="stat-label">Duplicates</div></div>
  `;

  // Insights
  const insightsBox = document.getElementById('insightsBox');
  insightsBox.innerHTML = insights.length === 0
    ? '<div class="insight-item">✅ Dataset looks clean — no major issues detected.</div>'
    : insights.map(i => `<div class="insight-item">${i}</div>`).join('');

  // Column profile table
  const tbody = document.getElementById('colTableBody');
  tbody.innerHTML = profile.columns.map(col => {
    const mp = col.missing_pct;
    const mpClass = mp > 20 ? 'missing-high' : mp > 5 ? 'missing-med' : 'missing-ok';
    const stats = col.stats || {};
    return `<tr>
      <td>${col.name}</td>
      <td><span class="dtype-tag">${col.dtype}</span></td>
      <td class="${mpClass}">${col.missing} (${mp}%)</td>
      <td>${col.unique}</td>
      <td>${stats.mean !== undefined ? stats.mean : '—'}</td>
      <td>${stats.std  !== undefined ? stats.std  : '—'}</td>
      <td>${stats.skew !== undefined ? stats.skew : '—'}</td>
    </tr>`;
  }).join('');

  // Correlations
  const corrList = document.getElementById('corrList');
  corrList.innerHTML = top_correlations.length === 0
    ? '<div style="color:var(--muted);font-family:var(--mono);font-size:12px;padding:12px 0">Not enough numeric columns for correlation analysis.</div>'
    : top_correlations.map(c => `
        <div class="corr-item">
          <div class="corr-cols"><strong>${c.col1}</strong> ↔ <strong>${c.col2}</strong></div>
          <div class="corr-bar-wrap"><div class="corr-bar" style="width:${c.value * 100}%"></div></div>
          <div class="corr-val">${c.value}</div>
        </div>
      `).join('');

  // Charts
  const chartsGrid = document.getElementById('chartsGrid');
  chartsGrid.innerHTML = '';
  const chartLabels = {
    distributions: 'Feature Distributions',
    correlation:   'Correlation Heatmap',
    missing:       'Missing Values',
    clustering:    'K-Means Clusters (PCA)',
  };
  for (const [key, b64] of Object.entries(charts)) {
    chartsGrid.innerHTML += `
      <div class="chart-card">
        <div class="chart-title">${chartLabels[key] || key}</div>
        <img src="data:image/png;base64,${b64}" alt="${key}">
      </div>`;
  }

  document.getElementById('results').classList.add('visible');
}

// ── Reset ──
function resetApp() {
  selectedFile = null;
  fileInput.value = '';
  fileSelected.classList.remove('visible');
  btnAnalyze.classList.remove('visible');
  document.getElementById('uploadSection').style.display = 'block';
  document.getElementById('results').classList.remove('visible');
  document.getElementById('loader').classList.remove('visible');
}