// ===== رنگ‌بندی ثابت برای هر corruption (استفاده در نقشه و نمودارها) =====
const CORRUPTION_COLORS = {
  clean: "#2dd4a7", blur: "#4fa3d1", jpeg: "#e0b15c", noise: "#e05c5c",
  resize: "#a06ce0", crop: "#5ce0c4", lowlight: "#7a7ae0", fog: "#c9a24b",
};
function colorFor(type) { return CORRUPTION_COLORS[type] || "#9db3ac"; }

// ===== تعویض تب‌ها =====
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
  });
});

// ===== انتخاب فایل =====
const fileInput = document.getElementById('imgInput');
const fileDrop = document.getElementById('fileDrop');
fileDrop.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) document.getElementById('fileLabel').textContent = '✅ ' + fileInput.files[0].name;
});

// ===== تعویض حالت (تکی / کامل) =====
let currentMode = 'single';
document.querySelectorAll('.mode-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentMode = btn.dataset.mode;
    document.getElementById('manualControls').style.display = currentMode === 'single' ? 'flex' : 'none';
  });
});

// ===== اجرای مدل =====
let mapSingle, markerSingle, trueMarkerSingle;
let mapAll;

document.getElementById('runBtn').addEventListener('click', async () => {
  if (!fileInput.files[0]) { alert('اول یک عکس انتخاب کن'); return; }

  document.getElementById('loadingMsg').style.display = 'flex';
  document.getElementById('singleResultPanel').style.display = 'none';
  document.getElementById('allResultPanel').style.display = 'none';

  const formData = new FormData();
  formData.append('image', fileInput.files[0]);
  formData.append('mode', currentMode);
  if (currentMode === 'single') {
    formData.append('corruption_type', document.getElementById('corruptionSelect').value);
    formData.append('severity', document.getElementById('severitySelect').value);
  }

  const res = await fetch('/api/predict', { method: 'POST', body: formData });
  const data = await res.json();
  document.getElementById('loadingMsg').style.display = 'none';

  if (currentMode === 'single') renderSingleResult(data);
  else renderAllResult(data);
});

function gtBannerHTML(data) {
  if (data.is_known_test_image) {
    return `<div class="gt-banner known">✅ این تصویر در subset تست OSV-5M شناسایی شد — مکان واقعی: ${data.true_lat.toFixed(3)}, ${data.true_lon.toFixed(3)} (${data.true_country})</div>`;
  }
  return `<div class="gt-banner unknown">ℹ️ این تصویر خارج از دیتاست OSV-5M است — مکان واقعی موجود نیست، فقط پیش‌بینی مدل نمایش داده می‌شود.</div>`;
}

function renderSingleResult(data) {
  const r = data.results[0];
  document.getElementById('singleResultPanel').style.display = 'block';
  document.getElementById('gtBanner').innerHTML = gtBannerHTML(data);
  document.getElementById('processedImg').src = 'data:image/jpeg;base64,' + r.thumbnail_b64;

  let metricsHtml = `
    <div class="metric-card"><span>مختصات پیش‌بینی‌شده</span><b>${r.pred_lat.toFixed(3)}, ${r.pred_lon.toFixed(3)}</b></div>
    <div class="metric-card"><span>Top-1 Score</span><b>${r.top1_score.toFixed(3)}</b></div>
    <div class="metric-card"><span>Margin</span><b>${r.margin.toFixed(3)}</b></div>
    <div class="metric-card"><span>Entropy</span><b>${r.entropy.toFixed(3)}</b></div>
    <div class="metric-card"><span>احتمال درستی</span><b>${(r.p_correct*100).toFixed(1)}%</b></div>
    <div class="metric-card ${r.decision}"><span>تصمیم Abstention</span><b>${r.decision === 'accept' ? '✅ پذیرفته شد' : '⚠️ رد شد'}</b></div>
  `;
  if (data.is_known_test_image) {
    metricsHtml += `<div class="metric-card"><span>خطای مکانی واقعی</span><b>${r.error_km.toFixed(1)} km</b></div>`;
  }
  document.getElementById('singleMetrics').innerHTML = metricsHtml;

  if (mapSingle) mapSingle.remove();
  mapSingle = L.map('mapSingle').setView([r.pred_lat, r.pred_lon], data.is_known_test_image ? 5 : 4);
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '&copy; OpenStreetMap contributors'
}).addTo(mapSingle);   // و همینطور برای mapAll
  L.marker([r.pred_lat, r.pred_lon]).addTo(mapSingle).bindPopup('پیش‌بینی مدل').openPopup();
  if (data.is_known_test_image) {
    L.circleMarker([data.true_lat, data.true_lon], {radius: 9, color: '#2dd4a7', fillColor:'#2dd4a7', fillOpacity:0.9})
      .addTo(mapSingle).bindPopup('مکان واقعی');
    const bounds = L.latLngBounds([[r.pred_lat, r.pred_lon],[data.true_lat, data.true_lon]]);
    mapSingle.fitBounds(bounds, {padding:[40,40]});
  }
}

function renderAllResult(data) {
  document.getElementById('allResultPanel').style.display = 'block';
  document.getElementById('gtBannerAll').innerHTML = gtBannerHTML(data);

  // ---- نقشه ----
  if (mapAll) mapAll.remove();
  mapAll = L.map('mapAll').setView([20, 0], 2);
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '&copy; OpenStreetMap contributors'
}).addTo(mapAll);   // و همینطور برای mapAll

  const allPoints = [];
  data.results.forEach(r => {
    const label = r.corruption_type === 'clean' ? 'Clean' : `${r.corruption_type} (sev ${r.severity})`;
    L.circleMarker([r.pred_lat, r.pred_lon], {radius: 7, color: colorFor(r.corruption_type), fillOpacity: 0.85})
      .addTo(mapAll).bindPopup(`<b>${label}</b><br>Entropy: ${r.entropy.toFixed(2)}<br>تصمیم: ${r.decision}`);
    allPoints.push([r.pred_lat, r.pred_lon]);
  });
  if (data.is_known_test_image) {
    L.circleMarker([data.true_lat, data.true_lon], {radius: 11, color: '#fff', fillColor:'#2dd4a7', fillOpacity:1, weight:2})
      .addTo(mapAll).bindPopup('⭐ مکان واقعی');
    allPoints.push([data.true_lat, data.true_lon]);
  }
  if (allPoints.length) mapAll.fitBounds(allPoints, {padding:[30,30]});

  // ---- جدول ----
  const tbody = document.getElementById('allResultsBody');
  tbody.innerHTML = '';
  data.results.forEach(r => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td style="color:${colorFor(r.corruption_type)}">${r.corruption_type}</td>
      <td>${r.severity}</td>
      <td><img src="data:image/jpeg;base64,${r.thumbnail_b64}"/></td>
      <td>${r.pred_lat.toFixed(2)}, ${r.pred_lon.toFixed(2)}</td>
      <td>${data.is_known_test_image ? r.error_km.toFixed(0) : '—'}</td>
      <td>${r.entropy.toFixed(2)}</td>
      <td>${r.margin.toFixed(2)}</td>
      <td>${r.top1_score.toFixed(2)}</td>
      <td class="${r.decision === 'accept' ? 'tag-accept' : 'tag-reject'}">${r.decision === 'accept' ? 'پذیرفته' : 'رد شده'}</td>
    `;
    tbody.appendChild(tr);
  });

  // ---- نمودارهای این تصویر خاص ----
  const labels = data.results.map(r => r.corruption_type === 'clean' ? 'clean' : `${r.corruption_type}-${r.severity}`);
  if (window._allErrorChart) window._allErrorChart.destroy();
  if (data.is_known_test_image) {
    window._allErrorChart = new Chart(document.getElementById('allErrorChart'), {
      type: 'bar',
      data: { labels, datasets: [{ label: 'خطا (km)', data: data.results.map(r => r.error_km), backgroundColor: labels.map((_,i)=>colorFor(data.results[i].corruption_type)) }] },
      options: chartOpts('خطای مکانی برای این تصویر')
    });
  }
  if (window._allEntropyChart) window._allEntropyChart.destroy();
  window._allEntropyChart = new Chart(document.getElementById('allEntropyChart'), {
    type: 'bar',
    data: { labels, datasets: [{ label: 'Entropy', data: data.results.map(r => r.entropy), backgroundColor: labels.map((_,i)=>colorFor(data.results[i].corruption_type)) }] },
    options: chartOpts('عدم‌قطعیت مدل (Entropy) برای این تصویر')
  });
}

// ===== تنظیمات مشترک نمودارها (سازگار با تم دارک) =====
function chartOpts(title) {
  return {
    responsive: true,
    plugins: {
      legend: { display: false },
      title: { display: !!title, text: title, color: '#e8f1ee', font: { family: 'Vazirmatn' } }
    },
    scales: {
      x: { ticks: { color: '#9db3ac', font: { family: 'Vazirmatn' } }, grid: { color: 'rgba(255,255,255,0.05)' } },
      y: { ticks: { color: '#9db3ac' }, grid: { color: 'rgba(255,255,255,0.05)' } }
    }
  };
}
Chart.defaults.font.family = 'Vazirmatn';

// ===================== نمودارهای تب "افت عملکرد" =====================
fetch('/api/summary').then(r => r.json()).then(data => {
  const labels = data.map(d => d.corruption_type + (d.severity !== 'none' ? '-' + d.severity : ''));
  new Chart(document.getElementById('corruptionAccChart'), {
    type: 'bar',
    data: { labels, datasets: [{ label: 'دقت کشور (%)', data: data.map(d => d.acc_country_750km), backgroundColor: '#10a37f' }] },
    options: chartOpts()
  });
  new Chart(document.getElementById('corruptionErrorChart'), {
    type: 'bar',
    data: { labels, datasets: [{ label: 'میانگین خطا (km)', data: data.map(d => d.mean_error_km), backgroundColor: '#1b6ca8' }] },
    options: chartOpts()
  });
});

fetch('/api/corruption_impact').then(r => r.json()).then(data => {
  const labels = data.map(d => d.corruption_type + (d.severity !== 'none' ? '-' + d.severity : ''));
  new Chart(document.getElementById('rejectRateChart'), {
    type: 'bar',
    data: { labels, datasets: [{ label: 'نرخ Reject (%)', data: data.map(d => d.reject_rate_pct), backgroundColor: '#e05c5c' }] },
    options: chartOpts()
  });

  // نمودار "قبل و بعد" (تب abstention)
  new Chart(document.getElementById('beforeAfterChart'), {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label: 'قبل از Abstention', data: data.map(d => d.country_acc_before_pct), backgroundColor: '#4fa3d1' },
        { label: 'بعد از Abstention', data: data.map(d => d.country_acc_after_pct), backgroundColor: '#2dd4a7' },
      ]
    },
    options: { ...chartOpts(), plugins: { legend: { display: true, labels: { color: '#e8f1ee' } } } }
  });
});

// ===================== نمودارهای تب "تحلیل اطمینان" =====================
fetch('/api/entropy_histogram').then(r => r.json()).then(data => {
  const labels = data.bin_edges.slice(0,-1).map((v,i) => v.toFixed(1));
  new Chart(document.getElementById('entropyHistChart'), {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label: 'Clean', data: data.clean_counts, backgroundColor: 'rgba(45,212,167,0.6)' },
        { label: 'Corrupted (نرمال‌شده)', data: data.corrupted_counts, backgroundColor: 'rgba(224,92,92,0.6)' },
      ]
    },
    options: { ...chartOpts(), plugins: { legend: { display: true, labels: { color: '#e8f1ee' } } } }
  });
});

fetch('/api/error_vs_entropy_decile').then(r => r.json()).then(data => {
  new Chart(document.getElementById('decileChart'), {
    type: 'line',
    data: {
      labels: data.map((d,i) => 'D' + (i+1)),
      datasets: [
        { label: 'دقت کشور (%)', data: data.map(d => d.country_acc_pct), borderColor: '#2dd4a7', yAxisID: 'y' },
        { label: 'میانگین خطا (km)', data: data.map(d => d.mean_error_km), borderColor: '#e05c5c', yAxisID: 'y1' },
      ]
    },
    options: {
      ...chartOpts(),
      plugins: { legend: { display: true, labels: { color: '#e8f1ee' } } },
      scales: {
        x: chartOpts().scales.x,
        y: { position: 'left', ticks: { color: '#2dd4a7' } },
        y1: { position: 'right', ticks: { color: '#e05c5c' }, grid: { drawOnChartArea: false } },
      }
    }
  });
});

fetch('/api/feature_importance').then(r => r.json()).then(data => {
  new Chart(document.getElementById('featureImportanceChart'), {
    type: 'bar',
    data: { labels: data.map(d => d.feature), datasets: [{ label: 'اهمیت (%)', data: data.map(d => d.abs_importance_pct), backgroundColor: ['#4fa3d1','#c9a24b','#e05c5c'] }] },
    options: { ...chartOpts(), indexAxis: 'y' }
  });
});

// ===================== نمودار Risk-Coverage =====================
fetch('/api/comparison').then(r => r.json()).then(data => {
  const clean = data.filter(d => d.subset === 'clean' && d.method === 'logistic_regression');
  const corrupted = data.filter(d => d.subset === 'corrupted' && d.method === 'logistic_regression');
  new Chart(document.getElementById('riskCoverageChart'), {
    type: 'line',
    data: {
      labels: clean.map(d => d.actual_coverage_pct.toFixed(0) + '%'),
      datasets: [
        { label: 'Clean', data: clean.map(d => d.selective_country_acc_pct), borderColor: '#2dd4a7', tension: 0.3 },
        { label: 'Corrupted', data: corrupted.map(d => d.selective_country_acc_pct), borderColor: '#e05c5c', tension: 0.3 }
      ]
    },
    options: { ...chartOpts(), plugins: { legend: { display: true, labels: { color: '#e8f1ee' } } } }
  });
});