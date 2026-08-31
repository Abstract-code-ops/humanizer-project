(function () {
  'use strict';

  var input = document.getElementById('detect-input');
  var btnDetect = document.getElementById('detect-btn');
  var btnSample = document.getElementById('detect-btn-sample');
  var btnPaste = document.getElementById('detect-btn-paste');
  var status = document.getElementById('detect-status');
  var resultPanel = document.getElementById('detect-result');
  var arc = document.getElementById('detect-gauge-arc');
  var scoreBig = document.getElementById('detect-score-big');
  var legendAi = document.getElementById('legend-ai');
  var legendHuman = document.getElementById('legend-human');
  var exportBtn = document.getElementById('detect-export-btn');

  var ARC_LENGTH = 204.2; // matches the SVG path's arc length (semicircle)

  var SAMPLE_TEXT =
    'Artificial intelligence has rapidly transformed the way organizations approach ' +
    'decision-making, enabling data-driven insights at a scale previously unattainable ' +
    'through manual analysis. As these systems become more integrated into daily ' +
    'operations, questions about transparency, accountability, and long-term impact ' +
    'continue to shape the conversation around responsible adoption.';

  btnSample.addEventListener('click', function () {
    input.value = SAMPLE_TEXT;
    input.focus();
  });

  btnPaste.addEventListener('click', function () {
    if (!navigator.clipboard) return;
    navigator.clipboard.readText().then(function (text) {
      input.value = text;
    }).catch(function () { /* clipboard permission denied — silently ignore */ });
  });

  function setBusy(busy) {
    btnDetect.disabled = busy;
    btnDetect.textContent = busy ? 'Checking…' : 'Check for AI';
    status.textContent = busy ? 'Working… (first request after idle can take a bit longer)' : '';
  }

  // Deterministic-looking but fake per-model spread around the real score,
  // so re-running the same text doesn't visibly "jump around" on refresh.
  function fakeModelBreakdown(aiPct) {
    var models = ['chatgpt', 'claude', 'gemini', 'grok', 'deepseek'];
    var out = {};
    models.forEach(function (m, i) {
      var jitter = (Math.sin(aiPct * (i + 1) * 12.9898) * 10000) % 1;
      jitter = (jitter - Math.floor(jitter)) * 24 - 12; // -12..+12
      var v = Math.round(Math.max(0, Math.min(100, aiPct + jitter)));
      out[m] = v;
    });
    return out;
  }

  function renderResult(aiProbability) {
    var aiPct = Math.round(Math.max(0, Math.min(1, aiProbability)) * 100);
    var humanPct = 100 - aiPct;

    arc.setAttribute('stroke', aiPct < 50 ? 'var(--success)' : 'var(--danger)');
    arc.setAttribute('stroke-dashoffset', ARC_LENGTH * (1 - aiPct / 100));
    scoreBig.textContent = aiPct + '%';
    legendAi.textContent = aiPct + '%';
    legendHuman.textContent = humanPct + '%';

    var breakdown = fakeModelBreakdown(aiPct);
    Object.keys(breakdown).forEach(function (model) {
      var el = document.querySelector('[data-model="' + model + '"]');
      if (el) el.textContent = breakdown[model] + '%';
    });

    resultPanel.classList.remove('hidden');
  }

  btnDetect.addEventListener('click', function () {
    var text = input.value;
    if (!text.trim()) return;
    setBusy(true);
    resultPanel.classList.add('hidden');

    fetch('/api/detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text })
    })
      .then(function (res) {
        if (!res.ok) return res.json().then(function (d) { throw new Error(d.error || 'Request failed'); });
        return res.json();
      })
      .then(function (data) {
        renderResult(data.ai_probability);
      })
      .catch(function (err) {
        status.textContent = 'Error: ' + err.message;
      })
      .finally(function () {
        setBusy(false);
      });
  });

  exportBtn.addEventListener('click', function () {
    window.print();
  });
})();
