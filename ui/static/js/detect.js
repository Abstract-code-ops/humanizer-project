(function () {
  'use strict';

  var input = document.getElementById('detect-input');
  var btnDetect = document.getElementById('detect-btn');
  var btnSample = document.getElementById('detect-btn-sample');
  var btnPaste = document.getElementById('detect-btn-paste');
  var btnUpload = document.getElementById('detect-btn-upload');
  var fileInput = document.getElementById('detect-file-input');
  var wordCounter = document.getElementById('detect-word-counter');
  var status = document.getElementById('detect-status');
  var resultPanel = document.getElementById('detect-result');
  var arc = document.getElementById('detect-gauge-arc');
  var scoreBig = document.getElementById('detect-score-big');
  var legendAi = document.getElementById('legend-ai');
  var legendMixed = document.getElementById('legend-mixed');
  var legendHuman = document.getElementById('legend-human');
  var exportBtn = document.getElementById('detect-export-btn');
  var alsoCheckedToggle = document.getElementById('also-checked-toggle');
  var detectorList = document.getElementById('detector-list');

  // Full-circle gauge: circumference = 2 * PI * r (r = 60, matches the SVG markup)
  var ARC_LENGTH = 2 * Math.PI * 60;

  var SAMPLE_TEXT =
    'Artificial intelligence has rapidly transformed the way organizations approach ' +
    'decision-making, enabling data-driven insights at a scale previously unattainable ' +
    'through manual analysis. As these systems become more integrated into daily ' +
    'operations, questions about transparency, accountability, and long-term impact ' +
    'continue to shape the conversation around responsible adoption.';

  function wordCount(text) {
    var trimmed = text.trim();
    return trimmed.length ? trimmed.split(/\s+/).length : 0;
  }

  function updateWordCounter() {
    if (!wordCounter) return;
    wordCounter.textContent = wordCount(input.value) + ' words';
  }
  input.addEventListener('input', updateWordCounter);

  btnSample.addEventListener('click', function () {
    input.value = SAMPLE_TEXT;
    updateWordCounter();
    input.focus();
  });

  btnPaste.addEventListener('click', function () {
    if (!navigator.clipboard) return;
    navigator.clipboard.readText().then(function (text) {
      input.value = text;
      updateWordCounter();
    }).catch(function () { /* clipboard permission denied — silently ignore */ });
  });

  // ---------------- Upload file ----------------

  if (btnUpload && fileInput) {
    btnUpload.addEventListener('click', function () {
      fileInput.click();
    });

    fileInput.addEventListener('change', function () {
      var file = fileInput.files && fileInput.files[0];
      if (!file) return;
      var reader = new FileReader();
      reader.onload = function (e) {
        input.value = String(e.target.result || '');
        updateWordCounter();
      };
      reader.onerror = function () {
        status.textContent = 'Could not read that file — try pasting the text instead.';
      };
      reader.readAsText(file);
      fileInput.value = '';
    });
  }

  // ---------------- Also-checked expand/collapse ----------------

  if (alsoCheckedToggle && detectorList) {
    alsoCheckedToggle.addEventListener('click', function () {
      var isOpen = detectorList.classList.toggle('expanded');
      alsoCheckedToggle.classList.toggle('open', isOpen);
      alsoCheckedToggle.setAttribute('aria-expanded', String(isOpen));
      alsoCheckedToggle.setAttribute('aria-label', isOpen ? 'Show fewer detectors' : 'Show all detectors');
    });
  }

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

  // Splits the non-AI remainder into "mixed" and "human-written" so the
  // three-way legend always sums to 100. Purely illustrative — replace
  // with a real mixed-classification signal once the backend supports it.
  function splitRemainder(aiPct) {
    var remainder = 100 - aiPct;
    var mixedPct = Math.round(remainder * 0.32);
    var humanPct = remainder - mixedPct;
    return { mixedPct: mixedPct, humanPct: humanPct };
  }

  function renderResult(aiProbability) {
    var aiPct = Math.round(Math.max(0, Math.min(1, aiProbability)) * 100);
    var split = splitRemainder(aiPct);

    var arcColor = 'var(--success)';
    if (aiPct >= 60) arcColor = 'var(--danger)';
    else if (aiPct >= 30) arcColor = 'var(--mixed)';

    arc.setAttribute('stroke', arcColor);
    arc.setAttribute('stroke-dasharray', ARC_LENGTH);
    arc.setAttribute('stroke-dashoffset', ARC_LENGTH * (1 - aiPct / 100));
    scoreBig.textContent = aiPct + '%';
    legendAi.textContent = aiPct + '%';
    legendMixed.textContent = split.mixedPct + '%';
    legendHuman.textContent = split.humanPct + '%';

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

  updateWordCounter();
})();
