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

  var verdictGrid = document.getElementById('verdict-grid');
  var verdictBox = document.getElementById('verdict-box');
  var verdictListEl = document.getElementById('verdict-list');
  var alsoCheckedSection = document.getElementById('also-checked-section');
  var modelBreakdownSection = document.getElementById('model-breakdown-section');
  var footnote = document.getElementById('detect-footnote');

  var prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

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

    var breakdown = fakeModelBreakdown(aiPct);
    Object.keys(breakdown).forEach(function (model) {
      var el = document.querySelector('[data-model="' + model + '"]');
      if (el) el.textContent = breakdown[model] + '%';
    });

    resultPanel.classList.remove('hidden');
    playRevealSequence(aiPct, split, arcColor);
  }

  // ---------------- Reveal sequence ----------------
  // 1. Verdict grid fades in; gauge + legend numbers count up from 0 together.
  // 2. "What we looked at" box fades in, then its bullets appear one by one.
  // 3. Remaining sections cascade in, ~500ms apart.

  var GAUGE_DURATION = prefersReducedMotion ? 0 : 900;
  var BULLET_STAGGER = prefersReducedMotion ? 0 : 300;
  var SECTION_STAGGER = prefersReducedMotion ? 0 : 500;

  var revealFadeSections = [verdictGrid, verdictBox, alsoCheckedSection, modelBreakdownSection, exportBtn, footnote];
  var bulletItems = verdictListEl ? Array.prototype.slice.call(verdictListEl.querySelectorAll('.bullet-item')) : [];

  function resetRevealState() {
    revealFadeSections.forEach(function (el) {
      if (el) el.classList.remove('visible');
    });
    bulletItems.forEach(function (li) { li.classList.remove('visible'); });
    scoreBig.textContent = '0%';
    legendAi.textContent = '0%';
    legendMixed.textContent = '0%';
    legendHuman.textContent = '0%';
    arc.setAttribute('stroke-dasharray', ARC_LENGTH);
    arc.setAttribute('stroke-dashoffset', ARC_LENGTH);
  }

  function easeOutCubic(t) { return 1 - Math.pow(1 - t, 3); }

  function animateGaugeAndLegend(aiPct, split, arcColor, duration, onDone) {
    arc.setAttribute('stroke', arcColor);
    if (duration <= 0) {
      arc.setAttribute('stroke-dashoffset', ARC_LENGTH * (1 - aiPct / 100));
      scoreBig.textContent = aiPct + '%';
      legendAi.textContent = aiPct + '%';
      legendMixed.textContent = split.mixedPct + '%';
      legendHuman.textContent = split.humanPct + '%';
      if (onDone) onDone();
      return;
    }
    var start = null;
    function step(timestamp) {
      if (start === null) start = timestamp;
      var elapsed = timestamp - start;
      var progress = Math.min(1, elapsed / duration);
      var eased = easeOutCubic(progress);

      var curAi = Math.round(aiPct * eased);
      var curMixed = Math.round(split.mixedPct * eased);
      var curHuman = Math.round(split.humanPct * eased);

      arc.setAttribute('stroke-dashoffset', ARC_LENGTH * (1 - (aiPct * eased) / 100));
      scoreBig.textContent = curAi + '%';
      legendAi.textContent = curAi + '%';
      legendMixed.textContent = curMixed + '%';
      legendHuman.textContent = curHuman + '%';

      if (progress < 1) {
        requestAnimationFrame(step);
      } else {
        // Snap to exact final values to avoid rounding drift.
        arc.setAttribute('stroke-dashoffset', ARC_LENGTH * (1 - aiPct / 100));
        scoreBig.textContent = aiPct + '%';
        legendAi.textContent = aiPct + '%';
        legendMixed.textContent = split.mixedPct + '%';
        legendHuman.textContent = split.humanPct + '%';
        if (onDone) onDone();
      }
    }
    requestAnimationFrame(step);
  }

  function revealBulletsSequentially(items, stagger, onDone) {
    if (!items.length) { if (onDone) onDone(); return; }
    items.forEach(function (li, i) {
      setTimeout(function () {
        li.classList.add('visible');
        if (i === items.length - 1 && onDone) {
          setTimeout(onDone, 400);
        }
      }, i * stagger);
    });
  }

  function playRevealSequence(aiPct, split, arcColor) {
    resetRevealState();

    // Step 1 — verdict grid fades in; gauge + legend count up together.
    if (verdictGrid) verdictGrid.classList.add('visible');
    animateGaugeAndLegend(aiPct, split, arcColor, GAUGE_DURATION, function () {

      // Step 2 — "what we looked at" box fades in, then bullets cascade.
      if (verdictBox) verdictBox.classList.add('visible');
      setTimeout(function () {
        revealBulletsSequentially(bulletItems, BULLET_STAGGER, function () {

          // Step 3 — remaining sections cascade in, ~500ms apart.
          var rest = [alsoCheckedSection, modelBreakdownSection, exportBtn, footnote];
          rest.forEach(function (el, i) {
            setTimeout(function () {
              if (el) el.classList.add('visible');
            }, i * SECTION_STAGGER);
          });
        });
      }, prefersReducedMotion ? 0 : 150);
    });
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
