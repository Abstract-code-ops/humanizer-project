(function () {
  'use strict';

  var WORD_LIMIT = window.WORD_CAP || 200;
  var STAGE2_MIN_MS = 2500;
  var STAGE2_MAX_MS = 4000;
  var SLOW_HINT_MS = 15000;

  var input = document.getElementById('input-text');
  var counter = document.getElementById('word-counter');
  var btnHumanize = document.getElementById('btn-humanize');
  var btnPaste = document.getElementById('btn-paste');
  var btnClear = document.getElementById('btn-clear');
  var btnSample = document.getElementById('btn-sample');
  var btnUpload = document.getElementById('btn-upload');
  var fileInput = document.getElementById('file-input');
  var btnCopy = document.getElementById('btn-copy');
  var resultPanel = document.getElementById('result-panel');
  var themeToggle = document.getElementById('theme-toggle');

  var backdrop = document.getElementById('modal-backdrop');
  var modal = document.getElementById('modal');
  var spinner = document.getElementById('modal-spinner');
  var modalLabel = document.getElementById('modal-label');
  var modalCaption = document.getElementById('modal-caption');
  var detectorRow = document.getElementById('detector-row');
  var simulatedTag = document.getElementById('modal-simulated-tag');
  var modalError = document.getElementById('modal-error');
  var detectorChips = Array.prototype.slice.call(document.querySelectorAll('.detector-chip'));

  var prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // Theme toggle now lives in theme.js (shared across all pages via _base.html)

  // ---------------- Word counter ----------------

  function wordCount(text) {
    var trimmed = text.trim();
    return trimmed.length ? trimmed.split(/\s+/).length : 0;
  }

  function updateCounter() {
    var n = wordCount(input.value);
    var overLimit = n > WORD_LIMIT;
    counter.textContent = n + ' / ' + WORD_LIMIT + ' words';
    counter.classList.toggle('over-limit', overLimit);
    counter.title = overLimit
      ? 'Input exceeds the limit. Shorten it before sending.'
      : 'Longer inputs are truncated to the first ' + WORD_LIMIT + ' words before processing.';
    btnHumanize.disabled = input.value.trim().length === 0 || overLimit;
  }

  input.addEventListener('input', updateCounter);

  // ---------------- Utility buttons ----------------

  btnPaste.addEventListener('click', function () {
    if (!navigator.clipboard) return;
    navigator.clipboard.readText().then(function (text) {
      input.value = text;
      updateCounter();
    }).catch(function () { /* clipboard permission denied — silently ignore */ });
  });

  btnClear.addEventListener('click', function () {
    input.value = '';
    updateCounter();
    resultPanel.classList.remove('visible');
    input.focus();
  });

  var SAMPLE_TEXTS = [
    'Artificial intelligence has rapidly transformed the way organizations approach ' +
    'decision-making, enabling data-driven insights at a scale previously unattainable ' +
    'through manual analysis. As these systems become more integrated into daily ' +
    'operations, questions about transparency, accountability, and long-term impact ' +
    'continue to shape the conversation around responsible adoption.'
  ];

  btnSample.addEventListener('click', function () {
    var sample = SAMPLE_TEXTS[Math.floor(Math.random() * SAMPLE_TEXTS.length)];
    input.value = sample;
    updateCounter();
    input.focus();
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
        updateCounter();
      };
      reader.readAsText(file);
      fileInput.value = '';
    });
  }

  function truncateToLimit(text) {
    var words = text.trim().split(/\s+/);
    if (words.length <= WORD_LIMIT) return text;
    return words.slice(0, WORD_LIMIT).join(' ');
  }

  // ---------------- Copy result ----------------

  btnCopy.addEventListener('click', function () {
    var text = document.getElementById('rewritten-text').textContent;
    if (!navigator.clipboard) return;
    navigator.clipboard.writeText(text).then(function () {
      var original = btnCopy.textContent;
      btnCopy.textContent = 'Copied';
      setTimeout(function () { btnCopy.textContent = original; }, 1500);
    });
  });

  // ---------------- Gauge ----------------

  function renderGauge(score) {
    // score expected 0–1, low = human-like (green), high = AI-like (red)
    var clamped = Math.max(0, Math.min(1, score));
    var circumference = 2 * Math.PI * 50;
    var arc = document.getElementById('gauge-arc');
    var valueLabel = document.getElementById('gauge-value');
    arc.setAttribute('stroke-dasharray', circumference);
    arc.setAttribute('stroke-dashoffset', circumference * (1 - clamped));
    arc.setAttribute('stroke', clamped < 0.5 ? 'var(--success)' : 'var(--danger)');
    valueLabel.textContent = Math.round(clamped * 100) + '%';
  }

  function readabilityLabel(fleschScore) {
    if (fleschScore >= 70) return 'Easy';
    if (fleschScore >= 50) return 'Standard';
    return 'Hard';
  }

  // ---------------- Modal stage machine ----------------

  var lastFocused = null;
  var slowHintTimer = null;

  function trapFocus(e) {
    if (e.key !== 'Tab') return;
    var focusable = modal.querySelectorAll('button');
    if (!focusable.length) return;
    var first = focusable[0];
    var last = focusable[focusable.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  }

  function openModal() {
    lastFocused = document.activeElement;
    backdrop.classList.add('open');
    document.addEventListener('keydown', trapFocus);
    setStageHumanizing();
  }

  function closeModal() {
    backdrop.classList.remove('open');
    document.removeEventListener('keydown', trapFocus);
    clearTimeout(slowHintTimer);
    if (lastFocused) lastFocused.focus();
  }

  function setStageHumanizing() {
    spinner.hidden = false;
    modalLabel.textContent = 'Humanizing…';
    modalCaption.textContent = '';
    detectorRow.hidden = true;
    simulatedTag.hidden = true;
    modalError.hidden = true;
    detectorChips.forEach(function (chip) {
      chip.classList.remove('active', 'done');
    });
    clearTimeout(slowHintTimer);
    slowHintTimer = setTimeout(function () {
      modalCaption.textContent = 'Still working — first request after idle can take a bit longer.';
    }, SLOW_HINT_MS);
  }

  function setStageTesting(onComplete) {
    clearTimeout(slowHintTimer);
    spinner.hidden = true;
    modalLabel.textContent = 'Testing against AI detectors';
    modalCaption.textContent = '';
    detectorRow.hidden = false;
    simulatedTag.hidden = false;

    if (prefersReducedMotion) {
      modalCaption.textContent = 'Checking result…';
      setTimeout(onComplete, 1200);
      return;
    }

    var staggerMs = 250;
    detectorChips.forEach(function (chip, i) {
      setTimeout(function () { chip.classList.add('active'); }, i * staggerMs);
      setTimeout(function () {
        chip.classList.remove('active');
        chip.classList.add('done');
      }, i * staggerMs + 300);
    });

    var totalMs = detectorChips.length * staggerMs + 600;
    var waitMs = Math.max(STAGE2_MIN_MS, Math.min(STAGE2_MAX_MS, totalMs));
    setTimeout(onComplete, waitMs);
  }

  function setStageError(message) {
    clearTimeout(slowHintTimer);
    spinner.hidden = true;
    detectorRow.hidden = true;
    simulatedTag.hidden = true;
    modalLabel.textContent = 'Something went wrong';
    modalCaption.textContent = '';
    modalError.hidden = false;
    modalError.textContent = message;
    setTimeout(closeModal, 3000);
  }

  // ---------------- Humanize flow ----------------

  btnHumanize.addEventListener('click', function () {
    var text = input.value.trim();
    if (!text) return;
    if (wordCount(text) > WORD_LIMIT) {
      counter.classList.add('over-limit');
      counter.title = 'Input exceeds the limit. Shorten it before sending.';
      setStageError('Input exceeds the ' + WORD_LIMIT + '-word limit. Please shorten it before sending.');
      return;
    }

    openModal();

    fetch('/api/humanize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text })
    })
      .then(function (res) {
        if (!res.ok) throw new Error('Request failed (' + res.status + ')');
        return res.json();
      })
      .then(function (data) {
        setStageTesting(function () {
          closeModal();
          renderGauge(data.proxy_score);
          document.getElementById('stat-similarity').textContent =
            Math.round(data.similarity * 100) + '%';
          document.getElementById('stat-readability').textContent =
            data.readability + ' (' + readabilityLabel(data.readability) + ')';
          document.getElementById('rewritten-text').textContent = data.humanized;
          resultPanel.classList.add('visible');
          resultPanel.scrollIntoView({ behavior: prefersReducedMotion ? 'auto' : 'smooth', block: 'start' });
        });
      })
      .catch(function (err) {
        setStageError('The request didn\u2019t go through. ' + err.message + '. Try again in a moment.');
      });
  });

  // ---------------- Init ----------------

  updateCounter();
})();
