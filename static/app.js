/* InkVerify front-end: theme switch, tabs, drag-drop + previews, viewer, summary export. */
(function () {
  var root = document.documentElement;
  var $ = function (s, c) { return (c || document).querySelector(s); };
  var $$ = function (s, c) { return Array.prototype.slice.call((c || document).querySelectorAll(s)); };

  // ----- theme (left/right switch) -----
  var sw = $('#theme-switch');
  function current() {
    return root.dataset.theme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  }
  if (sw) {
    sw.checked = current() === 'dark';
    sw.addEventListener('change', function () {
      var next = sw.checked ? 'dark' : 'light';
      root.dataset.theme = next;
      try { localStorage.setItem('iv-theme', next); } catch (e) {}
    });
  }

  // ----- tabs (index page) -----
  var tabs = $$('[data-tab]');
  function showTab(name) {
    if (!tabs.length) return;
    if (!$('[data-panel="' + name + '"]')) name = 'signature';
    tabs.forEach(function (t) { t.setAttribute('aria-selected', t.dataset.tab === name ? 'true' : 'false'); });
    $$('[data-panel]').forEach(function (p) { p.classList.toggle('on', p.dataset.panel === name); });
    $$('[data-pane]').forEach(function (p) { p.classList.toggle('on', p.dataset.pane === name); });
  }
  if (tabs.length) {
    tabs.forEach(function (t) {
      t.addEventListener('click', function () { showTab(t.dataset.tab); history.replaceState(null, '', '#' + t.dataset.tab); });
    });
    showTab((location.hash || '').slice(1) || 'signature');
  }

  // ----- file pickers: previews + drag and drop -----
  function size(b) { return b > 1048576 ? (b / 1048576).toFixed(1) + ' MB' : Math.max(1, Math.round(b / 1024)) + ' KB'; }
  function render(input) {
    var host = document.getElementById(input.dataset.target);
    var cta = $('.cta', input.closest('.pick'));
    var files = Array.prototype.slice.call(input.files);
    if (cta) cta.textContent = files.length ? files.length + (files.length > 1 ? ' files selected' : ' file selected') + ' - change' : cta.dataset.empty;
    if (!host) return;
    host.innerHTML = '';
    files.slice(0, 10).forEach(function (f, i) {
      var fig = document.createElement('figure'); fig.className = 'slot';
      var box = document.createElement('div'); box.className = 'im';
      var img = new Image(); img.alt = f.name; img.src = URL.createObjectURL(f);
      img.onload = function () { URL.revokeObjectURL(img.src); };
      box.appendChild(img);
      var cap = document.createElement('figcaption');
      var tag = document.createElement('b');
      tag.textContent = input.dataset.label + (input.multiple ? (i + 1) : '');
      cap.appendChild(tag); cap.appendChild(document.createTextNode(f.name + ' - ' + size(f.size)));
      fig.appendChild(box); fig.appendChild(cap); host.appendChild(fig);
    });
    var loaded = $$('.slot').length;
    var note = $('#pane-note'); if (note) note.textContent = loaded ? loaded + ' image' + (loaded > 1 ? 's' : '') + ' loaded' : 'nothing loaded';
  }
  $$('.pick input[type=file]').forEach(function (input) {
    input.addEventListener('change', function () { render(input); });
    var pick = input.closest('.pick');
    ['dragenter', 'dragover'].forEach(function (ev) {
      pick.addEventListener(ev, function (e) { e.preventDefault(); pick.classList.add('drag'); });
    });
    ['dragleave', 'drop'].forEach(function (ev) {
      pick.addEventListener(ev, function () { pick.classList.remove('drag'); });
    });
    pick.addEventListener('drop', function (e) {
      e.preventDefault();
      var list = e.dataTransfer && e.dataTransfer.files;
      if (!list || !list.length) return;
      try {
        var dt = new DataTransfer();
        Array.prototype.slice.call(list).slice(0, input.multiple ? 10 : 1).forEach(function (f) { dt.items.add(f); });
        input.files = dt.files; render(input);
      } catch (err) { /* older browsers: use the file dialog */ }
    });
  });

  // ----- form checks + busy state -----
  $$('form[data-check]').forEach(function (form) {
    form.addEventListener('submit', function (e) {
      var refs = $('input[name=references]', form);
      var old = $('.notice.js', form); if (old) old.remove();
      if (refs && (refs.files.length < 2 || refs.files.length > 10)) {
        e.preventDefault();
        var p = document.createElement('p');
        p.className = 'notice err js'; p.setAttribute('role', 'alert');
        p.textContent = 'Add between 2 and 10 known specimens.';
        form.insertBefore(p, form.firstChild);
        return;
      }
      var b = $('button[type=submit]', form);
      b.disabled = true; b.textContent = 'Working...';
    });
  });
  window.addEventListener('pageshow', function (ev) {
    if (!ev.persisted) return;
    $$('form[data-check] button[type=submit]').forEach(function (b) { b.disabled = false; b.textContent = b.dataset.label; });
  });

  // ----- document viewer -----
  var viewer = $('#viewer');
  if (viewer) {
    var handle = $('#handle'), topImg = $('.v-top', viewer), timer = null;
    function setPos(pct) {
      pct = Math.max(0, Math.min(100, pct));
      viewer.style.setProperty('--pos', pct + '%');
      handle.setAttribute('aria-valuenow', Math.round(pct));
    }
    function fromEvent(e) {
      var r = viewer.getBoundingClientRect();
      setPos((e.clientX - r.left) / r.width * 100);
    }
    var dragging = false;
    viewer.addEventListener('pointerdown', function (e) {
      if (viewer.dataset.mode !== 'swipe') return;
      dragging = true; viewer.setPointerCapture(e.pointerId); fromEvent(e);
    });
    viewer.addEventListener('pointermove', function (e) { if (dragging) fromEvent(e); });
    ['pointerup', 'pointercancel'].forEach(function (ev) { viewer.addEventListener(ev, function () { dragging = false; }); });
    handle.addEventListener('keydown', function (e) {
      var v = Number(handle.getAttribute('aria-valuenow'));
      if (e.key === 'ArrowLeft') { setPos(v - 3); e.preventDefault(); }
      if (e.key === 'ArrowRight') { setPos(v + 3); e.preventDefault(); }
      if (e.key === 'Home') setPos(0);
      if (e.key === 'End') setPos(100);
    });
    function setMode(mode) {
      viewer.dataset.mode = mode;
      $$('[data-mode]').forEach(function (b) { if (b.tagName === 'BUTTON') b.setAttribute('aria-pressed', b.dataset.mode === mode ? 'true' : 'false'); });
      clearInterval(timer); topImg.classList.remove('on');
      if (mode === 'flicker') timer = setInterval(function () { topImg.classList.toggle('on'); }, 650);
    }
    $$('[data-mode]').forEach(function (b) { if (b.tagName === 'BUTTON') b.addEventListener('click', function () { setMode(b.dataset.mode); }); });
    var showBoxes = $('#show-boxes');
    if (showBoxes) showBoxes.addEventListener('change', function () { viewer.classList.toggle('show-boxes', showBoxes.checked); });
    $$('#regions tr[data-i]').forEach(function (row) {
      var box = $('.box[data-i="' + row.dataset.i + '"]', viewer);
      if (!box) return;
      row.addEventListener('mouseenter', function () { box.classList.add('hot'); });
      row.addEventListener('mouseleave', function () { box.classList.remove('hot'); });
      row.addEventListener('click', function () {
        box.classList.add('hot'); viewer.scrollIntoView({ behavior: 'smooth', block: 'center' });
        setTimeout(function () { box.classList.remove('hot'); }, 1800);
      });
    });
    window.addEventListener('beforeunload', function () { clearInterval(timer); });
  }

  // ----- summary export / print -----
  var save = $('#save-summary');
  if (save) save.addEventListener('click', function () {
    var blob = new Blob([$('#summary-text').value], { type: 'text/plain;charset=utf-8' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = save.dataset.filename;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(function () { URL.revokeObjectURL(a.href); }, 1000);
  });
  var pr = $('#print-btn');
  if (pr) pr.addEventListener('click', function () { window.print(); });
})();
