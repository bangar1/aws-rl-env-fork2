/* ===== Navbar scroll — pill shape on scroll ===== */
const nav = document.getElementById('navbar');

window.addEventListener('scroll', () => {
  nav.classList.toggle('scrolled', window.scrollY > 40);
}, { passive: true });

/* ===== Active nav link on scroll ===== */
const sectionWrappers = document.querySelectorAll('.section-wrapper[id]');
const navLinks = document.querySelectorAll('.nav-links a');

function updateActiveNav() {
  const readingLine = window.scrollY + window.innerHeight / 3;
  let current = '';

  sectionWrappers.forEach(s => {
    const rect = s.getBoundingClientRect();
    const absoluteTop = rect.top + window.scrollY;
    if (absoluteTop <= readingLine) {
      current = s.id;
    }
  });

  // Update nav links
  navLinks.forEach(l => {
    const href = l.getAttribute('href');
    const isActive = href === '#' + current;
    l.classList.toggle('active', isActive);
  });

  // Mobile: show active section name
  nav.setAttribute('data-active-section', current || '');
}

window.addEventListener('scroll', updateActiveNav, { passive: true });
updateActiveNav();

/* ===== Smooth scroll with offset for fixed nav ===== */
document.querySelectorAll('a[href^="#"]').forEach(link => {
  link.addEventListener('click', e => {
    const target = document.querySelector(link.getAttribute('href'));
    if (!target) return;
    e.preventDefault();
    const rect = target.getBoundingClientRect();
    const absoluteTop = rect.top + window.scrollY;
    const offset = 100;
    window.scrollTo({
      top: absoluteTop - offset,
      behavior: 'smooth'
    });
  });
});

/* ===== Hero parallax grid & spotlight ===== */
const heroBg = document.querySelector('.hero-bg');

document.addEventListener('mousemove', e => {
  const x = e.clientX;
  const y = e.clientY;

  // Hero parallax
  if (heroBg) {
    heroBg.style.setProperty('--bg-x', (x * 0.02) + 'px');
    heroBg.style.setProperty('--bg-y', (y * 0.02) + 'px');
    heroBg.style.setProperty('--mouse-x', x + 'px');
    heroBg.style.setProperty('--mouse-y', y + 'px');
  }

  // Card spotlight tracking
  document.querySelectorAll('.card, .minimal-card').forEach(card => {
    const r = card.getBoundingClientRect();
    card.style.setProperty('--mouse-x', (x - r.left) + 'px');
    card.style.setProperty('--mouse-y', (y - r.top) + 'px');
  });
}, { passive: true });

/* ===== Typewriter — character-by-character reveal ===== */
function typewrite(el, delay) {
  const text = el.textContent;
  el.innerHTML = '';

  const chars = [];
  for (const ch of text) {
    const span = document.createElement('span');
    span.classList.add('char');
    span.textContent = ch;
    el.appendChild(span);
    chars.push(span);
  }

  // Insert a real cursor element that moves with the text
  const cursor = document.createElement('span');
  cursor.classList.add('typing-cursor');
  cursor.textContent = '|';

  return new Promise(resolve => {
    chars.forEach((span, i) => {
      setTimeout(() => {
        span.classList.add('visible');
        // Move cursor right after the latest visible char
        span.after(cursor);
        if (i === chars.length - 1) {
          resolve();
        }
      }, delay + i * 30);
    });
    if (chars.length === 0) resolve();
  });
}

// Typewrite hero elements sequentially: subtitle starts after title finishes
(async function () {
  const heroTitle = document.getElementById('hero-title');
  const heroSub = document.getElementById('hero-subtitle');

  // Hide subtitle until its turn
  if (heroSub) heroSub.style.visibility = 'hidden';

  if (heroTitle) {
    await typewrite(heroTitle, 300);
    heroTitle.querySelector('.typing-cursor')?.remove();
  }

  if (heroSub) {
    heroSub.style.visibility = 'visible';
    await typewrite(heroSub, 200);
    heroSub.querySelector('.typing-cursor')?.remove();
  }

  // Fade in hero CTA after both animations complete
  setTimeout(() => {
    document.querySelectorAll('.hero-fade-up').forEach(el => el.classList.add('visible'));
  }, 200);
})();

/* ===== Intersection Observer — fade-up on scroll ===== */
const observer = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      e.target.classList.add('visible');
      observer.unobserve(e.target);
    }
  });
}, { threshold: 0.1, rootMargin: '-50px' });

document.querySelectorAll('.animate-up').forEach(el => observer.observe(el));

/* ===== Playground Logic ===== */
const COLORS = {
  warmup: '#34a853', beginner: '#1a73e8', intermediate: '#f9ab00',
  advanced: '#ea4335', expert: '#7627bb'
};
const COLOR_BG = {
  warmup: '#e6f4ea', beginner: '#e8f0fe', intermediate: '#fef7e0',
  advanced: '#fce8e6', expert: '#f3e8fd'
};
let stepCount = 0;

// Services that have official AWS SVG files in /static/img/aws/
const SVC_IMG_FILES = ['s3', 'sqs', 'sns', 'lambda', 'dynamodb', 'iam', 'ec2', 'rds', 'cloudformation', 'cloudwatch', 'route53', 'apigateway', 'apigateway_v1', 'elasticache', 'elbv2', 'events', 'ssm', 'cognito-idp', 'glue', 'firehose', 'athena', 'emr', 'efs', 'ebs', 'kinesis', 'logs', 'monitoring', 'ses', 'ses_v2', 'acm', 'wafv2', 'states', 'secretsmanager', 'ecs', 'elasticmapreduce', 'elasticloadbalancing', 'elasticfilesystem'];

const DEFAULT_ICON = '<circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/>';

function _svcIconHtml(svc) {
  if (SVC_IMG_FILES.includes(svc)) {
    return '<img src="/static/img/aws/' + svc + '.svg" alt="' + svc + '" style="width:36px;height:36px;border-radius:6px;">';
  }
  return '<svg viewBox="0 0 24 24">' + DEFAULT_ICON + '</svg>';
}

// Cache infra data for modal drill-down
let _lastInfraServices = {};

async function refreshState() {
  try {
    const res = await fetch('/web/state');
    const state = await res.json();

    // Update sidebar stats
    document.getElementById('stateSteps').textContent = state.tracker ? state.tracker.step_count : '0';
    document.getElementById('stateHints').textContent = state.tracker ? state.tracker.hints_used : '0';
    const chaosEl = document.getElementById('stateChaos');
    if (state.chaos_occurred) {
      chaosEl.textContent = 'Active';
      chaosEl.className = 'state-value chaos-active';
    } else {
      chaosEl.textContent = 'None';
      chaosEl.className = 'state-value chaos-inactive';
    }

    // Render infra tiles
    const grid = document.getElementById('infraGrid');
    const services = state.infra_state && state.infra_state.services ? state.infra_state.services : {};
    _lastInfraServices = services;
    const svcKeys = Object.keys(services);
    if (svcKeys.length === 0) {
      grid.innerHTML = '<p style="color:var(--text-muted);font-size:0.9rem;margin:0;">No data.</p>';
      return;
    }

    let html = '';
    for (const svc of svcKeys) {
      const data = services[svc];
      let totalCount = 0;
      for (const [, resData] of Object.entries(data)) {
        if (resData && typeof resData === 'object') {
          if (typeof resData.count === 'number') {
            totalCount += resData.count;
          } else if (Array.isArray(resData)) {
            totalCount += resData.length;
          } else {
            // Nested object keyed by ID (e.g. apigateway_v1 rest_apis)
            const keys = Object.keys(resData);
            if (keys.length > 0) totalCount += keys.length;
          }
        }
      }
      const hasRes = totalCount > 0;
      html += '<div class="infra-tile' + (hasRes ? ' has-resources' : '') + '" onclick="openInfraModal(\'' + svc + '\')">' +
        (hasRes ? '<span class="infra-tile-badge">' + totalCount + '</span>' : '') +
        '<div class="infra-tile-icon">' + _svcIconHtml(svc) + '</div>' +
        '<span class="infra-tile-name">' + escHtml(svc) + '</span>' +
        '</div>';
    }
    grid.className = 'infra-tiles';
    grid.innerHTML = html;
  } catch (e) {
    // Silent fail
  }
}

// Infra modal
function _renderResItems(obj) {
  // Renders items for the modal body — handles arrays, {count,names}, and nested objects
  if (!obj || typeof obj !== 'object') return '<div class="infra-res-item">' + escHtml(String(obj)) + '</div>';
  if (Array.isArray(obj)) {
    return obj.map(function (item) { return '<div class="infra-res-item">' + escHtml(String(item)) + '</div>'; }).join('');
  }
  // Has {count, names/ids} pattern
  if (typeof obj.count === 'number') {
    var items = obj.names || obj.ids || [];
    return items.map(function (item) { return '<div class="infra-res-item">' + escHtml(String(item)) + '</div>'; }).join('') ||
      '<div class="infra-res-item" style="color:var(--text-muted);">Empty (' + obj.count + ')</div>';
  }
  // Nested keyed object — render each key as a sub-item
  var keys = Object.keys(obj);
  if (keys.length === 0) return '';
  var out = '';
  for (var k of keys) {
    var val = obj[k];
    if (val && typeof val === 'object' && !Array.isArray(val)) {
      // Show key with a summary
      var name = val.name || val.Name || val.id || val.Id || k;
      var detail = val.description || val.engine || val.runtime || val.protocol || '';
      out += '<div class="infra-res-item"><strong>' + escHtml(String(name)) + '</strong>' +
        (detail ? ' <span style="color:var(--text-muted);">\u2014 ' + escHtml(String(detail)) + '</span>' : '') +
        '</div>';
    } else {
      out += '<div class="infra-res-item">' + escHtml(k + ': ' + JSON.stringify(val)) + '</div>';
    }
  }
  return out;
}

function _countResources(resData) {
  if (!resData || typeof resData !== 'object') return 0;
  if (typeof resData.count === 'number') return resData.count;
  if (Array.isArray(resData)) return resData.length;
  return Object.keys(resData).length;
}

function openInfraModal(svc) {
  const data = _lastInfraServices[svc];
  if (!data) return;
  document.getElementById('infra-modal-title').textContent = svc.toUpperCase();
  const body = document.getElementById('infra-modal-body');
  let html = '';
  for (const [resType, resData] of Object.entries(data)) {
    if (!resData || typeof resData !== 'object') continue;
    var count = _countResources(resData);
    const groupId = 'infra-g-' + svc + '-' + resType.replace(/[^a-z0-9]/gi, '');
    html += '<div class="infra-res-group">' +
      '<div class="infra-res-header" onclick="var el=document.getElementById(\'' + groupId + '\');if(el)el.classList.toggle(\'open\')">' +
      '<span class="infra-res-title">' + escHtml(resType.replace(/_/g, ' ')) + '</span>' +
      '<span class="infra-res-count">' + count + '</span>' +
      '</div>';
    var itemsHtml = _renderResItems(resData);
    if (itemsHtml) {
      html += '<div class="infra-res-body" id="' + groupId + '">' + itemsHtml + '</div>';
    }
    html += '</div>';
  }
  body.innerHTML = html || '<p style="color:var(--text-muted);">No resources in this service.</p>';
  document.getElementById('infra-modal').classList.add('open');
  document.body.style.overflow = 'hidden';
}

function closeInfraModal() {
  document.getElementById('infra-modal').classList.remove('open');
  document.body.style.overflow = '';
}

// Command log modal
let _logEntries = [];

function openLogModal(index) {
  const entry = _logEntries[index];
  if (!entry) return;
  document.getElementById('log-modal-title').textContent = 'Step #' + entry.step;
  document.getElementById('log-modal-cmd').textContent = entry.command;
  document.getElementById('log-modal-status').innerHTML = entry.success
    ? '<span style="color:#34a853;font-weight:500;">Success</span>'
    : '<span style="color:#ea4335;font-weight:500;">Failed</span>';
  document.getElementById('log-modal-reward').textContent = (entry.reward >= 0 ? '+' : '') + entry.reward.toFixed(2);
  document.getElementById('log-modal-output').textContent = entry.output || 'No output';
  document.getElementById('log-modal').classList.add('open');
  document.body.style.overflow = 'hidden';
}

function closeLogModal() {
  document.getElementById('log-modal').classList.remove('open');
  document.body.style.overflow = '';
}

// Close modals on Escape / backdrop click
document.addEventListener('keydown', function (e) {
  if (e.key === 'Escape') { closeInfraModal(); closeLogModal(); }
});
['infra-modal', 'log-modal'].forEach(function (id) {
  var el = document.getElementById(id);
  if (el) el.addEventListener('click', function (e) {
    if (e.target.id === id) { closeInfraModal(); closeLogModal(); }
  });
});

function setStatus(msg, type) {
  const bar = document.getElementById('statusBar');
  bar.className = 'status-bar ' + (type || '');
  bar.innerHTML = msg;
}

function setLoading(btn, loading) {
  if (loading) {
    btn.disabled = true;
    btn.dataset.orig = btn.textContent;
  }
  btn.innerHTML = loading
    ? '<span class="spinner"></span>' + (btn.dataset.orig || '')
    : (btn.dataset.orig || btn.textContent);
}

function escHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

async function resetEnv() {
  const btn = document.getElementById('resetBtn');
  setLoading(btn, true);
  setStatus('Resetting environment...', 'info');
  try {
    const res = await fetch('/web/reset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: '{}'
    });
    const data = await res.json();
    const obs = data.observation;
    stepCount = 0;

    const task = obs.task;
    const box = document.getElementById('taskBox');
    if (task) {
      const color = COLORS[task.difficulty] || '#5f6368';
      const bg = COLOR_BG[task.difficulty] || '#f1f3f4';
      box.className = 'task-box';
      box.style.borderLeftColor = color;
      box.innerHTML =
        '<div>' +
        '<span class="task-badge" style="background:' + bg + ';color:' + color + ';">' + escHtml(task.difficulty) + '</span>' +
        '<span class="task-meta">Task #' + task.task_id + '</span>' +
        '</div>' +
        '<p class="task-desc">' + escHtml(task.description) + '</p>';
    }

    document.getElementById('outputBox').textContent = obs.command_output || '';
    document.getElementById('logBody').innerHTML =
      '<tr><td colspan="4" class="log-empty">No commands executed yet</td></tr>';
    _logEntries = [];
    // Enable command controls
    document.getElementById('cmdInput').disabled = false;
    document.getElementById('runBtn').disabled = false;
    delete document.getElementById('runBtn').dataset.ended;
    document.getElementById('solutionBtn').disabled = false;
    document.getElementById('solutionBtn').innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/></svg> Show Solution';
    document.getElementById('solutionPanel').style.display = 'none';
    document.getElementById('solutionCommands').innerHTML = '';
    document.getElementById('cmdInput').value = '';
    document.getElementById('cmdInput').focus();

    // Update state box
    document.getElementById('stateTier').textContent = task ? task.difficulty : '\u2014';
    document.getElementById('stateEpisode').textContent = obs.episode_id || '1';
    document.getElementById('stateProgress').style.width = '0%';
    document.getElementById('stateReward').textContent = '0.00';
    setStatus('New episode started. Difficulty: <strong>' + (task ? escHtml(task.difficulty) : 'unknown') + '</strong>', 'info');
    refreshState();
  } catch (e) {
    setStatus('Reset failed: ' + escHtml(e.message), 'error');
  } finally {
    setLoading(btn, false);
    btn.disabled = false;
  }
}

async function runCmd() {
  const input = document.getElementById('cmdInput');
  const cmd = input.value.trim();
  if (!cmd) return;

  const btn = document.getElementById('runBtn');
  setLoading(btn, true);
  try {
    const res = await fetch('/web/step', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: { command: cmd } })
    });
    const data = await res.json();

    if (!res.ok) {
      setStatus('Error: ' + escHtml(data.detail || JSON.stringify(data)), 'error');
      return;
    }

    const obs = data.observation;
    stepCount++;

    const output = obs.command_success
      ? (obs.command_output || '')
      : (obs.error || obs.command_output || '');
    document.getElementById('outputBox').textContent = output;

    const tbody = document.getElementById('logBody');
    if (stepCount === 1) { tbody.innerHTML = ''; _logEntries = []; }
    const reward = (obs.reward != null ? obs.reward : (data.reward || 0));
    const logIdx = _logEntries.length;
    _logEntries.push({ step: stepCount, command: cmd, success: obs.command_success, reward: reward, output: output });
    const tr = document.createElement('tr');
    tr.onclick = function () { openLogModal(logIdx); };
    const displayCmd = cmd.length > 60 ? cmd.slice(0, 57) + '...' : cmd;
    tr.innerHTML =
      '<td>' + stepCount + '</td>' +
      '<td class="cmd">' + escHtml(displayCmd) + '</td>' +
      '<td class="' + (obs.command_success ? 'yes' : 'no') + '">' + (obs.command_success ? 'Yes' : 'No') + '</td>' +
      '<td>' + (reward >= 0 ? '+' : '') + Number(reward).toFixed(2) + '</td>';
    tbody.appendChild(tr);

    // Update state box
    const progress = obs.partial_progress != null ? obs.partial_progress : 0;
    document.getElementById('stateProgress').style.width = (progress * 100) + '%';
    const cumReward = parseFloat(document.getElementById('stateReward').textContent) + reward;
    document.getElementById('stateReward').textContent = cumReward.toFixed(2);

    if (obs.task_achieved) {
      setStatus('Task completed! Step ' + obs.step_count + ', reward: +' + Number(reward).toFixed(2) + '. Click <strong>New Episode</strong> for the next task.', 'success');
      document.getElementById('cmdInput').disabled = true;
      document.getElementById('runBtn').disabled = true;
      document.getElementById('runBtn').dataset.ended = '1';
      document.getElementById('solutionBtn').disabled = true;
    } else if (data.done) {
      setStatus('Episode ended. Click <strong>New Episode</strong> to try again.', 'error');
      document.getElementById('cmdInput').disabled = true;
      document.getElementById('runBtn').disabled = true;
      document.getElementById('runBtn').dataset.ended = '1';
      document.getElementById('solutionBtn').disabled = true;
    } else {
      setStatus('Step <strong>' + obs.step_count + '</strong> &mdash; ' + (obs.command_success ? 'Command succeeded.' : 'Command failed.'), obs.command_success ? 'info' : 'error');
    }

    refreshState();
    input.value = '';
    input.focus();
  } catch (e) {
    setStatus('Request failed: ' + escHtml(e.message), 'error');
  } finally {
    setLoading(btn, false);
    // Re-enable if episode is still active (not disabled by completion/done handlers above)
    if (!btn.dataset.ended) {
      btn.disabled = false;
    }
  }
}
