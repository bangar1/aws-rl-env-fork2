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
  el.classList.add('typing-cursor-inline');

  const chars = [];
  for (const ch of text) {
    const span = document.createElement('span');
    span.classList.add('char');
    span.textContent = ch === ' ' ? '\u00A0' : ch;
    el.appendChild(span);
    chars.push(span);
  }

  chars.forEach((span, i) => {
    setTimeout(() => {
      span.classList.add('visible');
      // Last char: switch to blinking cursor
      if (i === chars.length - 1) {
        el.classList.add('blinking');
      }
    }, delay + i * 30);
  });
}

// Typewrite hero elements
document.querySelectorAll('.hero .type-animate').forEach((el, i) => {
  typewrite(el, 300 + i * 600);
});

// Fade in hero CTA after typewriter completes
const heroTitle = document.getElementById('hero-title');
const heroSub = document.getElementById('hero-subtitle');
if (heroTitle && heroSub) {
  const titleLen = heroTitle.textContent.replace(/\u00A0/g, ' ').length;
  const subLen = heroSub.textContent.replace(/\u00A0/g, ' ').length;
  const totalDelay = 300 + titleLen * 30 + 600 + subLen * 30 + 200;
  setTimeout(() => {
    document.querySelectorAll('.hero-fade-up').forEach(el => el.classList.add('visible'));
  }, totalDelay);
}

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
    if (stepCount === 1) tbody.innerHTML = '';
    const tr = document.createElement('tr');
    const reward = (obs.reward != null ? obs.reward : (data.reward || 0));
    const displayCmd = cmd.length > 50 ? cmd.slice(0, 47) + '...' : cmd;
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
