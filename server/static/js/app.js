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
  btn.disabled = loading;
  if (loading) btn.dataset.orig = btn.textContent;
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
    document.getElementById('cmdInput').value = '';
    document.getElementById('cmdInput').focus();
    setStatus('New episode started. Difficulty: <strong>' + (task ? escHtml(task.difficulty) : 'unknown') + '</strong>', 'info');
  } catch (e) {
    setStatus('Reset failed: ' + escHtml(e.message), 'error');
  } finally {
    setLoading(btn, false);
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

    if (obs.task_achieved) {
      setStatus('Task completed! Step ' + obs.step_count + ', reward: +' + Number(reward).toFixed(2) + '. Click <strong>New Episode</strong> for the next task.', 'success');
    } else if (data.done) {
      setStatus('Episode ended. Click <strong>New Episode</strong> to try again.', 'error');
    } else {
      setStatus('Step <strong>' + obs.step_count + '</strong> &mdash; ' + (obs.command_success ? 'Command succeeded.' : 'Command failed.'), obs.command_success ? 'info' : 'error');
    }

    input.value = '';
    input.focus();
  } catch (e) {
    setStatus('Request failed: ' + escHtml(e.message), 'error');
  } finally {
    setLoading(btn, false);
  }
}
