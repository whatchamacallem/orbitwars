"""
Orbit Wars interactive visualizer.

Usage in your agent:
    from visualizer import Visualizer
    viz = Visualizer()

    def agent(obs):
        viz.record(obs)
        viz.add_line(obs['step'], x1, y1, x2, y2, color='cyan')   # optional
        viz.add_text(obs['step'], "some debug text")              # optional
        ...

    # After env.run():
    viz.save("viz.html")
"""

import json


class Visualizer:
    def __init__(self):
        self._frames = []   # list of {obs, lines, texts}
        self._frame_map = {}  # step -> index
        self._recording = True

    def record(self, obs):
        """Call once per turn with the raw obs dict."""
        if hasattr(obs, '__dict__'):
            obs = vars(obs)
        if not isinstance(obs, dict):
            raise TypeError(f"obs must be a dict, got {type(obs)}")
        # obs.step=N means the engine has completed N-1 rotations, so planets sit at N-1.
        step = obs['step'] - 1
        if 'planets' not in obs:
            raise KeyError(f"obs missing 'planets' at step {step}")
        if 'fleets' not in obs:
            raise KeyError(f"obs missing 'fleets' at step {step}")
        if not obs['planets']:
            return  # step 0 init call has empty state; skip it
        self._recording = obs.get('player') == 0
        if not self._recording:
            return
        entry = {'obs': _serialize(obs), 'lines': [], 'texts': [], 'labels': []}
        self._frame_map[step] = len(self._frames)
        self._frames.append(entry)

    def _get_or_create(self, step):
        if step in self._frame_map:
            return self._frames[self._frame_map[step]]
        entry = {'obs': {}, 'lines': [], 'texts': [], 'labels': []}
        self._frame_map[step] = len(self._frames)
        self._frames.append(entry)
        return entry

    def add_line(self, step, x1, y1, x2, y2, color='yellow', width=1):
        """Add a colored line segment overlay to a specific frame."""
        if not self._recording:
            return
        self._get_or_create(step)['lines'].append(
            {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'color': color, 'width': width}
        )

    def add_arrow(self, step, x1, y1, x2, y2, color='yellow', width=1, length_frac=0.5, head_size=6):
        """Add a directed arrow: drawn from x1,y1 toward x2,y2 but only length_frac of the way."""
        if not self._recording:
            return
        self._get_or_create(step)['lines'].append({
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'color': color, 'width': width,
            'arrow': True, 'length_frac': length_frac, 'head_size': head_size,
        })

    def add_text(self, step, text):
        """Add a debug text string shown when that frame is active."""
        if not self._recording:
            return
        self._get_or_create(step)['texts'].append(str(text))

    def add_label(self, step, x, y, text, color='#ffffff', font='13px monospace'):
        """Draw a text label at canvas position (x, y) in game-world coordinates."""
        if not self._recording:
            return
        self._get_or_create(step)['labels'].append(
            {'x': x, 'y': y, 'text': str(text), 'color': color, 'font': font}
        )

    def save(self, path):
        """Write the interactive HTML visualizer to path."""
        html = _build_html(self._frames)
        with open(path, 'w') as f:
            f.write(html)
        print(f"Visualizer saved to {path} ({len(self._frames)} frames)")


_PLANET_FIELDS = ('id', 'owner', 'x', 'y', 'radius', 'ships', 'production')
_FLEET_FIELDS  = ('id', 'owner', 'x', 'y', 'angle', 'from_planet_id', 'ships')

def _serialize(obj, _key=None):
    """Recursively convert Namespace/objects to plain dicts/lists."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[k] = _serialize(v, _key=k)
        return out
    if isinstance(obj, (list, tuple)):
        # Detect Planet/Fleet objects stored as list elements and normalize them
        result = []
        for item in obj:
            result.append(_serialize(item, _key=_key))
        return result
    if hasattr(obj, '__dict__'):
        d = vars(obj)
        # Planet-like object: has all planet fields
        if all(f in d for f in _PLANET_FIELDS):
            return [d[f] for f in _PLANET_FIELDS]
        # Fleet-like object: has all fleet fields
        if all(f in d for f in _FLEET_FIELDS):
            return [d[f] for f in _FLEET_FIELDS]
        return _serialize(d)
    return obj


def _build_html(frames):
    frames_json = json.dumps(frames)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Orbit Wars Visualizer</title>
<style>
  body {{ margin: 0; background: #0a0a12; color: #ccc; font-family: monospace; display: flex; flex-direction: column; align-items: center; }}
#controls {{ display: flex; align-items: center; gap: 8px; margin: 4px 0; flex-wrap: wrap; justify-content: center; }}
  button {{ background: #1e2a3a; color: #7af; border: 1px solid #345; padding: 4px 12px; cursor: pointer; border-radius: 3px; }}
  button:hover {{ background: #2a3f55; }}
  #frameLabel {{ color: #fa8; min-width: 80px; text-align: center; }}
  #slider {{ width: 500px; max-width: 90vw; accent-color: #7af; }}
  #speedLabel {{ min-width: 60px; }}
  #main {{ display: flex; gap: 12px; align-items: flex-start; }}
  canvas {{ border: 1px solid #334; background: #05050f; }}
  #sidebar {{ width: 520px; max-height: 640px; overflow-y: auto; font-size: 0.75rem; }}
  .panel {{ background: #0f1520; border: 1px solid #234; border-radius: 4px; padding: 6px 8px; margin-bottom: 8px; }}
  .panel h3 {{ margin: 0 0 4px; color: #7af; font-size: 0.8rem; border-bottom: 1px solid #234; padding-bottom: 2px; }}
  .item {{ margin: 2px 0; }}
  .planet-owner-0 {{ color: #4af; }}
  .planet-owner-1 {{ color: #f84; }}
  .planet-owner-2 {{ color: #4f8; }}
  .planet-owner-3 {{ color: #f4f; }}
  .planet-owner-neutral {{ color: #888; }}
  .fleet-owner-0 {{ color: #4af; }}
  .fleet-owner-1 {{ color: #f84; }}
  .fleet-owner-2 {{ color: #4f8; }}
  .fleet-owner-3 {{ color: #f4f; }}
  #debugText {{ white-space: pre-wrap; color: #ff8; }}
  #playBtn.playing {{ color: #fa0; }}
</style>
</head>
<body>
<div id="controls">
  <button id="prevBtn">&#9664; Prev</button>
  <button id="playBtn">&#9654; Play</button>
  <button id="nextBtn">Next &#9654;</button>
  <span id="frameLabel">Frame 0</span>
  <input type="range" id="slider" min="0" value="0">
  <label>Speed: <input type="range" id="speedSlider" min="1" max="30" value="8" style="width:80px">
    <span id="speedLabel">8 fps</span></label>
</div>
<div id="main">
  <canvas id="canvas" width="640" height="640"></canvas>
  <div id="sidebar">
    <div class="panel" id="metaPanel"><h3>Status</h3><div id="metaContent"></div></div>
    <div class="panel" id="textPanel" style="display:none"><h3>Debug Text</h3><pre id="debugText"></pre></div>
    <div class="panel"><h3>Planets</h3><div id="planetContent"></div></div>
    <div class="panel"><h3>Fleets</h3><div id="fleetContent"></div></div>
  </div>
</div>
<script>
const FRAMES = {frames_json};
const COLORS = ['#44aaff','#ff8844','#44ff88','#ff44ff'];
const NEUTRAL_COLOR = '#666';
const COMET_COLOR = '#fc8';

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const slider = document.getElementById('slider');
const frameLabel = document.getElementById('frameLabel');
const playBtn = document.getElementById('playBtn');
const speedSlider = document.getElementById('speedSlider');
const speedLabel = document.getElementById('speedLabel');

slider.max = FRAMES.length - 1;
let current = 0;
let playing = false;
let playInterval = null;

function ownerColor(owner) {{
  if (owner === -1 || owner === undefined) return NEUTRAL_COLOR;
  return COLORS[owner % COLORS.length];
}}

// board is 100x100, canvas is 640x640
function tx(x) {{ return x * 6.4; }}
function ty(y) {{ return y * 6.4; }}
function tr(r) {{ return r * 6.4; }}

function drawFrame(idx) {{
  const frame = FRAMES[idx];
  const obs = frame.obs;
  ctx.clearRect(0, 0, 640, 640);

  // Background grid (faint)
  ctx.strokeStyle = '#1e2e40';
  ctx.lineWidth = 0.8;
  for (let i = 0; i <= 10; i++) {{
    ctx.beginPath(); ctx.moveTo(i*64, 0); ctx.lineTo(i*64, 640); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, i*64); ctx.lineTo(640, i*64); ctx.stroke();
  }}

  // Sun
  const sunRadius = 10;
  ctx.beginPath();
  ctx.arc(tx(50), ty(50), tr(sunRadius), 0, Math.PI*2);
  ctx.fillStyle = '#888888';
  ctx.fill();

  const cometIds = new Set(obs.comet_planet_ids || []);
  const planets = obs.planets || [];
  const fleets = obs.fleets || [];

  // Planets
  for (const p of planets) {{
    const [id, owner, x, y, radius, ships, production] = p;
    const cx = tx(x), cy = ty(y), cr = tr(radius);
    const isComet = cometIds.has(id);
    const col = ownerColor(owner);

    // Glow
    if (owner >= 0) {{
      const glow = ctx.createRadialGradient(cx, cy, 0, cx, cy, cr*2.5);
      glow.addColorStop(0, col + '55');
      glow.addColorStop(1, col + '00');
      ctx.beginPath();
      ctx.arc(cx, cy, cr*2.5, 0, Math.PI*2);
      ctx.fillStyle = glow;
      ctx.fill();
    }}

    ctx.beginPath();
    ctx.arc(cx, cy, cr, 0, Math.PI*2);
    ctx.fillStyle = isComet ? COMET_COLOR : (owner >= 0 ? col : '#334');
    ctx.fill();
    ctx.strokeStyle = isComet ? '#fd0' : col;
    ctx.lineWidth = isComet ? 1.5 : (owner >= 0 ? 1.5 : 0.8);
    ctx.stroke();

    // Ship count label
    ctx.fillStyle = '#fff';
    ctx.font = `bold ${{Math.max(9, cr*0.9)}}px monospace`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(ships, cx, cy);

    // ID label (small, above)
    ctx.fillStyle = '#aaa';
    ctx.font = '13px monospace';
    ctx.fillText('P' + id, cx, cy - cr - 5);
  }}

  // Fleets
  for (const f of fleets) {{
    const [id, owner, x, y, angle, from_id, ships] = f;
    const fx = tx(x), fy = ty(y);
    const col = ownerColor(owner);

    // Arrow body
    const len = (6 + Math.log1p(ships) * 1.5) * 1.5;
    const dx = Math.cos(angle), dy = Math.sin(angle);
    ctx.strokeStyle = col;
    ctx.lineWidth = 2.25;
    ctx.beginPath();
    ctx.moveTo(fx - dx*len*0.5, fy - dy*len*0.5);
    ctx.lineTo(fx + dx*len*0.5, fy + dy*len*0.5);
    ctx.stroke();

    // Arrowhead
    ctx.fillStyle = col;
    ctx.beginPath();
    const hx = fx + dx*len*0.5, hy = fy + dy*len*0.5;
    const px = -dy, py = dx; // perpendicular
    ctx.moveTo(hx, hy);
    ctx.lineTo(hx - dx*7.5 + px*4.5, hy - dy*7.5 + py*4.5);
    ctx.lineTo(hx - dx*7.5 - px*4.5, hy - dy*7.5 - py*4.5);
    ctx.closePath();
    ctx.fill();

    // Ship count (for larger fleets)
    if (ships >= 5) {{
      ctx.fillStyle = '#fff';
      ctx.font = '13px monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(ships, fx + px*9, fy + py*9);
    }}
  }}

  // Debug overlay lines / arrows
  for (const line of (frame.lines || [])) {{
    const ax = tx(line.x1), ay = ty(line.y1);
    const bx = tx(line.x2), by = ty(line.y2);
    const col = line.color || 'yellow';
    const lw = line.width || 1;
    if (line.arrow) {{
      const frac = line.length_frac !== undefined ? line.length_frac : 0.5;
      const hs = line.head_size !== undefined ? line.head_size : 6;
      const ex = ax + (bx - ax) * frac, ey = ay + (by - ay) * frac;
      const dx = (bx - ax), dy = (by - ay);
      const len = Math.sqrt(dx*dx + dy*dy) || 1;
      const ux = dx/len, uy = dy/len;
      const px = -uy, py = ux;
      ctx.strokeStyle = col; ctx.lineWidth = lw;
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(ex, ey); ctx.stroke();
      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.moveTo(ex, ey);
      ctx.lineTo(ex - ux*hs + px*(hs*0.5), ey - uy*hs + py*(hs*0.5));
      ctx.lineTo(ex - ux*hs - px*(hs*0.5), ey - uy*hs - py*(hs*0.5));
      ctx.closePath(); ctx.fill();
    }} else {{
      ctx.beginPath();
      ctx.moveTo(ax, ay); ctx.lineTo(bx, by);
      ctx.strokeStyle = col; ctx.lineWidth = lw;
      ctx.stroke();
    }}
  }}

  // Canvas labels (e.g. future-position planet markers)
  for (const lbl of (frame.labels || [])) {{
    ctx.font = lbl.font || '13px monospace';
    ctx.fillStyle = lbl.color || '#ffffff';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(lbl.text, tx(lbl.x), ty(lbl.y));
  }}

  // Update sidebar
  frameLabel.textContent = 'Frame ' + idx + ' / ' + (FRAMES.length - 1);

  // Meta
  const step = obs.step !== undefined ? obs.step : idx;
  const player = obs.player !== undefined ? obs.player : '?';
  const av = obs.angular_velocity !== undefined ? obs.angular_velocity.toFixed(4) : '?';
  const ot = obs.remainingOverageTime !== undefined ? obs.remainingOverageTime.toFixed(2) : '?';
  document.getElementById('metaContent').innerHTML =
    `<div class="item">Step: <b>${{step}}</b> &nbsp; Player: <b>${{player}}</b></div>
     <div class="item">Angular vel: ${{av}} &nbsp; Overage: ${{ot}}s</div>
     <div class="item">Comets: ${{(obs.comet_planet_ids||[]).join(', ') || 'none'}}</div>`;

  // Debug texts
  const texts = frame.texts || [];
  const textPanel = document.getElementById('textPanel');
  if (texts.length > 0) {{
    textPanel.style.display = '';
    document.getElementById('debugText').textContent = texts.join('\\n');
  }} else {{
    textPanel.style.display = 'none';
  }}

  // Planets list
  let pHtml = '';
  for (const p of planets) {{
    const [id, owner, x, y, radius, ships, production] = p;
    const cls = owner >= 0 ? `planet-owner-${{owner}}` : 'planet-owner-neutral';
    const isComet = cometIds.has(id);
    pHtml += `<div class="item ${{cls}}">[P${{id}}${{isComet?'*':''}}] owner=${{owner}} ships=${{ships}} prod=${{production}} (${{x.toFixed(1)}},${{y.toFixed(1)}})</div>`;
  }}
  document.getElementById('planetContent').innerHTML = pHtml || '<div class="item">none</div>';

  // Fleets list
  let fHtml = '';
  for (const f of fleets) {{
    const [id, owner, x, y, angle, from_id, ships] = f;
    const cls = `fleet-owner-${{owner}}`;
    fHtml += `<div class="item ${{cls}}">[F${{id}}] owner=${{owner}} ships=${{ships}} from=P${{from_id}} (${{x.toFixed(1)}},${{y.toFixed(1)}}) ang=${{angle.toFixed(2)}}</div>`;
  }}
  document.getElementById('fleetContent').innerHTML = fHtml || '<div class="item">none</div>';
}}

function go(idx) {{
  current = Math.max(0, Math.min(FRAMES.length - 1, idx));
  slider.value = current;
  drawFrame(current);
}}

document.getElementById('prevBtn').onclick = () => {{ stopPlay(); go(current - 1); }};
document.getElementById('nextBtn').onclick = () => {{ stopPlay(); go(current + 1); }};
slider.oninput = () => {{ go(parseInt(slider.value)); }};

document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {{ stopPlay(); go(current - 1); }}
  if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {{ stopPlay(); go(current + 1); }};
  if (e.key === ' ') {{ e.preventDefault(); togglePlay(); }}
}});

function togglePlay() {{
  playing ? stopPlay() : startPlay();
}}

function startPlay() {{
  playing = true;
  playBtn.textContent = '⏸ Pause';
  playBtn.classList.add('playing');
  const fps = parseInt(speedSlider.value);
  playInterval = setInterval(() => {{
    if (current >= FRAMES.length - 1) {{ stopPlay(); return; }}
    go(current + 1);
  }}, 1000 / fps);
}}

function stopPlay() {{
  playing = false;
  playBtn.textContent = '▶ Play';
  playBtn.classList.remove('playing');
  clearInterval(playInterval);
  playInterval = null;
}}

playBtn.onclick = togglePlay;
canvas.onclick = togglePlay;

speedSlider.oninput = () => {{
  const fps = parseInt(speedSlider.value);
  speedLabel.textContent = fps + ' fps';
  if (playing) {{ stopPlay(); startPlay(); }}
}};

go(0);
</script>
</body>
</html>"""
