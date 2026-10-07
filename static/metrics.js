// Resource and performance metrics, shared by the chat and training pages.
// Usage: startMetrics(container, summaryEl?) polls /api/metrics and renders into `container`;
// the optional `summaryEl` receives a one-line digest.

const METRICS_POLL_MS = 2000;
const SVG_NS = "http://www.w3.org/2000/svg";
const DASH = "—";

function mEl(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function formatBytes(bytes) {
  if (bytes == null) return DASH;
  const gb = bytes / 1024 ** 3;
  if (gb >= 1) return `${gb.toFixed(gb >= 10 ? 1 : 2)} GB`;
  return `${Math.round(bytes / 1024 ** 2)} MB`;
}

function formatParams(count) {
  return count >= 1e9 ? `${(count / 1e9).toFixed(1)}B` : `${Math.round(count / 1e6)}M`;
}

function formatMs(ms) {
  return ms == null ? DASH : `${ms} ms`;
}

function formatUptime(seconds) {
  if (seconds < 60) return `${seconds} s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min`;
  return `${Math.floor(seconds / 3600)} h ${Math.floor((seconds % 3600) / 60)} min`;
}

function counts(entries) {
  const dl = mEl("dl", "counts");
  for (const [label, value] of entries) {
    const row = mEl("div");
    row.append(mEl("dt", null, label), mEl("dd", null, value ?? DASH));
    dl.append(row);
  }
  return dl;
}

function usageMeter(label, used, total) {
  const wrap = mEl("div", "usage");
  wrap.append(mEl("p", "usage-label", `${label}: ${formatBytes(used)} de ${formatBytes(total)}`));
  const meter = mEl("div", "meter");
  meter.setAttribute("role", "progressbar");
  meter.setAttribute("aria-label", label);
  meter.setAttribute("aria-valuemin", "0");
  meter.setAttribute("aria-valuemax", "100");
  const percent = total ? Math.min(100, (used / total) * 100) : 0;
  meter.setAttribute("aria-valuenow", String(Math.round(percent)));
  const fill = mEl("div", "meter-fill");
  fill.style.width = `${percent}%`;
  meter.append(fill);
  wrap.append(meter);
  return wrap;
}

function sparkline(values) {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("class", "sparkline");
  svg.setAttribute("viewBox", "0 0 100 28");
  svg.setAttribute("preserveAspectRatio", "none");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Latencia de los últimos mensajes");
  if (values.length < 2) return svg;
  const max = Math.max(...values);
  const min = Math.min(...values);
  const span = max - min || 1;
  const points = values.map((v, i) => {
    const x = (i / (values.length - 1)) * 100;
    const y = 26 - ((v - min) / span) * 24;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  const line = document.createElementNS(SVG_NS, "polyline");
  line.setAttribute("points", points.join(" "));
  line.setAttribute("fill", "none");
  line.setAttribute("stroke", "currentColor");
  line.setAttribute("stroke-width", "1.5");
  line.setAttribute("vector-effect", "non-scaling-stroke");
  svg.append(line);
  return svg;
}

function modelsSection(models) {
  const list = mEl("ul", "model-list");
  for (const m of models) {
    list.append(
      mEl("li", null, `${m.name} · ${m.device} · ${m.dtype} · ${formatParams(m.parameters)} parámetros · ${formatBytes(m.weights_bytes)}`),
    );
  }
  return list;
}

function renderMetrics(container, d) {
  const sections = [];

  if (d.models.length) {
    sections.push(mEl("h3", null, "Modelos cargados"), modelsSection(d.models));
  }

  const gm = d.gpu_memory;
  if (gm) {
    sections.push(mEl("h3", null, gm.name));
    sections.push(usageMeter("VRAM en uso", gm.total_bytes - gm.free_bytes, gm.total_bytes));
    sections.push(
      counts([
        ["Del proceso", formatBytes(gm.allocated_bytes)],
        ["Reservada", formatBytes(gm.reserved_bytes)],
        ["Pico", formatBytes(gm.peak_allocated_bytes)],
        ["Libre", formatBytes(gm.free_bytes)],
      ]),
    );
  }
  if (d.gpu) {
    sections.push(
      counts([
        ["Uso de GPU", `${d.gpu.gpu_percent}%`],
        ["Temperatura", `${d.gpu.temperature_c} °C`],
        ["Consumo", `${d.gpu.power_w} / ${d.gpu.power_limit_w} W`],
      ]),
    );
  }

  const h = d.host;
  if (h) {
    sections.push(mEl("h3", null, "Memoria y CPU"));
    sections.push(usageMeter("RAM del sistema", h.ram_used_bytes, h.ram_total_bytes));
    sections.push(
      counts([
        ["RAM del proceso", formatBytes(h.process_rss_bytes)],
        ["CPU del proceso", `${Math.round(h.process_cpu_percent)}%`],
        ["CPU del sistema", `${Math.round(h.system_cpu_percent)}%`],
        ["Hilos", h.cpu_threads],
      ]),
    );
  }

  const p = d.performance;
  if (p) {
    sections.push(mEl("h3", null, "Inferencia"));
    sections.push(
      counts([
        ["Inferencias", p.total],
        ["Último minuto", p.per_minute],
        ["Última", formatMs(p.last_ms)],
        ["Promedio", formatMs(p.avg_ms)],
        ["p50", formatMs(p.p50_ms)],
        ["p95", formatMs(p.p95_ms)],
        ["Máxima", formatMs(p.max_ms)],
        ["Activo hace", formatUptime(d.uptime_s)],
      ]),
    );
    sections.push(sparkline(p.recent_ms));
  }

  container.replaceChildren(...sections);
}

function summaryText(d) {
  const parts = [];
  const devices = [...new Set(d.models.map((m) => m.device))];
  if (devices.length) parts.push(devices.join(", "));
  if (d.gpu_memory) parts.push(`VRAM ${formatBytes(d.gpu_memory.total_bytes - d.gpu_memory.free_bytes)} / ${formatBytes(d.gpu_memory.total_bytes)}`);
  else if (d.host) parts.push(`RAM ${formatBytes(d.host.process_rss_bytes)}`);
  if (d.performance && d.performance.p50_ms != null) parts.push(`p50 ${d.performance.p50_ms} ms`);
  return parts.join(" · ") || DASH;
}

function startMetrics(container, summaryEl) {
  let timer = null;

  async function refresh() {
    try {
      const response = await fetch("/api/metrics");
      if (!response.ok) throw new Error(response.statusText);
      const data = await response.json();
      renderMetrics(container, data);
      if (summaryEl) summaryEl.textContent = summaryText(data);
    } catch {
      container.replaceChildren(mEl("p", "muted", "Métricas no disponibles."));
      if (summaryEl) summaryEl.textContent = DASH;
    }
  }

  function start() {
    if (timer === null) timer = setInterval(refresh, METRICS_POLL_MS);
  }

  function stop() {
    clearInterval(timer);
    timer = null;
  }

  // No polling while the tab is in the background.
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      stop();
    } else {
      refresh();
      start();
    }
  });

  refresh();
  if (!document.hidden) start();
}
