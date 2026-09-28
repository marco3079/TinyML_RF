#pragma once

const char INDEX_HTML[] PROGMEM = R"HTML(
<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>XIAO Face Monitor</title>
  <style>
    :root { --ink:#17211b; --paper:#f3f0e8; --accent:#de4d2f; --green:#23775b; --line:#cbc6b9; }
    * { box-sizing:border-box; }
    body { margin:0; color:var(--ink); background:radial-gradient(circle at 15% 20%,#fff 0 1px,transparent 2px) 0 0/22px 22px,var(--paper); font-family:Georgia,"Times New Roman",serif; }
    main { min-height:100vh; display:grid; grid-template-columns:minmax(0,1fr) 290px; }
    .viewer { padding:28px; display:grid; place-items:center; min-width:0; }
    .frame { width:min(100%,900px); position:relative; border:1px solid var(--ink); background:#161a18; box-shadow:10px 10px 0 #d7d1c3; aspect-ratio:1/1; }
    #stream { width:100%; height:100%; display:block; object-fit:contain; }
    .live { position:absolute; top:14px; left:14px; padding:7px 10px; background:var(--accent); color:white; font:700 12px/1 sans-serif; letter-spacing:0; }
    aside { border-left:1px solid var(--line); padding:34px 26px; background:rgba(255,255,255,.45); }
    h1 { margin:0 0 8px; font-size:34px; line-height:1; font-weight:500; }
    .board { margin:0 0 42px; color:#576159; font:13px/1.4 sans-serif; }
    dl { margin:0; }
    .metric { padding:18px 0; border-top:1px solid var(--line); }
    dt { color:#687168; font:11px/1 sans-serif; text-transform:uppercase; }
    dd { margin:7px 0 0; font-size:27px; }
    .status { margin-top:36px; display:flex; gap:8px; align-items:center; font:13px/1.3 sans-serif; }
    .dot { width:9px; height:9px; border-radius:50%; background:var(--green); }
    @media(max-width:760px){ main{grid-template-columns:1fr}.viewer{padding:18px}aside{border-left:0;border-top:1px solid var(--line);padding:24px}.frame{box-shadow:6px 6px 0 #d7d1c3}h1{font-size:28px}.board{margin-bottom:24px} }
  </style>
</head>
<body>
<main>
  <section class="viewer"><div class="frame"><img id="stream" alt="Camera ao vivo"><span class="live">AO VIVO</span></div></section>
  <aside>
    <h1>Face Monitor</h1><p class="board">XIAO ESP32S3 Sense + ESP-DL</p>
    <dl>
      <div class="metric"><dt>Faces no quadro</dt><dd id="faces">-</dd></div>
      <div class="metric"><dt>Candidatos</dt><dd id="candidates">-</dd></div>
      <div class="metric"><dt>Inferencia</dt><dd><span id="latency">-</span> ms</dd></div>
      <div class="metric"><dt>Taxa do stream</dt><dd><span id="fps">-</span> fps</dd></div>
    </dl>
    <div class="status"><span class="dot"></span><span id="status">Conectando ao dispositivo</span></div>
  </aside>
</main>
<script>
  const host = location.hostname;
  const facesElement = document.querySelector('#faces');
  const candidatesElement = document.querySelector('#candidates');
  const latencyElement = document.querySelector('#latency');
  const fpsElement = document.querySelector('#fps');
  const statusElement = document.querySelector('#status');
  document.querySelector('#stream').src = `http://${host}:81/stream`;
  async function update(){
    try {
      const data = await fetch('/api/status', {cache:'no-store'}).then(r => r.json());
      facesElement.textContent=data.faces;
      candidatesElement.textContent=data.candidates;
      latencyElement.textContent=data.inference_ms;
      fpsElement.textContent=data.fps.toFixed(1);
      statusElement.textContent=data.detector+' ativo';
    } catch (_) { statusElement.textContent='Sem telemetria'; }
  }
  update(); setInterval(update,1000);
</script>
</body>
</html>
)HTML";
