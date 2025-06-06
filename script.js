const cameraBtn = document.getElementById('cameraBtn');
const armBtn = document.getElementById('armBtn');
const video = document.getElementById('preview');
const canvas = document.getElementById('latency');
const log = document.getElementById('log');
let ws;
let token;
let lastSend = 0;

async function getToken() {
  const res = await fetch('/token?device=browser');
  const data = await res.json();
  token = data.token;
}

function logAction(txt) {
  const li = document.createElement('li');
  li.textContent = txt;
  log.appendChild(li);
  while (log.children.length > 10) log.removeChild(log.firstChild);
}

async function startCamera() {
  await getToken();
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
  const url = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/video?t=${token}`;
  ws = new WebSocket(url);
  ws.binaryType = 'arraybuffer';
  ws.onmessage = (ev) => {
    if (typeof ev.data !== 'string') {
      const latency = performance.now() - lastSend;
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillText(latency.toFixed(0) + ' ms', 10, 20);
    } else {
      logAction(ev.data);
    }
  };

  const off = document.createElement('canvas');
  const ctx = off.getContext('2d');

  function sendFrame() {
    video.requestVideoFrameCallback(() => {
      off.width = video.videoWidth;
      off.height = video.videoHeight;
      ctx.drawImage(video, 0, 0, off.width, off.height);
      off.toBlob((blob) => {
        if (ws.readyState === WebSocket.OPEN) {
          lastSend = performance.now();
          ws.send(blob);
        }
      }, 'image/jpeg', 0.7);
      setTimeout(sendFrame, 66); // ~15 fps
    });
  }
  sendFrame();
}

async function connectArm() {
  const port = await navigator.serial.requestPort();
  const info = port.getInfo();
  await fetch(`/connect_arm?t=${token}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(info)
  });
}

cameraBtn.onclick = startCamera;
armBtn.onclick = connectArm;
