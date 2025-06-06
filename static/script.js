const camBtn = document.getElementById('camBtn');
const armBtn = document.getElementById('armBtn');
const video = document.querySelector('video');
const log = document.getElementById('log');
const actionsBox = document.getElementById('actionsBox');

let ws = null;
let token = null;

function appendLog(msg) {
  log.textContent += msg + '\n';
}

async function getToken() {
  const device = 'webcam';
  const res = await fetch('/token?device=' + encodeURIComponent(device));
  const data = await res.json();
  return data.token;
}

async function connectCameraAPI() {
  try {
    const res = await fetch('/connect_camera', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({})
    });
    const data = await res.json();
    appendLog('Connect Camera: ' + JSON.stringify(data));
    
    if (data.status === 'connected') {
      // Start WebSocket for receiving actions
      token = await getToken();
      ws = new WebSocket(`ws://${window.location.host}/video?t=${token}`);
      ws.onopen = () => appendLog('WebSocket connected for inference.');
      ws.onclose = () => appendLog('WebSocket closed.');
      ws.onerror = (e) => appendLog('WebSocket error: ' + e.message);
      ws.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data);
          if (data.error) {
            appendLog('Error: ' + data.error);
          } else {
            // Show actions in the live box
            actionsBox.innerHTML = '<b>Live Actions</b><br>' + 
              Object.entries(data).map(([k,v]) => `${k}: <b>${typeof v === 'number' ? v.toFixed(3) : v}</b>`).join(' | ');
          }
        } catch (err) {
          appendLog('WebSocket message error: ' + err.message);
        }
      };
    }
  } catch (e) {
    appendLog('Camera API error: ' + e.message);
  }
}

async function connectArm() {
  try {
    // Use the known Feetech SO100 vendor/product IDs
    const res = await fetch('/connect_arm', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        usbVendorId: 6790,  // 1A86 hex = 6790 decimal
        usbProductId: 21971  // 55D3 hex = 21971 decimal
      })
    });
    const data = await res.json();
    appendLog('Connect Arm: ' + JSON.stringify(data));
  } catch (e) {
    appendLog('Arm error: ' + e.message);
  }
}

camBtn.addEventListener('click', connectCameraAPI);
armBtn.addEventListener('click', connectArm);
