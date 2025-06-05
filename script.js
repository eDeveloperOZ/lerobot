let token = null;
let ws = null;
const video = document.querySelector('video');
const latencyCanvas = document.getElementById('latency');
const latencyCtx = latencyCanvas.getContext('2d');
const logElem = document.getElementById('log');

function addLog(msg){
  const lines = logElem.textContent.trim().split('\n');
  if(msg) lines.push(msg);
  while(lines.length>10) lines.shift();
  logElem.textContent = lines.join('\n');
}

async function startCamera(){
  if(ws) return;
  const res = await fetch('/token?device=browser');
  token = (await res.json()).token;
  ws = new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/video?t=${token}`);
  ws.binaryType = 'arraybuffer';
  ws.onmessage = ev=>{
    if(typeof ev.data === 'string'){
      addLog(ev.data);
    }else{
      const view = new DataView(ev.data);
      const sent = view.getFloat64(0, true);
      const rtt = performance.now() - sent;
      latencyCtx.clearRect(0,0,latencyCanvas.width,latencyCanvas.height);
      latencyCtx.fillText(`${rtt.toFixed(1)} ms`, 10, 15);
    }
  };
  const stream = await navigator.mediaDevices.getUserMedia({video:true});
  video.srcObject = stream;
  video.play();
  const off = document.createElement('canvas');
  const offCtx = off.getContext('2d');
  function sendFrame(){
    off.width = video.videoWidth;
    off.height = video.videoHeight;
    offCtx.drawImage(video,0,0);
    off.toBlob(async blob=>{
      const buf = new ArrayBuffer(blob.size+8);
      const view = new DataView(buf);
      view.setFloat64(0, performance.now(), true);
      const arr = new Uint8Array(buf,8);
      arr.set(new Uint8Array(await blob.arrayBuffer()));
      ws.send(buf);
    },'image/jpeg',0.7);
    video.requestVideoFrameCallback(()=>setTimeout(sendFrame,66));
  }
  video.requestVideoFrameCallback(()=>setTimeout(sendFrame,66));
}

document.getElementById('camBtn').onclick=startCamera;

document.getElementById('armBtn').onclick=async()=>{
  try{
    const port = await navigator.serial.requestPort();
    const info = port.getInfo();
    await fetch('/connect_arm',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(info)});
  }catch(err){console.error(err);}
};
