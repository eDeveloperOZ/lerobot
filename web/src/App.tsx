import React, { useEffect, useRef, useState } from 'react';

interface Action {
  [key: string]: number;
}

export default function App() {
  const [motorSocket, setMotorSocket] = useState<WebSocket | null>(null);
  const [videoSocket, setVideoSocket] = useState<WebSocket | null>(null);
  const [latency, setLatency] = useState(0);
  const [dropped, setDropped] = useState(0);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const actionRef = useRef<Action>({});

  const connectArm = async () => {
    const port = await navigator.serial.requestPort({});
    const info = port.getInfo();
    const ws = new WebSocket(
      `ws://${location.host}/motors?vid=${info.usbVendorId}&pid=${info.usbProductId}`
    );
    ws.onmessage = (ev) => {
      const m = JSON.parse(ev.data);
      if (m.t) setLatency(performance.now() - m.t);
    };
    ws.onclose = () => setMotorSocket(null);
    setMotorSocket(ws);
  };

  const connectCameras = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    const video = document.createElement('video');
    video.srcObject = stream;
    await video.play();
    const ws = new WebSocket(`ws://${location.host}/video`);
    setVideoSocket(ws);
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext('2d')!;
    const loop = () => {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((b) => b && ws.send(b!), 'image/jpeg');
      requestAnimationFrame(loop);
    };
    loop();
  };

  useEffect(() => {
    const sendLoop = () => {
      if (motorSocket && motorSocket.readyState === WebSocket.OPEN) {
        motorSocket.send(
          JSON.stringify({ t: performance.now(), action: actionRef.current })
        );
      }
      requestAnimationFrame(sendLoop);
    };
    sendLoop();
  }, [motorSocket]);

  const handleKey = (e: KeyboardEvent) => {
    const val = e.type === 'keydown' ? 1 : 0;
    if (e.key === 'w') actionRef.current['w'] = val;
    if (e.key === 'a') actionRef.current['a'] = val;
    if (e.key === 's') actionRef.current['s'] = val;
    if (e.key === 'd') actionRef.current['d'] = val;
  };

  useEffect(() => {
    window.addEventListener('keydown', handleKey);
    window.addEventListener('keyup', handleKey);
    return () => {
      window.removeEventListener('keydown', handleKey);
      window.removeEventListener('keyup', handleKey);
    };
  }, []);

  return (
    <div>
      <h1>LeRobot Browser Teleop</h1>
      <button onClick={connectArm} disabled={!!motorSocket}>
        Connect Arm
      </button>
      <button onClick={connectCameras} disabled={!!videoSocket}>
        Connect Cameras
      </button>
      <p>RTT: {latency.toFixed(1)} ms</p>
      <p>Dropped frames: {dropped}</p>
      <canvas ref={canvasRef} width={320} height={240} />
    </div>
  );
}
