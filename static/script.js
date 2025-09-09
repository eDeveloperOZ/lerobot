// UI Elements
const camBtn = document.getElementById('camBtn');
const armBtn = document.getElementById('armBtn');
const forceArmBtn = document.getElementById('forceArmBtn');
const diagnoseBtn = document.getElementById('diagnoseBtn');
const clearLogBtn = document.getElementById('clearLogBtn');
const configBtn = document.getElementById('configBtn');
const motorConfigPanel = document.getElementById('motorConfigPanel');
const presetStandardBtn = document.getElementById('presetStandard');
const presetFollowerBtn = document.getElementById('presetFollower');
const applyConfigBtn = document.getElementById('applyConfig');
const cancelConfigBtn = document.getElementById('cancelConfig');
const video = document.querySelector('video');
const log = document.getElementById('log');
const actionsBox = document.getElementById('actionsBox');
const diagnosticsResults = document.getElementById('diagnosticsResults');
const armStatus = document.getElementById('armStatus');
const cameraStatus = document.getElementById('cameraStatus');
const inferenceStatus = document.getElementById('inferenceStatus');

// State
let ws = null;
let token = null;
let armConnected = false;
let cameraConnected = false;
let inferenceRunning = false;

// Current motor configuration
let motorConfig = {
  shoulder_pan: 1,
  shoulder_lift: 2, 
  elbow_flex: 3,
  wrist_flex: 4,
  wrist_roll: 5,
  gripper: 6
};

// Known SO100 USB IDs
const SO100_USB = {
  usbVendorId: 6790,  // 1A86 hex = 6790 decimal
  usbProductId: 21971  // 55D3 hex = 21971 decimal
};

function appendLog(msg) {
  const timestamp = new Date().toLocaleTimeString();
  log.textContent += `[${timestamp}] ${msg}\n`;
  log.scrollTop = log.scrollHeight; // Auto-scroll to bottom
}

function updateStatusIndicator(element, status) {
  const indicator = element.previousElementSibling;
  indicator.className = `status-indicator status-${status}`;
}

function updateSystemStatus() {
  // Update arm status
  if (armConnected) {
    armStatus.textContent = 'Connected ✓';
    updateStatusIndicator(armStatus, 'connected');
  } else {
    armStatus.textContent = 'Disconnected';
    updateStatusIndicator(armStatus, 'disconnected');
  }
  
  // Update camera status
  if (cameraConnected) {
    cameraStatus.textContent = 'Connected ✓';
    updateStatusIndicator(cameraStatus, 'connected');
  } else {
    cameraStatus.textContent = 'Disconnected';
    updateStatusIndicator(cameraStatus, 'disconnected');
  }
  
  // Update inference status
  if (inferenceRunning) {
    inferenceStatus.textContent = 'Running ✓';
    updateStatusIndicator(inferenceStatus, 'connected');
  } else {
    inferenceStatus.textContent = 'Not running';
    updateStatusIndicator(inferenceStatus, 'disconnected');
  }
}

async function getToken() {
  const device = 'webcam';
  const res = await fetch('/token?device=' + encodeURIComponent(device));
  const data = await res.json();
  return data.token;
}

function displayDiagnostics(data) {
  if (!data.diagnostics) {
    diagnosticsResults.innerHTML = '';
    return;
  }
  
  const diag = data.diagnostics;
  let html = '<div class="diagnostics">';
  html += `<h4>🔍 Motor Diagnostics Results</h4>`;
  html += `<p><strong>Port:</strong> ${data.port || 'Unknown'}</p>`;
  html += `<p><strong>Motors Found:</strong> ${diag.total_found}/${diag.total_expected}</p>`;
  
  // Motor status grid
  html += '<div class="motor-status">';
  
  // Show found motors
  if (diag.found_motors) {
    diag.found_motors.forEach(motor => {
      html += `<div class="motor-item motor-found">
        <strong>✓ Motor ${motor.id}</strong><br>
        ${motor.name}<br>
        <small>Model: ${motor.model}</small>
      </div>`;
    });
  }
  
  // Show missing motors
  if (diag.missing_motors) {
    diag.missing_motors.forEach(motor => {
      html += `<div class="motor-item motor-missing">
        <strong>✗ Motor ${motor.id}</strong><br>
        ${motor.name}<br>
        <small>Missing</small>
      </div>`;
    });
  }
  
  html += '</div>';
  html += '</div>';
  
  // Add troubleshooting if available
  if (data.troubleshooting) {
    html += '<div class="troubleshooting">';
    html += '<h4>🛠️ Troubleshooting</h4>';
    html += `<p><strong>Likely Cause:</strong> ${data.troubleshooting.likely_cause}</p>`;
    html += '<p><strong>Suggestions:</strong></p>';
    html += '<ul>';
    data.troubleshooting.suggestions.forEach(suggestion => {
      html += `<li>${suggestion}</li>`;
    });
    html += '</ul>';
    html += '</div>';
  }
  
  diagnosticsResults.innerHTML = html;
}

async function diagnoseMotors() {
  diagnoseBtn.disabled = true;
  diagnoseBtn.textContent = '🔍 Diagnosing...';
  
  try {
    appendLog('Running motor diagnostics...');
    
    const res = await fetch('/diagnose_motors', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ...SO100_USB,
        motorConfig: motorConfig
      })
    });
    const data = await res.json();
    
    appendLog('Diagnostics: ' + JSON.stringify(data));
    displayDiagnostics(data);
    
    if (data.status === 'success') {
      const found = data.diagnostics.total_found;
      const expected = data.diagnostics.total_expected;
      
      if (found === expected) {
        appendLog(`✅ All ${found} motors detected! Hardware looks good.`);
      } else if (found > 0) {
        appendLog(`⚠️ Partial detection: ${found}/${expected} motors found. Check connections.`);
      } else {
        appendLog('❌ No motors detected. Check power and connections.');
      }
    }
    
  } catch (e) {
    appendLog('Diagnostics error: ' + e.message);
    diagnosticsResults.innerHTML = `<div class="diagnostics">
      <h4>❌ Diagnostic Error</h4>
      <p>Failed to run diagnostics: ${e.message}</p>
    </div>`;
  } finally {
    diagnoseBtn.disabled = false;
    diagnoseBtn.textContent = '🔍 Diagnose Motors';
  }
}

async function connectCameraAPI() {
  camBtn.disabled = true;
  camBtn.textContent = '📹 Connecting...';
  
  try {
    appendLog('Connecting to camera...');
    
    const res = await fetch('/connect_camera', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({})
    });
    const data = await res.json();
    appendLog('Camera: ' + JSON.stringify(data));
    
    if (data.status === 'connected') {
      cameraConnected = true;
      appendLog(`✅ Camera connected: ${data.camera} (${data.resolution})`);
      
      // Start WebSocket for receiving actions
      token = await getToken();
      ws = new WebSocket(`ws://${window.location.host}/video?t=${token}`);
      
      ws.onopen = () => {
        appendLog('🔗 WebSocket connected for inference');
        inferenceRunning = true;
        updateSystemStatus();
      };
      
      ws.onclose = () => {
        appendLog('🔌 WebSocket closed');
        inferenceRunning = false;
        updateSystemStatus();
      };
      
      ws.onerror = (e) => {
        appendLog('❌ WebSocket error: ' + e.message);
        inferenceRunning = false;
        updateSystemStatus();
      };
      
      ws.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data);
          if (data.error) {
            appendLog('❌ Inference error: ' + data.error);
            actionsBox.innerHTML = `<span style="color: red;">Error: ${data.error}</span>`;
          } else {
            // Show actions in the live box
            const actions = Object.entries(data)
              .map(([k,v]) => `${k}: <b>${typeof v === 'number' ? v.toFixed(3) : v}</b>`)
              .join(' | ');
            actionsBox.innerHTML = `<b>🎯 Live Actions:</b><br>${actions}`;
          }
        } catch (err) {
          appendLog('❌ WebSocket message error: ' + err.message);
        }
      };
      
      camBtn.textContent = '📹 Camera Connected';
      camBtn.className = 'btn-success';
    } else {
      appendLog('❌ Camera connection failed: ' + data.message);
      camBtn.textContent = '📹 Connect Camera';
    }
  } catch (e) {
    appendLog('❌ Camera API error: ' + e.message);
    camBtn.textContent = '📹 Connect Camera';
  } finally {
    camBtn.disabled = false;
    updateSystemStatus();
  }
}

async function connectArm(force = false) {
  const btn = force ? forceArmBtn : armBtn;
  const btnText = force ? '⚡ Force Connect Arm' : '🦾 Connect Arm';
  const connectingText = force ? '⚡ Force Connecting...' : '🦾 Connecting...';
  
  btn.disabled = true;
  btn.textContent = connectingText;
  
  try {
    appendLog(force ? 'Force connecting to SO100 arm (bypassing diagnostics)...' : 'Connecting to SO100 arm...');
    
    const requestBody = force ? 
      { ...SO100_USB, force: true, motorConfig: motorConfig } : 
      { ...SO100_USB, motorConfig: motorConfig };
    const endpoint = force ? '/force_connect_arm' : '/connect_arm';
    
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody)
    });
    const data = await res.json();
    
    appendLog('Arm: ' + JSON.stringify(data));
    
    if (data.status === 'connected') {
      armConnected = true;
      const motorInfo = data.motors_found ? ` with ${data.motors_found} motors` : '';
      appendLog(`✅ SO100 arm connected on ${data.port}${motorInfo}`);
      
      // Update both buttons
      armBtn.textContent = '🦾 Arm Connected';
      armBtn.className = 'btn-success';
      forceArmBtn.textContent = '⚡ Arm Connected';
      forceArmBtn.className = 'btn-success';
      
      if (data.warning) {
        appendLog(`⚠️ ${data.warning}`);
      }
      
      // Display diagnostics if available
      displayDiagnostics(data);
    } else {
      appendLog(`❌ Arm connection failed: ${data.message}`);
      btn.textContent = btnText;
      
      // Display failed diagnostics with troubleshooting
      displayDiagnostics(data);
    }
  } catch (e) {
    appendLog('❌ Arm error: ' + e.message);
    btn.textContent = btnText;
  } finally {
    btn.disabled = false;
    updateSystemStatus();
  }
}

async function forceConnectArm() {
  return connectArm(true);
}

function clearLog() {
  log.textContent = '';
  appendLog('Log cleared');
}

// Motor Configuration Functions
function showMotorConfig() {
  // Load current config into inputs
  document.getElementById('motor1').value = motorConfig.shoulder_pan;
  document.getElementById('motor2').value = motorConfig.shoulder_lift;
  document.getElementById('motor3').value = motorConfig.elbow_flex;
  document.getElementById('motor4').value = motorConfig.wrist_flex;
  document.getElementById('motor5').value = motorConfig.wrist_roll;
  document.getElementById('motor6').value = motorConfig.gripper;
  
  motorConfigPanel.style.display = 'block';
  appendLog('Motor configuration panel opened');
}

function hideMotorConfig() {
  motorConfigPanel.style.display = 'none';
}

function setMotorPreset(preset) {
  if (preset === 'standard') {
    // Standard SO100 configuration (1-6)
    document.getElementById('motor1').value = 1;
    document.getElementById('motor2').value = 2;
    document.getElementById('motor3').value = 3;
    document.getElementById('motor4').value = 4;
    document.getElementById('motor5').value = 5;
    document.getElementById('motor6').value = 6;
    appendLog('Applied standard motor configuration (1-6)');
  } else if (preset === 'follower') {
    // Follower arm configuration (7-12)
    document.getElementById('motor1').value = 7;
    document.getElementById('motor2').value = 8;
    document.getElementById('motor3').value = 9;
    document.getElementById('motor4').value = 10;
    document.getElementById('motor5').value = 11;
    document.getElementById('motor6').value = 12;
    appendLog('Applied follower motor configuration (7-12)');
  }
}

function applyMotorConfig() {
  // Read values from inputs
  motorConfig.shoulder_pan = parseInt(document.getElementById('motor1').value);
  motorConfig.shoulder_lift = parseInt(document.getElementById('motor2').value);
  motorConfig.elbow_flex = parseInt(document.getElementById('motor3').value);
  motorConfig.wrist_flex = parseInt(document.getElementById('motor4').value);
  motorConfig.wrist_roll = parseInt(document.getElementById('motor5').value);
  motorConfig.gripper = parseInt(document.getElementById('motor6').value);
  
  appendLog(`Motor configuration updated: [${motorConfig.shoulder_pan}, ${motorConfig.shoulder_lift}, ${motorConfig.elbow_flex}, ${motorConfig.wrist_flex}, ${motorConfig.wrist_roll}, ${motorConfig.gripper}]`);
  
  // Hide panel
  hideMotorConfig();
  
  // Save to localStorage for persistence
  localStorage.setItem('motorConfig', JSON.stringify(motorConfig));
  
  appendLog('✅ Motor configuration saved! Reconnect arm to use new settings.');
}

// Load saved motor configuration on startup
function loadSavedMotorConfig() {
  const saved = localStorage.getItem('motorConfig');
  if (saved) {
    try {
      motorConfig = JSON.parse(saved);
      appendLog(`Loaded saved motor configuration: [${motorConfig.shoulder_pan}, ${motorConfig.shoulder_lift}, ${motorConfig.elbow_flex}, ${motorConfig.wrist_flex}, ${motorConfig.wrist_roll}, ${motorConfig.gripper}]`);
    } catch (e) {
      appendLog('Failed to load saved motor config, using defaults');
    }
  }
}

// Event listeners
camBtn.addEventListener('click', connectCameraAPI);
armBtn.addEventListener('click', () => connectArm(false));
forceArmBtn.addEventListener('click', forceConnectArm);
diagnoseBtn.addEventListener('click', diagnoseMotors);
clearLogBtn.addEventListener('click', clearLog);

// Motor configuration event listeners
configBtn.addEventListener('click', showMotorConfig);
presetStandardBtn.addEventListener('click', () => setMotorPreset('standard'));
presetFollowerBtn.addEventListener('click', () => setMotorPreset('follower'));
applyConfigBtn.addEventListener('click', applyMotorConfig);
cancelConfigBtn.addEventListener('click', hideMotorConfig);

// Initialize
loadSavedMotorConfig();
updateSystemStatus();
appendLog('🚀 LeRobot SO100 Interface loaded');

