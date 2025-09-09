#!/usr/bin/env python3
"""
Diagnostic script for SO100 Feetech motor connection issues.
This script helps identify and troubleshoot problems when only some motors are detected.
"""

import logging
import sys
import time
from pathlib import Path

import serial.tools.list_ports as lp
from lerobot.common.motors.feetech import FeetechMotorsBus
from lerobot.common.motors import Motor, MotorNormMode

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def find_available_ports():
    """Find all available serial ports."""
    ports = list(lp.comports())
    print("\n=== Available Serial Ports ===")
    for p in ports:
        vid = getattr(p, 'vid', None)
        pid = getattr(p, 'pid', None)
        print(f"Port: {p.device}")
        print(f"  Description: {p.description}")
        print(f"  USB VID:PID: {vid:04x}:{pid:04x}" if vid and pid else "  USB VID:PID: N/A")
        print(f"  Manufacturer: {getattr(p, 'manufacturer', 'N/A')}")
        print()
    return [p.device for p in ports]

def scan_motors_on_port(port, baudrates=None):
    """Scan for motors on a specific port at different baudrates."""
    if baudrates is None:
        # Common Feetech baudrates
        baudrates = [1000000, 500000, 57600, 115200, 1200000]
    
    print(f"\n=== Scanning Motors on Port: {port} ===")
    
    # Create a minimal FeetechMotorsBus for scanning
    dummy_motors = {"test": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100)}
    
    try:
        bus = FeetechMotorsBus(port=port, motors=dummy_motors)
        bus._connect(handshake=False)
        
        found_motors = {}
        
        for baudrate in baudrates:
            print(f"\nTesting baudrate: {baudrate}")
            try:
                bus.set_baudrate(baudrate)
                time.sleep(0.1)  # Allow baudrate to settle
                
                # Try to ping all possible motor IDs
                motors_at_baudrate = {}
                for motor_id in range(1, 21):  # Check IDs 1-20
                    try:
                        model_number = bus.ping(motor_id, num_retry=1)
                        if model_number is not None:
                            motors_at_baudrate[motor_id] = model_number
                            print(f"  Found motor ID {motor_id} with model {model_number}")
                    except Exception as e:
                        # Silent fail for individual motor pings
                        pass
                
                if motors_at_baudrate:
                    found_motors[baudrate] = motors_at_baudrate
                    print(f"  Total motors found at {baudrate}: {len(motors_at_baudrate)}")
                else:
                    print(f"  No motors found at {baudrate}")
                    
            except Exception as e:
                print(f"  Error at baudrate {baudrate}: {e}")
        
        bus.disconnect(disable_torque=False)
        return found_motors
        
    except Exception as e:
        print(f"Failed to connect to port {port}: {e}")
        return {}

def diagnose_so100_issue(port):
    """Specific diagnostics for SO100 arm configuration."""
    print(f"\n=== SO100 Diagnostic Analysis for Port: {port} ===")
    
    # Expected SO100 configuration
    expected_motors = {
        "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
        "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
        "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
        "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
        "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
        "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    }
    
    try:
        # Create bus with expected configuration
        bus = FeetechMotorsBus(port=port, motors=expected_motors)
        bus._connect(handshake=False)
        
        # Test at common SO100 baudrates
        common_baudrates = [1000000, 500000]
        
        print("\nTesting SO100 expected configuration...")
        
        for baudrate in common_baudrates:
            print(f"\nBaudrate: {baudrate}")
            bus.set_baudrate(baudrate)
            time.sleep(0.1)
            
            found_ids = []
            missing_ids = []
            
            for motor_name, motor in expected_motors.items():
                try:
                    model_number = bus.ping(motor.id, num_retry=2)
                    if model_number is not None:
                        found_ids.append(motor.id)
                        print(f"  ✓ Motor {motor.id} ({motor_name}): Found (model {model_number})")
                    else:
                        missing_ids.append(motor.id)
                        print(f"  ✗ Motor {motor.id} ({motor_name}): Missing")
                except Exception as e:
                    missing_ids.append(motor.id)
                    print(f"  ✗ Motor {motor.id} ({motor_name}): Error - {e}")
            
            print(f"\nSummary for baudrate {baudrate}:")
            print(f"  Found: {found_ids}")
            print(f"  Missing: {missing_ids}")
            
            if len(found_ids) == 6:
                print("  ✓ All SO100 motors detected!")
                break
            elif len(found_ids) > 0:
                print(f"  ⚠ Partial detection: {len(found_ids)}/6 motors found")
                
                # Additional diagnostic for partial detection
                print("\n  Checking for communication chain issues...")
                
                # Check if missing IDs might be at different addresses
                print("  Scanning for motors with unexpected IDs...")
                unexpected_motors = {}
                for test_id in range(1, 21):
                    if test_id not in [m.id for m in expected_motors.values()]:
                        try:
                            model_number = bus.ping(test_id, num_retry=1)
                            if model_number is not None:
                                unexpected_motors[test_id] = model_number
                                print(f"    Found unexpected motor at ID {test_id} (model {model_number})")
                        except:
                            pass
                
                if unexpected_motors:
                    print("  ⚠ Found motors with unexpected IDs. This might indicate ID configuration issues.")
                else:
                    print("  No motors found at unexpected IDs.")
        
        bus.disconnect(disable_torque=False)
        
    except Exception as e:
        print(f"Failed during SO100 diagnostics: {e}")

def provide_troubleshooting_tips(scan_results):
    """Provide troubleshooting recommendations based on scan results."""
    print("\n=== Troubleshooting Recommendations ===")
    
    total_motors_found = sum(len(motors) for motors in scan_results.values())
    
    if total_motors_found == 0:
        print("""
🔴 NO MOTORS DETECTED
Possible causes:
1. Power supply issues - Check if motors are powered on
2. Wrong port - Verify you're using the correct USB port
3. Cable connection - Check USB cable and motor daisy-chain connections
4. Driver issues - Ensure proper USB-to-serial drivers are installed

Next steps:
- Verify power LED on motors are lit
- Try a different USB cable
- Check if device appears in system's device manager
- Try a different USB port
""")
    
    elif total_motors_found == 1:
        print("""
🟡 ONLY ONE MOTOR DETECTED (matching your error)
Possible causes:
1. Daisy-chain break - Communication chain is broken after first motor
2. Power issues - Insufficient power for all motors
3. Motor ID conflicts - Multiple motors might have same ID
4. Faulty motor in chain - One motor preventing communication to others
5. Loose connections - Check all connectors in the daisy chain

Next steps:
- Check physical connections between motors
- Verify power supply can handle all 6 motors (check amperage)
- Try powering motors individually to isolate faulty units
- Check if motor IDs are properly configured (1,2,3,4,5,6)
""")
        
        found_baudrate = next(iter(scan_results.keys()))
        found_id = next(iter(scan_results[found_baudrate].keys()))
        print(f"Motor found: ID {found_id} at baudrate {found_baudrate}")
        
    elif 1 < total_motors_found < 6:
        print(f"""
🟡 PARTIAL DETECTION ({total_motors_found}/6 motors found)
This suggests the daisy-chain is working but incomplete.

Possible causes:
1. Break in daisy-chain after detected motors
2. Power drop - Later motors not getting enough power
3. Faulty motor preventing further communication
4. Loose connection in middle of chain

Next steps:
- Identify which motors are missing
- Check connections after the last detected motor
- Test each motor individually
- Verify power distribution
""")
    
    elif total_motors_found == 6:
        print("""
🟢 ALL 6 MOTORS DETECTED
The hardware seems fine. The issue might be:
1. Calibration problems
2. Software configuration mismatch
3. Timing issues during connection

Try reconnecting with the server.
""")
    
    elif total_motors_found > 6:
        print(f"""
🟡 MORE THAN 6 MOTORS DETECTED ({total_motors_found} found)
This suggests ID conflicts or unexpected motors on the bus.

Possible causes:
1. Motor ID conflicts - Multiple motors with same IDs
2. Extra motors connected
3. Motors not properly configured for SO100

Next steps:
- Check motor ID configuration
- Ensure only SO100 motors are connected
- Reconfigure motor IDs if needed
""")

def main():
    print("SO100 Feetech Motor Diagnostic Tool")
    print("===================================")
    
    # Find available ports
    ports = find_available_ports()
    
    if not ports:
        print("No serial ports found!")
        return
    
    # Auto-detect SO100 port or let user choose
    target_port = None
    
    # Look for common SO100 USB identifiers
    for p in lp.comports():
        device_info = f"{p.description} {getattr(p, 'manufacturer', '')}"
        if any(keyword in device_info.lower() for keyword in ['ch340', 'cp210', 'ftdi', 'silicon']):
            print(f"Potential SO100 port detected: {p.device} ({device_info})")
            target_port = p.device
            break
    
    if not target_port:
        print("\nNo obvious SO100 port detected. Available ports:")
        for i, port in enumerate(ports):
            print(f"{i+1}. {port}")
        
        try:
            choice = int(input(f"\nChoose port (1-{len(ports)}): ")) - 1
            target_port = ports[choice]
        except (ValueError, IndexError):
            print("Invalid choice. Using first port.")
            target_port = ports[0]
    
    print(f"\nUsing port: {target_port}")
    
    # Scan for motors
    scan_results = scan_motors_on_port(target_port)
    
    # Run SO100-specific diagnostics
    if scan_results:
        diagnose_so100_issue(target_port)
    
    # Provide troubleshooting recommendations
    provide_troubleshooting_tips(scan_results)
    
    print("\n=== Diagnostic Complete ===")
    if scan_results:
        print("Run this script again after making hardware changes to verify fixes.")
    else:
        print("No motors detected. Check hardware connections and power supply.")

if __name__ == "__main__":
    main() 