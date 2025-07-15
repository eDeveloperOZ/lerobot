import os
import runpod

def handler(job):
    """
    The handler for the Runpod serverless worker.
    It returns the public IP and the assigned TCP port.
    """
    public_ip = os.environ.get('RUNPOD_PUBLIC_IP')
    tcp_port = os.environ.get('RUNPOD_TCP_PORT_8765')
    print(f"Public IP: {public_ip}, TCP Port: {tcp_port}")
    return {
        "ip": public_ip,
        "port": tcp_port
    }

if __name__ == '__main__':
    # Start the Runpod serverless worker in the main process.
    runpod.serverless.start({"handler": handler})
