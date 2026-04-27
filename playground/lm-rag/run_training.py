"""
Foolproof Training Pipeline — 5 stages, each gates the next.

Iron Rule: Pod dies ONLY after model is verified locally. No exceptions.

Stage 1: CREATE POD (persistent volume)
Stage 2: WAIT FOR SSH (TCP test + buffer)
Stage 3: TRAIN (upload, install, run)
Stage 4: DOWNLOAD (rsync + verify size + verify encoder + verify output)
Stage 5: TERMINATE (only after all verification passes)
"""

import os
import sys
import time
import socket
import subprocess
import runpod
from runpod.cli.utils.ssh_cmd import SSHConnection

# Load API key
with open(os.path.expanduser('~/webmind-research/playground/create_gpu_pod.py')) as f:
    for line in f:
        if 'api_key' in line and '=' in line:
            runpod.api_key = line.split('"')[1]
            break

TRAIN_SCRIPT = 'train_smollm.py'
SAVE_DIR = 'smollm_orchestrator_v9'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NTFY_TOPIC = 'lm-rag-train'  # https://ntfy.sh/lm-rag-train


def notify(msg, priority='default'):
    """Send push notification via ntfy.sh."""
    try:
        import urllib.request
        req = urllib.request.Request(
            f'https://ntfy.sh/{NTFY_TOPIC}',
            data=msg.encode(),
            headers={'Priority': priority, 'Title': 'LM-RAG Training'},
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass  # don't let notification failure break training


# ── Stage 1: CREATE POD ─────────────────────────────────────────
def create_pod():
    """Create GPU pod with persistent volume."""
    gpus = [
        'NVIDIA A100-SXM4-80GB',
        'NVIDIA A100 80GB PCIe',
        'NVIDIA GeForce RTX 4090',
        'NVIDIA GeForce RTX 3090',
    ]
    for gpu in gpus:
        try:
            pod = runpod.create_pod(
                name='LM-RAG-Train',
                image_name='runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04',
                gpu_type_id=gpu,
                volume_in_gb=30,
                container_disk_in_gb=20,
                ports='22/tcp',
            )
            print(f'[Stage 1] Got {gpu}! Pod: {pod["id"]}')
            return pod['id']
        except Exception:
            print(f'[Stage 1] {gpu}: unavailable')
    raise RuntimeError('No GPUs available')


# ── Stage 2: WAIT FOR SSH ───────────────────────────────────────
def wait_for_ssh(pod_id, timeout=300):
    """Wait for pod SSH to actually accept connections."""
    # Phase 1: RunPod API reports ports
    print('[Stage 2] Waiting for RunPod to assign ports...')
    ssh_ip, ssh_port = None, None
    for _ in range(timeout // 10):
        pods = runpod.get_pods()
        for pod in pods:
            if pod['id'] == pod_id:
                runtime = pod.get('runtime', {})
                if runtime and runtime.get('ports'):
                    for p in runtime['ports']:
                        if p.get('privatePort') == 22:
                            ssh_ip = p['ip']
                            ssh_port = p['publicPort']
                    if ssh_ip:
                        break
        if ssh_ip:
            break
        time.sleep(10)

    if not ssh_ip:
        print('[Stage 2] FAILED — no ports after timeout')
        return False

    print(f'[Stage 2] Ports assigned: {ssh_ip}:{ssh_port}')

    # Phase 2: TCP connection test
    print('[Stage 2] Testing TCP connection to SSH port...')
    for i in range(30):
        try:
            s = socket.create_connection((ssh_ip, ssh_port), timeout=5)
            s.close()
            print(f'[Stage 2] SSH port accepting connections after {i * 5}s')
            break
        except (socket.timeout, ConnectionRefusedError, OSError):
            time.sleep(5)
    else:
        print('[Stage 2] FAILED — SSH port never accepted connections')
        return False

    # Phase 3: Extra buffer for sshd to fully initialize
    print('[Stage 2] Waiting 10s for sshd to fully start...')
    time.sleep(10)
    print('[Stage 2] Ready.')
    return True


# ── Stage 3: TRAIN ──────────────────────────────────────────────
def train(pod_id):
    """Upload script, install deps, run training."""
    script_path = os.path.join(SCRIPT_DIR, TRAIN_SCRIPT)

    print(f'[Stage 3] Connecting to pod {pod_id}...')
    with SSHConnection(pod_id=pod_id) as ssh:
        print('[Stage 3] Uploading training scripts...')
        ssh.put_file(script_path, f'/workspace/{TRAIN_SCRIPT}')
        # Upload dependencies (data builder + model class)
        for extra in ['train_orchestrator.py', 'minilm_reasoning.py']:
            extra_path = os.path.join(SCRIPT_DIR, extra)
            if os.path.exists(extra_path):
                ssh.put_file(extra_path, f'/workspace/{extra}')

        print('[Stage 3] Installing deps + running training...')
        ssh.run_commands([
            'pip install transformers datasets sentencepiece accelerate',
            f'cd /workspace && python {TRAIN_SCRIPT}',
        ])

    print('[Stage 3] Training complete.')
    notify('Training complete. Starting download.')


# ── Stage 4: DOWNLOAD + VERIFY ──────────────────────────────────
def download_and_verify(pod_id):
    """Download model via rsync, verify size + encoder + output."""
    local_save = os.path.join(SCRIPT_DIR, SAVE_DIR)
    os.makedirs(local_save, exist_ok=True)

    # Get SSH connection details for rsync
    print(f'[Stage 4] Connecting for download...')
    with SSHConnection(pod_id=pod_id) as ssh:
        pod_ip = ssh.pod_ip
        pod_port = ssh.pod_port
        key_file = ssh.key_file

        # List model files on pod (T5-base may have sharded weights)
        ssh.run_commands([f'ls -lh /workspace/{SAVE_DIR}/'])

    # Download via scp — discover files dynamically
    # T5-small: model.safetensors (~242MB)
    # T5-base: model.safetensors (~892MB) or model-00001-of-*.safetensors (sharded)
    base_files = ['config.json', 'generation_config.json',
                  'tokenizer.json', 'tokenizer_config.json']

    # Check for sharded vs single model file
    scp_test = subprocess.run([
        'ssh', '-o', 'StrictHostKeyChecking=no', '-o', 'LogLevel=ERROR',
        '-p', str(pod_port), '-i', key_file,
        f'root@{pod_ip}',
        f'ls /workspace/{SAVE_DIR}/model*.safetensors /workspace/{SAVE_DIR}/model.safetensors.index.json 2>/dev/null',
    ], capture_output=True, text=True)
    model_files = [os.path.basename(f.strip()) for f in scp_test.stdout.strip().split('\n') if f.strip()]
    if not model_files:
        model_files = ['model.safetensors']  # fallback
    files = base_files + model_files
    print(f'[Stage 4] Files to download: {files}')

    for attempt in range(3):
        print(f'[Stage 4] scp attempt {attempt + 1}/3...')
        all_ok = True
        for fname in files:
            scp_cmd = [
                'scp',
                '-o', 'StrictHostKeyChecking=no',
                '-o', 'LogLevel=ERROR',
                '-P', str(pod_port),
                '-i', key_file,
                f'root@{pod_ip}:/workspace/{SAVE_DIR}/{fname}',
                os.path.join(local_save, fname),
            ]
            result = subprocess.run(scp_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f'[Stage 4] scp failed for {fname}: {result.stderr[:100]}')
                all_ok = False
                break
            else:
                size = os.path.getsize(os.path.join(local_save, fname))
                print(f'[Stage 4]   {fname}: {size:,} bytes')

        if all_ok:
            print(f'[Stage 4] scp succeeded on attempt {attempt + 1}')
            break
        elif attempt < 2:
            time.sleep(10)
    else:
        print('[Stage 4] FAILED — all 3 scp attempts failed')
        return False

    # Verify
    return verify_model(local_save)


def verify_model(local_dir):
    """Verify downloaded model: size + encoder + generates output."""
    model_path = os.path.join(local_dir, 'model.safetensors')

    # Check 1: file exists and size
    # Check for model files (single or sharded)
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / 1e6
    else:
        # Check for sharded files
        import glob
        shards = glob.glob(os.path.join(local_dir, 'model-*.safetensors'))
        if shards:
            size_mb = sum(os.path.getsize(s) for s in shards) / 1e6
        else:
            print('[Stage 4] VERIFY FAILED — no model files found')
            return False

    if size_mb < 200:
        print(f'[Stage 4] VERIFY FAILED — model only {size_mb:.1f}MB (expected ~242MB+)')
        return False
    print(f'[Stage 4] Size OK: {size_mb:.1f}MB')

    # Check 2: encoder not collapsed
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    model = T5ForConditionalGeneration.from_pretrained(local_dir)
    enc_zeros = sum(1 for n, p in model.named_parameters()
                    if 'encoder' in n and p.data.abs().sum() == 0)
    enc_total = sum(1 for n, _ in model.named_parameters() if 'encoder' in n)
    if enc_zeros > enc_total * 0.5:
        print(f'[Stage 4] VERIFY FAILED — encoder collapsed: {enc_zeros}/{enc_total} zero')
        return False
    print(f'[Stage 4] Encoder OK: {enc_zeros}/{enc_total} zero params')

    # Check 3: model generates non-empty output
    tok = T5Tokenizer.from_pretrained(local_dir)
    ids = tok("route: What is the capital of France?", return_tensors='pt').input_ids
    out = model.generate(ids, max_new_tokens=20)
    result = tok.decode(out[0], skip_special_tokens=True)
    if not result.strip():
        print('[Stage 4] VERIFY FAILED — model generates empty output')
        return False
    print(f'[Stage 4] Output OK: "{result}"')

    print('[Stage 4] ALL VERIFICATIONS PASSED')
    return True


# ── Stage 5: TERMINATE (only on success) ────────────────────────
def main():
    # Stage 1
    pod_id = create_pod()

    # Stage 2
    if not wait_for_ssh(pod_id):
        print(f'\n*** Pod {pod_id} is alive but SSH failed ***')
        print(f'*** NOT terminating — persistent volume preserves data ***')
        return

    # Stage 3
    try:
        train(pod_id)
    except Exception as e:
        print(f'\n[Stage 3] Training error: {e}')
        print(f'*** Pod {pod_id} is alive — model may be on persistent volume ***')
        print(f'*** Check: runpod SSH, then ls /workspace/{SAVE_DIR}/ ***')
        return

    # Stage 4
    verified = download_and_verify(pod_id)

    # Stage 5
    if verified:
        print(f'\n[Stage 5] All checks passed. Terminating pod {pod_id}...')
        runpod.terminate_pod(pod_id)
        print('[Stage 5] Done. Model is safe locally.')
        notify('v9 DONE — model verified and downloaded. Pod terminated.', priority='high')
    else:
        print(f'\n*** VERIFICATION FAILED — NOT terminating pod {pod_id} ***')
        print(f'*** Model is safe on persistent volume at /workspace/{SAVE_DIR}/ ***')
        print(f'*** Fix the issue, then manually: ***')
        print(f'***   python -c "import runpod; runpod.terminate_pod(\'{pod_id}\')" ***')
        notify(f'v9 FAILED — verification failed. Pod {pod_id} still alive.', priority='urgent')


if __name__ == '__main__':
    main()
