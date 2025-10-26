#!/usr/bin/env bash
# Create a cloud-init based VM on Proxmox for a GitHub Actions self-hosted runner.
# Usage: sudo ./create-runner-vm.sh <VMID> <VM_NAME> [STORAGE=local] [IMAGE_URL]
# Example:
#   sudo ./create-runner-vm.sh 210 github-runner-01 local https://cloud-images.ubuntu.com/jammy/current/jammy-server-cloudimg-amd64.img

set -euo pipefail
VMID=${1:-}
VMNAME=${2:-github-runner}
STORAGE=${3:-local}
IMG_URL=${4:-https://cloud-images.ubuntu.com/jammy/current/jammy-server-cloudimg-amd64.img}
MEM=${MEM:-4096}
CORES=${CORES:-2}
DISK_SIZE=${DISK_SIZE:-16G}
BRIDGE=${BRIDGE:-vmbr0}

if [ -z "$VMID" ]; then
  echo "Usage: $0 <VMID> <VM_NAME> [STORAGE] [IMAGE_URL]"
  exit 1
fi

if ! command -v qm >/dev/null 2>&1; then
  echo "This script must run on a Proxmox host with 'qm' available." >&2
  exit 1
fi

# Prepare paths
IMG_NAME="vm-${VMID}-cloudimg.img"
IMG_PATH="/var/lib/vz/template/iso/${IMG_NAME}"

echo "Downloading cloud image to ${IMG_PATH} (if not present)"
mkdir -p /var/lib/vz/template/iso
if [ ! -f "$IMG_PATH" ]; then
  wget -O "$IMG_PATH" "$IMG_URL"
fi

echo "Creating VM $VMID ($VMNAME)"
qm create $VMID --name $VMNAME --memory $MEM --cores $CORES --net0 virtio,bridge=${BRIDGE}

# Import the disk into storage as scsi0
echo "Importing disk to storage ${STORAGE}"
qm importdisk $VMID "$IMG_PATH" $STORAGE --format qcow2

# Attach disk and set up cloud-init drive
qm set $VMID --scsihw virtio-scsi-pci --scsi0 ${STORAGE}:vm-${VMID}-disk-0
qm set $VMID --ide2 ${STORAGE}:cloudinit
qm set $VMID --boot c --bootdisk scsi0
qm set $VMID --serial0 socket --vga serial0

# Set VM to use cloud-init user data via storage snippets
SNIPPET_NAME="user-data-${VMID}.yaml"
SNIPPET_PATH="/tmp/${SNIPPET_NAME}"
cat > "$SNIPPET_PATH" <<'EOF'
#cloud-config
password: runner
chpasswd: { expire: False }
ssh_pwauth: True
users:
  - name: runner
    gecos: Runner User
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
    lock_passwd: false
runcmd:
  - [ bash, -lc, 'mkdir -p /home/runner/actions-runner' ]
EOF

# Upload snippet to Proxmox storage 'local:snippets'
if ! pvesm available >/dev/null 2>&1; then
  echo "Warning: pvesm not available; attempting pvesm via /usr/sbin/pvesm"
fi
STORAGE_SNIPPETS=${STORAGE}:snippets
echo "Uploading cloud-init snippet to ${STORAGE_SNIPPETS}/${SNIPPET_NAME}"
pvesm upload $STORAGE "$SNIPPET_PATH" snippets/$SNIPPET_NAME

# Point VM at the uploaded user-data
qm set $VMID --cicustom user=$STORAGE:snippets/${SNIPPET_NAME}

# Set cloud-init defaults
qm set $VMID --ciuser runner --cipassword runner

# Resize disk if needed (handled at import with specified size, optionally resize)
# Start the VM
qm start $VMID

echo "VM $VMID created and started. Use 'qm console $VMID' to view the console."