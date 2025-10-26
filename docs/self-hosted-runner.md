## Self-hosted runner (Proxmox) — cloud-init template

This repository now includes a cloud-init user-data template you can use to provision a self-hosted GitHub Actions runner in Proxmox or similar hypervisors.

Files added
- `infra/cloud-init/self-hosted-runner-userdata.yaml` — cloud-init user-data (template). Replace placeholders before use.

Quick overview
- The template creates a `runner` user, installs Docker and basic tools, downloads the latest GitHub Actions runner binary, configures it for a repository, and registers it as a system service.
- You must provide a registration token (short-lived) from GitHub and substitute it into the template before launching the VM.

How to generate a registration token

Repository-level registration token (recommended for repo-scoped runners):

1. Create a PAT with `repo` (or repository admin) scope or use a GitHub App / workflow that can mint tokens.
2. Run the API call to create a registration token:

```bash
# Replace OWNER and REPO with your values and set AUTH_TOKEN to a PAT with the correct scope
curl -sX POST -H "Authorization: token ${AUTH_TOKEN}" \
  "https://api.github.com/repos/OWNER/REPO/actions/runners/registration-token" | jq -r .token
```

Organization-level registration token (for org runners):

```bash
curl -sX POST -H "Authorization: token ${AUTH_TOKEN}" \
  "https://api.github.com/orgs/ORG/actions/runners/registration-token" | jq -r .token
```

Important: registration tokens are short-lived (minutes). Pass them into the cloud-init template when creating the VM. Do NOT commit tokens to the repository.

How to use the cloud-init template in Proxmox

1. Create a VM using a cloud-init enabled template (Debian/Ubuntu cloud image recommended).
2. In the VM's "Cloud-Init" -> "User Data" field, paste the contents of `infra/cloud-init/self-hosted-runner-userdata.yaml` after replacing placeholders:
   - `OWNER/REPO` -> the GitHub repo or `https://github.com/ORG` (for org-level runner adjust config command accordingly)
   - `RUNNER_TOKEN` -> the token you generated with the API
   - `RUNNER_NAME` -> unique name for this runner (e.g. `proxmox-runner-01`)
   - `SSH_PUB_KEY` -> optional public ssh key for the `runner` account
3. Boot the VM. Cloud-init will run the setup and register the runner.

Security notes and best practices
- Use short-lived registration tokens. Automate token creation using a secure PAT or GitHub App (server-side) and inject it into cloud-init at VM-creation time.
- Consider using a dedicated machine image with the runner preinstalled (skip the download step) for faster provisioning.
- Make the runner ephemeral if you want autoscaling: register it on boot and deregister on shutdown.
- For heavy builds, give the VM larger disk and CPU and configure Docker's storage driver to use a large data disk.

Troubleshooting
- Check `/var/log/runner-setup.log` on the VM for cloud-init progress.
- Runner logs are in `/home/runner/actions-runner/_diag`.
- If config fails due to token errors, confirm the token is unexpired and targets the correct repo/org.

If you want, I can:
- provide a version of the cloud-init that fetches a registration token from a secure vault or a small helper service (requires an endpoint/credentials).
- add a systemd unit to automatically re-register the runner if the VM image is cloned (careful with duplicate names).
