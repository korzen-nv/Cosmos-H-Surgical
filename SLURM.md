## Remote Cluster Access

All cluster commands run via SSH from local machine:
```bash
ssh pkorzeniowsk@pkorzeniowsk-oci-iad-cs.park.nvidia.com "<command>"
```

- **Cluster**: NVIDIA MARS (Slurm HPC)
- **Account**: `healthcareeng_holoscan`
- **Storage**: `/lustre/fsw/portfolios/healthcareeng/users/pkorzeniowsk/` (use this, NOT home dir — 10GB quota)
- **Containers**: Enroot/Pyxis (`.sqsh` files)
- **GPUs**: 8 per node (DGX)

## Slurm Essentials

```bash
# Submit job
sbatch <script.slurm>

# Monitor jobs
squeue -A healthcareeng_holoscan

# Cancel job
scancel <job_id>

# Job history
sacct -u $USER --format=jobid,nnodes,start,end,state,jobname

# Storage quota
lfs quota -u $USER -h /lustre/fsw/portfolios/healthcareeng/users/$USER
```

### SBATCH Directives
```bash
#SBATCH --job-name=<name>
#SBATCH --output=logs/%x_%j.log
#SBATCH --time=04:00:00              # Max 4h on most partitions
#SBATCH --nodes=<N>
#SBATCH --gpus-per-node=8
#SBATCH --partition=interactive,batch_block1,batch_block3,batch_block4
#SBATCH --account=healthcareeng_holoscan
```

### Multi-Node Distributed Training
```bash
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12341

srun \
  --container-image="/path/to/container.sqsh" \
  --container-mounts="/source:/dest" \
  --container-workdir=/workspace \
  bash -c 'torchrun --nnodes=$SLURM_NNODES --nproc_per_node=8 ...'
```

### QOS Limits
- Max 2 array jobs per user at a time (`QOSMaxSubmitJobPerUserLimit`)

### Google Drive (rclone)
```bash
# Upload results
ssh $SSH_HOST "rclone copy /path/to/folder 'gdrive:folder_name' --drive-root-folder-id 'FOLDER_ID' --progress"
```

### File Transfer
```bash
# Copy file from cluster to local
scp pkorzeniowsk@pkorzeniowsk-oci-iad-cs.park.nvidia.com:/path/to/file .
```
