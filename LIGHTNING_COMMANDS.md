# Lightning Commands

Upload `cloud_removal_india.zip` to `/teamspace/studios/this_studio`.

## Unzip And Setup

```bash
cd /teamspace/studios/this_studio
unzip cloud_removal_india.zip
cd cloud_removal_india
chmod +x scripts/*.sh
bash scripts/setup_lightning.sh
```

## Run In Background

Recommended tiny check first:

```bash
nohup bash scripts/run_fast_check.sh > logs/fast_check.log 2>&1 &
tail -f logs/fast_check.log
```

Full run:

```bash
nohup bash scripts/run_full_pipeline.sh > logs/full_pipeline.log 2>&1 &
```

## Watch Progress

```bash
tail -f logs/full_pipeline.log
```

## Check GPU

```bash
nvidia-smi
```

## Package Results

```bash
bash scripts/package_results.sh
```

## Result Zip Path

```text
/teamspace/studios/this_studio/cloud_removal_india/results/cloud_removal_india_results.zip
```

## Useful Debug Commands

```bash
tail -n 100 logs/full_pipeline.log
ls -lh data/patches | head
ls -lh models
ls -lh results
```
