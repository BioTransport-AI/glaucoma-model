# Data

Nothing in this directory is committed — it's gitignored. Fetch SMDG-19
yourself:

```bash
# from the biotransport-ai repo (fetch script lives there, shared across datasets)
cd ../ai-research/datasets
./scripts/fetch_kaggle.sh dataset deathtrooper/multichannel-glaucoma-benchmark-dataset smdg-19-glaucoma
```

The script resolves its destination relative to the `ai-research` repo root
(two directories up from `datasets/scripts/`), not inside `datasets/` itself
— it lands in `ai-research/data/smdg-19-glaucoma/`. Symlink that here:

```bash
ln -s ../../ai-research/data/smdg-19-glaucoma /home/noahrundell/Documents/cs/glaucoma-model/data/smdg-19-glaucoma
```

Requires the Kaggle CLI and an API token at `~/.kaggle/kaggle.json` — see the
`ai-research` repo's `datasets/README.md` for the one-time setup.
