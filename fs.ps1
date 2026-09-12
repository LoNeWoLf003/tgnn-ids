New-Item -ItemType Directory -Force -Path data/raw, data/interim, data/processed
New-Item -ItemType Directory -Force -Path src/models
New-Item -ItemType Directory -Force -Path checkpoints, results, paper
New-Item -ItemType File -Force -Path src/__init__.py, src/models/__init__.py