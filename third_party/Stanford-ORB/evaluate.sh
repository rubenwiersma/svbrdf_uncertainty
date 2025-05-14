#!/bin/bash
python scripts/test_cache.py --method sh --scenes full
python scripts/test.py --input-path logs/test/sh.json --output-path logs/test/sh_results.json --scenes full