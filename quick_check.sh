#!/bin/bash
# Quick training status check - just the essentials

echo "=== QUICK STATUS CHECK ==="
echo ""

# Job status
echo "Job Status:"
ssh soc-cluster "squeue -u \$USER | grep char_flow || echo 'Job not running (completed or failed)'"
echo ""

# Latest output
echo "Latest Training Output:"
ssh soc-cluster "tail -15 ~/cs4243_captcha/train_character_v2_283036.out"
echo ""

# Sample count
SAMPLE_COUNT=$(ssh soc-cluster "ls ~/cs4243_captcha/results/character_model_v2/samples/*.png 2>/dev/null | wc -l")
echo "Generated Samples: $SAMPLE_COUNT images"
if [ "$SAMPLE_COUNT" -gt 0 ]; then
    ssh soc-cluster "ls -lh ~/cs4243_captcha/results/character_model_v2/samples/*.png | tail -3"
fi
