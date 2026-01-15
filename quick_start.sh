#!/bin/bash

# Quick Start Script for CAPTCHA Flow Training
# This script guides you through the entire process

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════╗"
echo -e "║  CAPTCHA Flow - Quick Start Guide     ║"
echo -e "╚════════════════════════════════════════╝${NC}"
echo ""

echo -e "${YELLOW}This script will:${NC}"
echo "1. Transfer all files to SOC cluster"
echo "2. Guide you to submit the training job"
echo ""

# Ask for confirmation
read -p "Do you want to proceed? (y/n): " confirm
if [[ $confirm != [yY] && $confirm != [yY][eE][sS] ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo -e "${BLUE}=========================================="
echo "Step 1: Transferring Files to Cluster"
echo -e "==========================================${NC}"
echo ""

# Run transfer script
./setup_and_transfer.sh

if [ $? -ne 0 ]; then
    echo -e "${RED}Transfer failed. Please check the errors above.${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}=========================================="
echo "Step 2: Submit Training Job"
echo -e "==========================================${NC}"
echo ""

echo -e "${YELLOW}Enter your SOC username again:${NC}"
read -p "Username: " SOC_USERNAME

if [ -z "$SOC_USERNAME" ]; then
    echo -e "${RED}ERROR: Username cannot be empty${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Files successfully transferred!${NC}"
echo ""
echo -e "${YELLOW}Now connecting to cluster to submit job...${NC}"
echo ""

# SSH and submit job
ssh -t ${SOC_USERNAME}@xlogin0.comp.nus.edu.sg << 'ENDSSH'
cd ~/captcha_flow
echo "Current directory: $(pwd)"
echo ""
echo "Contents:"
ls -lh
echo ""
echo "Submitting training job..."
sbatch train_job.sh
echo ""
echo "Checking job queue..."
sleep 2
squeue -u $USER
echo ""
echo "═══════════════════════════════════════"
echo "Training job submitted!"
echo "═══════════════════════════════════════"
echo ""
echo "Monitor your job with:"
echo "  squeue -u \$USER"
echo ""
echo "View logs with:"
echo "  tail -f ~/captcha_flow/logs/train_*.out"
echo ""
echo "Press Enter to keep this SSH session open, or Ctrl+D to exit"
bash
ENDSSH

echo ""
echo -e "${GREEN}All done! Check the cluster for training progress.${NC}"
