#!/bin/bash
# Quick Start Script for Unified Experiments
# ==========================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Unified Experiments Quick Start${NC}"
echo -e "${GREEN}================================${NC}"
echo ""

# Create directories
mkdir -p results logs figures

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found!${NC}"
    exit 1
fi

echo -e "${YELLOW}Select experiment mode:${NC}"
echo "  1) test       - Quick test (2-3 min, 6 configs)"
echo "  2) validation - Full validation (30-60 min, ~40 configs)"
echo "  3) extensive  - Overnight sweep (8-12 hours, ~500 configs)"
echo "  4) custom     - Custom configuration"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        MODE="test"
        WORKERS=4
        ;;
    2)
        MODE="validation"
        read -p "Number of workers (default: 8): " WORKERS
        WORKERS=${WORKERS:-8}
        ;;
    3)
        MODE="extensive"
        read -p "Number of workers (default: all CPUs): " WORKERS
        WORKERS=${WORKERS:-}
        ;;
    4)
        MODE="custom"
        read -p "T values (comma-separated, e.g., 20,50,100): " T_VALS
        read -p "c_Q values (e.g., 0.5,1.0,2.0): " CQ_VALS
        read -p "lambda values (e.g., 0.01,0.1): " LAMBDA_VALS
        read -p "m values (e.g., 100,200): " M_VALS
        read -p "Number of workers: " WORKERS
        ;;
    *)
        echo -e "${RED}Invalid choice!${NC}"
        exit 1
        ;;
esac

# Generate output filename
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_FILE="results/unified_${MODE}_${TIMESTAMP}.json"
LOG_FILE="logs/unified_${MODE}_${TIMESTAMP}.log"

echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  Mode: $MODE"
echo "  Workers: ${WORKERS:-all CPUs}"
echo "  Output: $OUT_FILE"
echo "  Log: $LOG_FILE"
echo ""

# Build command
CMD="python3 experiments/unified_experiments.py --mode $MODE --out $OUT_FILE"

if [ ! -z "$WORKERS" ]; then
    CMD="$CMD --workers $WORKERS"
fi

if [ "$MODE" = "custom" ]; then
    CMD="$CMD --T $T_VALS --c_Q $CQ_VALS --lambda $LAMBDA_VALS --m $M_VALS"
fi

read -p "Run in background? [y/N]: " BACKGROUND

if [[ $BACKGROUND =~ ^[Yy]$ ]]; then
    echo ""
    echo -e "${GREEN}Starting experiment in background...${NC}"
    nohup $CMD > $LOG_FILE 2>&1 &
    PID=$!
    echo "  PID: $PID"
    echo "  Monitor with: tail -f $LOG_FILE"
    echo "  Stop with: kill $PID"
    echo ""
    echo "Experiment started!"
    echo "Check progress: tail -f $LOG_FILE"
else
    echo ""
    echo -e "${GREEN}Starting experiment...${NC}"
    echo "$CMD"
    echo ""
    $CMD | tee $LOG_FILE
fi

echo ""
echo -e "${GREEN}Done!${NC}"
echo "Results: $OUT_FILE"
echo "Log: $LOG_FILE"
