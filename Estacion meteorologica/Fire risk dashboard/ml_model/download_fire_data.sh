#!/bin/bash
# Script to download fire dataset from Dataverse
# Usage: ./download_fire_data.sh YOUR_API_KEY

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Fire Dataset Download Script"
echo "=========================================="
echo ""

# Check if API key provided
if [ -z "$1" ]; then
    echo -e "${RED}ERROR: API key not provided${NC}"
    echo ""
    echo "Usage: ./download_fire_data.sh YOUR_API_KEY"
    echo ""
    echo "To get your API key:"
    echo "  1. Visit https://datospararesiliencia.cl"
    echo "  2. Log in or create an account"
    echo "  3. Navigate to Profile → Developer Tools"
    echo "  4. Generate an API key"
    echo ""
    exit 1
fi

API_KEY="$1"
DATASET_ID="doi:10.71578/XAZAKP"

echo -e "${YELLOW}Downloading fire dataset...${NC}"
echo "Dataset: $DATASET_ID"
echo "Target: ml_model/data/"
echo ""

# Activate conda environment
echo -e "${YELLOW}Activating conda environment...${NC}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fire_risk_dashboard

# Download fire dataset
cd "$(dirname "$0")/.."
python download_dataverse.py \
    --api-key "$API_KEY" \
    --dataset-id "$DATASET_ID" \
    --files "cicatrices_incendios_resumen.geojson" \
    --output-dir "ml_model/data"

# Check if download was successful
if [ -f "ml_model/data/cicatrices_incendios_resumen.geojson" ]; then
    echo ""
    echo -e "${GREEN}✓ Download successful!${NC}"
    echo ""
    echo "File saved to: ml_model/data/cicatrices_incendios_resumen.geojson"
    echo ""
    echo "Next steps:"
    echo "  1. cd ml_model"
    echo "  2. python prepare_training_data.py"
    echo "  3. python train_fire_model.py"
    echo ""
else
    echo -e "${RED}✗ Download failed${NC}"
    exit 1
fi
