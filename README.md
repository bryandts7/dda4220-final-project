# Predictive Vision for Robotics: Fine-tuned InstructPix2Pix Model

**Authors:** Bryan Sibarani, Cahyadi Bodhiputra, Daniel Tanuwijaya, Filbert Hamijoyo  
**Affiliation:** Chinese University of Hong Kong, Shenzhen

## Overview

This project addresses the challenge of predicting future visual observations of robots by fine-tuning InstructPix2Pix, a state-of-the-art diffusion-based model, for action-conditioned frame prediction. Our approach leverages the RoboTwin simulation framework to generate a comprehensive dataset and enables realistic future frame generation given current observations and natural language action descriptions.

### Key Achievements
- **PSNR:** 33.12 dB (vs. 28.77 dB baseline)
- **SSIM:** 0.8181 (vs. 0.4363 baseline)
- **LPIPS:** 0.1052 (vs. 0.3155 baseline)
- **FID:** 0.0010 (vs. 0.0025 baseline)

This repository consists of two main components:
1. **Instruction-Tuned Stable Diffusion** - A model for action-conditioned frame prediction in robotic tasks
2. **RoboTwin** - A dual-arm robot benchmark with generative digital twins used for data generation

## Dataset: RoboTwin

### Description
The [RoboTwin Dataset](https://github.com/TianxingChen/RoboTwin) is a large-scale, high-fidelity dataset for sim-to-real imitation learning in robotic manipulation. It supports high-resolution visual observations, 3D scene understanding, and multi-arm control for complex coordination.
For installation requirements and instructions, see [RoboTwin Installation](https://github.com/TianxingChen/RoboTwin/blob/main/INSTALLATION.md).

### Features
- 17+ robotic manipulation tasks
- Support for multiple camera types (D435, L515)
- Baseline implementations (RDT, Diffusion Policy, 3D Diffusion Policy)
- Data collection and evaluation tools
- Sim2Real capabilities

### RoboTwin Data Collection
The dataset is constructed from RoboTwin simulations across three robotic tasks:
1. "Beat the block with the hammer"
2. "Handover the blocks"
3. "Stack blocks"

```bash
# For each of our three target tasks
bash run_task.sh block_hammer_beat 0
bash run_task.sh block_handover 0
bash run_task.sh blocks_stack_easy 0
```

For each task:
- 100 episodes collected
- Each episode contains 200-400 frames
- Training pairs extracted with 50-frame temporal offset
- Total dataset combines (current frame, action instruction, future frame) triplets

 For more information about usage of the dataset, see [RoboTwin Usage](https://github.com/TianxingChen/RoboTwin?tab=readme-ov-file#-usage). 
 
**Dataset Access:** [HuggingFace Dataset](https://huggingface.co/datasets/bryandts/robotwin-action-prediction-dataset)

## Model Architecture

Based on InstructPix2Pix with the following components:
1. **Text Encoder:** CLIP model for processing textual action instructions
2. **Image Encoder/Decoder:** VAE for latent space encoding/decoding
3. **UNet:** Fine-tuned diffusion model for denoising conditioned on text and image inputs

## Instruction-Tuned Stable Diffusion

### Installation
```bash
# Clone the repository
git clone https://github.com/bryandts7/dda4220-final-project.git
cd dda4220-final-project

# Create and activate a Python virtual environment
conda create -n instruct-sd python=3.8
conda activate instruct-sd

# Install PyTorch (adjust command for your CUDA version)
# Example for CUDA 11.6
pip install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu116

# Install other dependencies
pip install -r instruction-tuned-sd/requirements.txt

# Optional: Install xformers for memory-efficient training
pip install xformers
```

### Training Configuration
```bash
export MODEL_ID="runwayml/stable-diffusion-v1-5"
export DATASET_ID="bryandts/robotwin-action-prediction-dataset"
export OUTPUT_DIR="robotwin-action-prediction"

accelerate launch --mixed_precision="fp16" train_instruct_pix2pix.py \
  --pretrained_model_name_or_path=$MODEL_ID \
  --dataset_name=$DATASET_ID \
  --use_ema \
  --enable_xformers_memory_efficient_attention \
  --resolution=128 --random_flip \
  --train_batch_size=16 --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --checkpointing_steps=5000 --checkpoints_total_limit=1 \
  --learning_rate=5e-05 --lr_warmup_steps=0 \
  --mixed_precision=fp16 \
  --gradient_checkpointing \
  --seed=42 \
  --output_dir=$OUTPUT_DIR \
  --report_to=wandb \
  --push_to_hub
```

### Fine-tuned Model
Our trained model is available at: [HuggingFace Model Hub](https://huggingface.co/bryandts/instruct-pix2pix-robotwin-action-finetuned). Finetuning script can be found in `instruction-tuned-sd/script_finetune.txt`.

### Inference
```python
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.utils import load_image

model_id = "bryandts/instruct-pix2pix-robotwin-action-finetuned"
pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16
).to("cuda")

# Load current frame
current_frame = load_image("path/to/current_frame.png")

# Predict future frame
future_frame = pipeline(
    "Beat the block with the hammer",  # Action instruction
    image=current_frame
).images[0]

future_frame.save("predicted_future_frame.png")
```

## Model Evaluation

The repository includes a `model_evaluation.ipynb` notebook that provides tools for evaluating the performance of the trained model using four metrics:
- **PSNR (Peak Signal-to-Noise Ratio):** Measures pixel-wise fidelity
- **SSIM (Structural Similarity Index):** Evaluates structural similarity
- **LPIPS (Learned Perceptual Image Patch Similarity):** Assesses perceptual similarity
- **FID (Fréchet Inception Distance):** Measures distribution similarity

## Requirements

### System Requirements
- GPU: RTX 4090 or equivalent (24GB VRAM recommended)
- CUDA 11.6 or higher

### Instruction-Tuned Stable Diffusion
- Python 3.8+
- PyTorch 1.13.1+ (CUDA 11.6+ recommended)
- diffusers
- transformers
- accelerate
- wandb (for experiment tracking)
- xformers (optional, for memory optimization)

### RoboTwin
- Python 3.8 or 3.10
- PyTorch 2.4.1
- CUDA 12.1 (recommended)
- Vulkan
- NVIDIA Driver >= 470 (for ray tracing)

## Future Work

1. **Temporal Consistency:** Extend to multi-frame sequence prediction
2. **Real-World Deployment:** Test on physical robotic systems
3. **Diverse Tasks:** Expand to more complex manipulation tasks

## Citation

If you use this work in your research, please cite:

```bibtex
@article{sibarani2025predictive,
  title={Predictive Vision for Robotics: Fine-tuned InstructPix2Pix Model for Robotic Action Frame Prediction},
  author={Sibarani, Bryan and Bodhiputra, Cahyadi and Tanuwijaya, Daniel and Hamijoyo, Filbert},
  journal={DDA4220 Final Project},
  year={2025},
  institution={Chinese University of Hong Kong, Shenzhen}
}
```

For Instruction-Tuned Stable Diffusion:
```
@article{brooks2022instructpix2pix,
  title={InstructPix2Pix: Learning to Follow Image Editing Instructions},
  author={Brooks, Tim and Holynski, Aleksander and Efros, Alexei A},
  journal={arXiv preprint arXiv:2211.09800},
  year={2022}
}
```

For RoboTwin:
```
@article{mu2025robotwin,
  title={RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins},
  author={Mu, Yao and Chen, Tianxing and Chen, Zanxin and Peng, Shijia and Lan, Zhiqian and Gao, Zeyu and Liang, Zhixuan and Yu, Qiaojun and Zou, Yude and Xu, Mingkun and Lin, Lunkai and Xie, Zhiqiang and Ding, Mingyu and Luo, Ping},
  journal={arXiv preprint arXiv:2504.13059},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This work builds upon:
- [InstructPix2Pix](https://arxiv.org/abs/2211.09800) by Brooks et al.
- [RoboTwin](https://arxiv.org/abs/2504.13059) by Mu et al.
- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) 
