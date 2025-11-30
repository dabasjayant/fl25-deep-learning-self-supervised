# DINO-Lite

Steps:
1. Prepare CUDA environment
   - Tested on cuda v11
   - Not optimized for Apple Silicon or other GPUs
2. Download dataset by running `python main.py -d` from the project root.
3. Run the command to pre-train model
   - Adjust arguments as needed
   - `nohup accelerate launch -m dino_single --data_path ./../../dataset/images --epochs 100 --batch_size 256`
   - `nohup accelerate launch -m dino_single --data_path ./../../dataset/images --epochs 100 --batch_size 512 --resume checkpoints_ssl/checkpoint_epoch_005.pth`