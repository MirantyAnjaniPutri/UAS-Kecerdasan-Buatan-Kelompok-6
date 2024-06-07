import yaml
from ultralytics import YOLO

def main():
    # Load dataset configuration
    with open('C:/Users/miran/code_skripsi/datasets/data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)

    # Define model configurations
    model_configs = {
        'yolov8n-obb': 'C:/Users/miran/code_skripsi/Yolov8n-obb.pt'
        # 'yolov8s-seg': 'C:/Users/miran/code_skripsi/yolov8s-seg.pt',
        # 'yolov8n-seg': 'C:/Users/miran/code_skripsi/yolov8n-seg.pt'
    }

    # Define common training parameters
    common_params = {
        # 'data': 'C:/Users/miran/code_skripsi/dataset_binarize/data.yaml',
        'data': 'C:/Users/miran/code_skripsi/datasets/data.yaml',
        'epochs': 50,  # Number of epochs to train
        'batch': 8,  # Batch size
        'imgsz': 640  # Image size
    }

    # Train each model and save results
    for model_name, model_weights in model_configs.items():
        print(f"Training {model_name}...")
        model = YOLO(model_weights)  # Initialize the model with pre-trained weights
        train_params = {
            **common_params,
            'name': model_name  # Name for the run, this will create a directory for each model's results
        }
        model.train(**train_params)
        print(f"Training for {model_name} completed. Results saved in runs/{model_name}/")

    print("All models have been trained.")

if __name__ == '__main__':
    main()