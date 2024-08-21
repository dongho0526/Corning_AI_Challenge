# Corning_AI_Challenge

Please note that all codes refernces the official implementation [YOLO v8](https://github.com/ultralytics/ultralytics), [YOLO v5](https://github.com/ultralytics/yolov5), [Grounding dino](https://github.com/open-mmlab/mmdetection/blob/main/configs/grounding_dino/README.md)

## Mandatory Implementation Requirements
1. Able to perform inference on a single image or multiple image
2. Able to learn on a new dataset
3. Able to tune hyperparameters
4. Able to check Bounding Box and Confidence Score

## Data Example
![image](https://github.com/user-attachments/assets/34371972-143f-48d9-a0ac-97c0458c2d18)
- A total of 5 defect classes and 159 images are available.(including roation and translation)

## Problems
- __The images are simple in form, but despite this, the number of data samples is low.__
- Since we don't have information about the data, it is not possible to collect additional data.

## Solutions

![image](https://github.com/user-attachments/assets/3ca5839e-4e8d-4e83-a34c-1700c0ea3f12)
- Data was generated using the generative model, VAE.
