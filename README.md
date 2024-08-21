# Corning AI Challenge

Please note that all codes refernces the official implementation [YOLO v8](https://github.com/ultralytics/ultralytics), [YOLO v5](https://github.com/ultralytics/yolov5), [Grounding dino](https://github.com/open-mmlab/mmdetection/blob/main/configs/grounding_dino/README.md)

## Mandatory Implementation Requirements
1. Able to perform inference on a single image or multiple image
2. Able to learn on a new dataset
3. Able to tune hyperparameters
4. Able to check Bounding Box and Confidence Score

## Data Example
![image](https://github.com/user-attachments/assets/34371972-143f-48d9-a0ac-97c0458c2d18)
- A total of 5 defect classes and 159 images are available.(including roation and translation)

## Problem-1
- __The images are simple in form, but despite this, the number of data samples is low.__
- Since we don't have information about the data, it is not possible to collect additional data.

## Solutions-1

![image](https://github.com/user-attachments/assets/3ca5839e-4e8d-4e83-a34c-1700c0ea3f12)
- Data was generated using the generative model, VAE.

![image](https://github.com/user-attachments/assets/a8a94fdb-0be7-4b8f-a7f4-4acadf3873d7)
- By processing everything in one go, rather than detecting data sepreately for each class, we address the issue of data scarcity.

## Problem-2
- Absesnce of Bounding Box Coordinate values

## Solution-2
![image](https://github.com/user-attachments/assets/fefd272d-fff1-4b1e-8335-ee68ed051b8f)
![image](https://github.com/user-attachments/assets/0101d3e3-d7f4-43c7-ba5f-f2fc13c8950f)
- We used Roboflow to appropriately structure the dataset's file organization and generated txt files containing Bounding Box and class IDs.

## UI Implementation

https://github.com/user-attachments/assets/986e34a9-e679-4ec6-9883-2a3b979e0a99

