# multimodal_emotion_classifier_in_docker



```Bash
sudo docker build -t emotion_classifier:1 .
sudo docker run --rm --shm-size=16g --runtime=nvidia --gpus all emotion_classifier:1
```
