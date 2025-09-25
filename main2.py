import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import pandas as pd
import cv2
import librosa
import numpy as np
from vit_pytorch import ViT
from transformers import ASTFeatureExtractor, ASTModel
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
from torch.cuda.amp import autocast, GradScaler
from torch.nn import functional as F
import requests
import io
import tempfile
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Создаем сессию requests с повторными попытками и таймаутом
def create_requests_session(retries=5, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504)):
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

session = create_requests_session()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight


    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss



class MoseiDataset(Dataset):
    def __init__(self, csv_file, base_video_url, base_audio_url, download_prefix="", transform_video=None):
        self.data = pd.read_csv(csv_file, on_bad_lines='skip')
        self.transform_video = transform_video
        self.feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.emotions = ['happy', 'sad', 'anger', 'surprise', 'disgust', 'fear']
        self.base_video_url = base_video_url
        self.base_audio_url = base_audio_url
        self.download_prefix = download_prefix.rstrip("/")


    def __len__(self):
        return len(self.data)


    def _load_video_frame(self, video_url):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
                    response = session.get(video_url, timeout=(10, 30))
                    if response.status_code != 200:
                        tmp_file.close()
                        raise RuntimeError(f"Не удалось загрузить видео {video_url}")
                    content_type = response.headers.get('Content-Type', '')
                    if not (content_type.startswith('video') or content_type == 'application/octet-stream'):
                        tmp_file.close()
                        raise RuntimeError(f"Ожидался видеофайл, но получен файл с типом {content_type} для {video_url}")
                    tmp_file.write(response.content)
                    tmp_filepath = tmp_file.name


                cap = cv2.VideoCapture(tmp_filepath)
                if not cap.isOpened():
                    cap.release()
                    os.remove(tmp_filepath)
                    raise RuntimeError(f"Не удалось открыть видео {video_url}")

                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if frame_count == 0:
                    cap.release()
                    os.remove(tmp_filepath)
                    raise RuntimeError(f"Видео {video_url} содержит 0 кадров")

                frame_idx = np.random.randint(0, min(25, frame_count))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                cap.release()
                os.remove(tmp_filepath)
                if not ret:
                    raise RuntimeError(f"Не удалось загрузить кадр из видео {video_url}")

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.transform_video:
                    frame = self.transform_video(frame)
                return frame

            except (requests.exceptions.RequestException, RuntimeError) as e:
                if attempt < max_retries - 1:
                    print(f"Ошибка загрузки видео {video_url}: {e}, попытка {attempt + 1}/{max_retries} повтор через задержку...")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise e


    def _load_audio(self, audio_url):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = session.get(audio_url, timeout=(10, 30))
                if response.status_code != 200:
                    raise RuntimeError(f"Не удалось загрузить аудио {audio_url}")

                audio_bytes = io.BytesIO(response.content)
                y, sr = librosa.load(audio_bytes, sr=16000)
                audio_input = self.feature_extractor(y, sampling_rate=16000, return_tensors="pt")
                return audio_input

            except (requests.exceptions.RequestException, RuntimeError) as e:
                if attempt < max_retries - 1:
                    print(f"Ошибка загрузки аудио {audio_url}: {e}, попытка {attempt + 1}/{max_retries} повтор через задержку...")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise e


    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_url = f"{self.download_prefix}/{self.base_video_url}/{row['video']}.mp4"
        audio_url = f"{self.download_prefix}/{self.base_audio_url}/{row['video']}.wav"

        frame = self._load_video_frame(video_url)
        audio_input = self._load_audio(audio_url)

        emotions_values = np.array([row[emotion] for emotion in self.emotions])
        label = int(np.argmax(emotions_values))

        return frame, audio_input, torch.tensor(label).long()



transform_video = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])



class AttentionFusion(nn.Module):
    def __init__(self, dim_audio, dim_video):
        super().__init__()
        self.query = nn.Linear(dim_video, dim_video)
        self.key = nn.Linear(dim_audio, dim_video)
        self.value = nn.Linear(dim_audio, dim_video)
        self.softmax = nn.Softmax(dim=-1)
        self.out = nn.Linear(dim_video, dim_video)


    def forward(self, video_feat, audio_feat):
        q = self.query(video_feat).unsqueeze(1)
        k = self.key(audio_feat).unsqueeze(2)
        v = self.value(audio_feat).unsqueeze(1)
        attn_weights = self.softmax(torch.bmm(q, k))
        fused = attn_weights * v
        fused = fused.squeeze(1)
        output = self.out(fused)
        return output



class MultiModalModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.vit = ViT(
            image_size=224,
            patch_size=16,
            num_classes=512,
            dim=512,
            depth=6,
            heads=8,
            mlp_dim=1024,
            dropout=0.1,
            emb_dropout=0.1
        )
        self.ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        parameters = list(self.ast.parameters())
        freeze_upto = int(0.5 * len(parameters))
        for param in parameters[:freeze_upto]:
            param.requires_grad = False


        self.ast_proj = nn.Linear(self.ast.config.hidden_size, 512)
        self.fusion = AttentionFusion(dim_audio=512, dim_video=512)
        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Dropout(p=0.4),
            nn.Linear(512, num_classes)
        )


    def forward(self, video, audio_input):
        video_feat = self.vit(video)
        audio_input = {k: v.to(video.device) for k, v in audio_input.items()}
        ast_outputs = self.ast(**audio_input)
        audio_feat = ast_outputs.last_hidden_state.mean(dim=1)
        audio_feat = self.ast_proj(audio_feat)
        fused_feat = self.fusion(video_feat, audio_feat)
        logits = self.classifier(fused_feat)
        return logits



def collate_fn(batch):
    frames, audio_inputs, labels = zip(*batch)
    frames = torch.stack(frames)
    input_values = torch.stack([ai["input_values"].squeeze(0) for ai in audio_inputs])
    audio_batch = {"input_values": input_values}
    labels = torch.stack(labels)
    return frames, audio_batch, labels



train_csv_local = "Data_Train_modified.csv"
val_csv_local = "Data_Val_modified.csv"


download_prefix = "https://cloclo62.cloud.mail.ru/public/2s6pwtRCzdYwiqAu2MA7/g/no"
video_base_url = "XSL6/w3a6wUwa6/Video/Combined"
audio_base_url = "XSL6/w3a6wUwa6/Audio/WAV_16000"


train_dataset = MoseiDataset(train_csv_local, video_base_url, audio_base_url, download_prefix=download_prefix, transform_video=transform_video)
val_dataset = MoseiDataset(val_csv_local, video_base_url, audio_base_url, download_prefix=download_prefix, transform_video=transform_video)


labels_list = [int(np.argmax(row)) for row in train_dataset.data[['happy', 'sad', 'anger', 'surprise', 'disgust', 'fear']].values]
class_weights = compute_class_weight(class_weight='balanced', classes=np.arange(6), y=labels_list)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)


sample_weights = [class_weights[label] for label in labels_list]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, num_workers=4, collate_fn=collate_fn, pin_memory=True)


model = MultiModalModel(num_classes=6).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
criterion = FocalLoss(gamma=2.0, weight=class_weights_tensor)
scaler = GradScaler()
epochs = 30


for epoch in range(epochs):
    model.train()
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    running_loss = 0.0
    all_preds = []
    all_labels = []


    for video, audio_input, labels in loop:
        video, labels = video.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(video, audio_input)
            loss = criterion(outputs, labels)


        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


        running_loss += loss.item()
        loop.set_postfix(loss=running_loss / (loop.n + 1))


        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"Epoch {epoch+1} accuracy: {accuracy:.4f}, F1-score: {f1:.4f}")


    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f"model_epoch_{epoch+1}.pth")
    print(f"Model saved to model_epoch_{epoch+1}.pth")


print("Обучение завершено")
