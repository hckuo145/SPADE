# SPADE: SPoofed Area DEtecting

SPADE is a framework designed for detecting spoofed segments in partially fake audio. It supports both binary and ternary classification tasks across different model architectures and dataset configurations.

paper  : https://arxiv.org/pdf/2501.03805
dataset: https://huggingface.co/datasets/PeacefulData/SINE?fbclid=IwZXh0bgNhZW0CMTEAAR76639rdbuEGyDN7nV4BqZw7I8kdjSERUSETLrTq5Kkh65mWhgTLsWRLU3CXw_aem_hkjpfI3XlM4mU4E3EkCcHg

## 🔧 Model Architecture Configuration

- **ADD 22-1st for 3-class Classification**  
  `config/model/NaiveFCN_3class-Wav2Vec2_ft-ASP.yaml`

- **ADD 22-1st for 2-class Classification**  
  `config/model/NaiveFCN_2class-Wav2Vec2_ft-ASP.yaml`

- **ADD 22-2nd for 2-class Classification**  
  `config/model/SEResNet34_base_2class-MSTFT-ASP.yaml`

- **ADD 23-2nd for 2-class Classification**  
  `config/model/Transsion_2class-MSTFT.yaml`

- **ADD 23-3rd for 2-class Classification**  
  `config/model/ConvGRU_2class-MSTFT.yaml`

## 📁 Dataset Scenario Configuration

- **real / infill**  
  `config/hyper/VoiceNoNG_2class_edit_real.yaml`

- **resyn / infill**  
  `config/hyper/VoiceNoNG_2class_edit_resyn.yaml`

- **real / paste**  
  `config/hyper/VoiceNoNG_2class_redit_real.yaml`

- **real / resyn / edit (infill + paste)**  
  `config/hyper/VoiceNoNG_3class_all.yaml`

Notes:
1. In each dataset scenario configuration, please update the `data_path` field to your local directory path:
   ```yaml
   data_path: {{Your Path}}
   ```
2. The `data_list` field refers to a label file. All label files used in our experiments can be found in:
   ```
   data/VoiceNoNG/medium-v3-unet-500k
   ```
   If you wish to create your own subset, ensure the label format matches the following:

   ```
   {{filename}} {{segmentation prediction (T/F: genuine/spoofed)}} {{utterance prediction (1/0: genuine/spoofed)}}
   ```

   **Examples:**

   - Genuine audio:
     ```
     dev_real_medium-2527-les_mis_vol01_0810_librivox_64kb_mp3-lesmiserables_vol1_58_hugo_64kb_34 0.00-6.20-T 1
     ```
   - Spoofed audio:
     ```
     dev_edit_medium-10179-kalevala_lonnrot_1608_librivox_64kb_mp3-kalevala_33_lonnrot_64kb_11 0.00-0.94-T/0.94-1.40-F/1.40-6.48-T 0
     ```
     For the above spoofed audio, the fake segment lies between **0.94 and 1.40 seconds**.

## 🧠 Main File for Each Model

- `main_pseudo.py`: 3-class classification for ADD 22-1st and 22-2nd
- `main_spade.py`: 2-class classification for ADD 22-1st and 22-2nd
- `main_trans.py`: 2-class classification for ADD 23-2nd
- `main_crnn.py`: 2-class classification for ADD 23-3rd

## ▶️ Execute Instruction

Model name mapping:

| Model ID | Model Name   |
|----------|--------------|
| 22-1st   | NaiveFCN     |
| 22-2nd   | SEResNet34   |
| 23-2nd   | Transsion    |
| 23-3rd   | ConvGRU      |

Set your variables:

```bash
cmd="--title ${exp_name} --model ${model_name} --device cuda --batch ${batch_size} \   
     --model_conf ${model_conf} --hyper_conf ${hyper_conf}"
```

**Train:**
```bash
python main_xxxx.py --train ${cmd}
```

**Test:**
```bash
python main_xxxx.py --test ${cmd}
```

> Replace `main_xxxx.py` with the appropriate main file based on the model.

## 📌 Example

_2-class classification of 22-1st model for "real / infill" scenario_

```bash
exp_name=NaiveFCN_2class-Wav2Vec2_ft-ASP+VoiceNoNG_2class_edit_real
cmd="--title ${exp_name} --model NaiveFCN --device cuda --batch 64  \      
     --model_conf config/model/NaiveFCN_2class-Wav2Vec2_ft-ASP.yaml \     
     --hyper_conf config/hyper/VoiceNoNG_2class_edit_real.yaml"

python main_spade.py --train ${cmd}

exp_name=NaiveFCN_2class-Wav2Vec2_ft-ASP+VoiceNoNG_2class_edit_real
cmd="--title ${exp_name} --model NaiveFCN --device cuda --batch 64  \      
     --model_conf config/model/NaiveFCN_2class-Wav2Vec2_ft-ASP.yaml \     
     --hyper_conf config/hyper/VoiceNoNG_2class_edit_real.yaml      \
     --paramas exp/${exp_name}/checkpoints/best_valid_frame_F1.pt"

python main_spade.py --test ${cmd}
```

---

Feel free to customize paths, batch size, and training flags according to your experiment setup.
