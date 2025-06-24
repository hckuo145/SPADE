# SPADE: SPoofed Area DEtecting

SPADE is a framework designed for detecting spoofed segments in partially fake audio. It supports both binary and ternary classification tasks across different model architectures and dataset configurations.

## 🔧 Model Architecture Configuration

- **22-1st for 3-class Classification**  
  `config/model/NaiveFCN_3class-Wav2Vec2_ft-ASP.yaml`

- **22-1st for 2-class Classification**  
  `config/model/NaiveFCN_2class-Wav2Vec2_ft-ASP.yaml`

- **22-2nd for 2-class Classification**  
  `config/model/SEResNet34_base_2class-MSTFT-ASP.yaml`

- **23-2nd for 2-class Classification**  
  `config/model/Transsion_2class-MSTFT.yaml`

- **23-3rd for 2-class Classification**  
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

## 🧠 Main File for Each Model

- `main_pseudo.py`: 3-class classification for 22-1st and 22-2nd
- `main_spade.py`: 2-class classification for 22-1st and 22-2nd
- `main_trans.py`: 2-class classification for 23-2nd
- `main_crnn.py`: 2-class classification for 23-3rd

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
python main_spade.py --test ${cmd}
```

---

Feel free to customize paths, batch size, and training flags according to your experiment setup.
