# title=NaiveFCN_3class-Wav2Vec2_ft-ASP+VoiceNoNG_3class_all
# cmd="--title ${title} --model NaiveFCN --device cuda --batch 16     \
#      --model_conf config/model/NaiveFCN_3class-Wav2Vec2_ft-ASP.yaml \
#      --hyper_conf config/hyper/VoiceNoNG_3class_all.yaml"
# 
# mkdir -p exp/${title}
# logfile=exp/${title}/history.log
# 
# if [ -f ${logfile} ]
# then
#      echo "Experiment ${title} exists."
# else
#      python main_pseudo.py --train ${cmd} > ${logfile}
# fi

# metric=VC_edit_frame_F1_edit
# cmd="--title ${title} --model NaiveFCN --device cuda --batch 16     \
#      --model_conf config/model/NaiveFCN_3class-Wav2Vec2_ft-ASP.yaml \
#      --hyper_conf config/hyper/VoiceNoNG_3class_all.yaml            \
#      --params exp/${title}/checkpoints/best_valid_${metric}.pt"

# python main_pseudo.py --test  ${cmd}

#######################################################################
# title=NaiveFCN_2class-Wav2Vec2_ft-ASP+VoiceNoNG_2class_edit_real
# cmd="--title ${title} --model NaiveFCN --device cuda --batch 64     \
#      --model_conf config/model/NaiveFCN_2class-Wav2Vec2_ft-ASP.yaml \
#      --hyper_conf config/hyper/VoiceNoNG_2class_edit_real.yaml      \
#      --params exp/${title}/checkpoints/best_valid_frame_F1.pt"
# 
# title=NaiveFCN_2class-Wav2Vec2_ft-ASP+VoiceNoNG_2class_edit_resyn
# cmd="--title ${title} --model NaiveFCN --device cuda --batch 64     \
#      --model_conf config/model/NaiveFCN_2class-Wav2Vec2_ft-ASP.yaml \
#      --hyper_conf config/hyper/VoiceNoNG_2class_edit_resyn.yaml     \
#      --params exp/${title}/checkpoints/best_valid_frame_F1.pt"
# 
title=NaiveFCN_2class-Wav2Vec2_ft-ASP+VoiceNoNG_2class_redit_real
cmd="--title ${title} --model NaiveFCN --device cuda --batch 64     \
     --model_conf config/model/NaiveFCN_2class-Wav2Vec2_ft-ASP.yaml \
     --hyper_conf config/hyper/VoiceNoNG_2class_edit_resyn.yaml     \
     --params exp/${title}/checkpoints/best_valid_frame_F1.pt"
#
#
# mkdir -p exp/${title}
# logfile=exp/${title}/history.log
# 
# if [ -f ${logfile} ]
# then
#      echo "Experiment ${title} exists."
# else
#      python main_spade.py --train ${cmd} > ${logfile}
# fi

python main_spade.py --test  ${cmd}

#######################################################################
# title=SEResNet34_base_2class-MSTFT-ASP+VoiceNoNG_2class_edit_real
# cmd="--title ${title} --model Spade --device cuda --batch 32         \
#      --model_conf config/model/SEResNet34_base_2class-MSTFT-ASP.yaml \
#      --hyper_conf config/hyper/VoiceNoNG_2class_edit_real.yaml       \
#      --params exp/${title}/checkpoints/best_valid_frame_F1.pt"
# 
# title=SEResNet34_base_2class-MSTFT-ASP+VoiceNoNG_2class_edit_resyn
# cmd="--title ${title} --model Spade --device cuda --batch 32         \
#      --model_conf config/model/SEResNet34_base_2class-MSTFT-ASP.yaml \
#      --hyper_conf config/hyper/VoiceNoNG_2class_edit_resyn.yaml      \
#      --params exp/${title}/checkpoints/best_valid_frame_F1.pt"
# 
# title=SEResNet34_base_2class-MSTFT-ASP+VoiceNoNG_2class_redit_real
# cmd="--title ${title} --model Spade --device cuda --batch 32         \
#      --model_conf config/model/SEResNet34_base_2class-MSTFT-ASP.yaml \
#      --hyper_conf config/hyper/VoiceNoNG_2class_redit_real.yaml      \
#      --params exp/${title}/checkpoints/best_valid_frame_F1.pt"
# 
# 
# mkdir -p exp/${title}
# logfile=exp/${title}/history.log
# 
# if [ -f ${logfile} ]
# then
#      echo "Experiment ${title} exists."
# else
#      python main_spade.py --train ${cmd} > ${logfile}
# fi
# 
# python main_spade.py --test  ${cmd}

#######################################################################
# title=Transsion_2class-MSTFT-VoiceNoNG_2class_edit_real
# cmd="--title ${title} --model Transsion --device cuda --batch 512 \
#      --model_conf config/model/Transsion_2class-MSTFT.yaml        \
#      --hyper_conf config/hyper/VoiceNoNG_2class_edit_real.yaml    \
#      --params exp/${title}/checkpoints/best_valid_frame_F1.pt"
# 
# title=Transsion_2class-MSTFT-VoiceNoNG_2class_edit_resyn
# cmd="--title ${title} --model Transsion --device cuda --batch 512 \
#      --model_conf config/model/Transsion_2class-MSTFT.yaml        \
#      --hyper_conf config/hyper/VoiceNoNG_2class_edit_resyn.yaml   \
#      --params exp/${title}/checkpoints/best_valid_frame_F1.pt"
# 
# title=Transsion_2class-MSTFT-VoiceNoNG_2class_redit_real
# cmd="--title ${title} --model Transsion --device cuda --batch 512 \
#      --model_conf config/model/Transsion_2class-MSTFT.yaml        \
#      --hyper_conf config/hyper/VoiceNoNG_2class_redit_real.yaml   \
#      --params exp/${title}/checkpoints/best_valid_frame_F1.pt"
# 
# 
# mkdir -p exp/${title}
# logfile=exp/${title}/history.log
# 
# if [ -f ${logfile} ]
# then
#      echo "Experiment ${title} exists."
# else
#      python main_trans.py --train ${cmd} > ${logfile}
# fi
# 
# python main_trans.py --test  ${cmd}

#######################################################################
# title=ConvGRU_2class-MSTFT-VoiceNoNG_2class_edit_real
# cmd="--title ${title} --model ConvGRU --device cuda --batch 256 \
#      --model_conf config/model/ConvGRU_2class-MSTFT.yaml        \
#      --hyper_conf config/hyper/VoiceNoNG_2class_edit_real.yaml  \
#      --params exp/${title}/checkpoints/best_valid_frame_F1.pt"
# 
# title=ConvGRU_2class-MSTFT-VoiceNoNG_2class_edit_resyn
# cmd="--title ${title} --model ConvGRU --device cuda --batch 256 \
#      --model_conf config/model/ConvGRU_2class-MSTFT.yaml        \
#      --hyper_conf config/hyper/VoiceNoNG_2class_edit_resyn.yaml \
#      --params exp/${title}/checkpoints/best_valid_frame_F1.pt"
# 
# title=ConvGRU_2class-MSTFT-VoiceNoNG_2class_redit_real
# cmd="--title ${title} --model ConvGRU --device cuda --batch 256 \
#      --model_conf config/model/ConvGRU_2class-MSTFT.yaml        \
#      --hyper_conf config/hyper/VoiceNoNG_2class_redit_real.yaml \
#      --params exp/${title}/checkpoints/best_valid_frame_F1.pt"
# 
# 
# mkdir -p exp/${title}
# logfile=exp/${title}/history.log
# 
# if [ -f ${logfile} ]
# then
#      echo "Experiment ${title} exists."
# else
#      python main_crnn.py --train ${cmd} > ${logfile}
# fi
# 
# python main_crnn.py --test  ${cmd}
