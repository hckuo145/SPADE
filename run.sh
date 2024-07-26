# title=NaiveFCN_Wav2Vec2_ASP-HAD
# cmd="--title ${title} --model NaiveFCN --device cuda --batch 16 \
#      --model_conf config/model/NaiveFCN_Wav2Vec2_ASP.yaml       \
#      --hyper_conf config/hyper/Spade_HAD.yaml                   \
#      --params exp/${title}/checkpoints/best_valid_score.pt"

# title=NaiveFCN_Wav2Vec2_ASP-VB_edit_real
# cmd="--title ${title} --model NaiveFCN --device cuda --batch 16 \
#      --model_conf config/model/NaiveFCN_Wav2Vec2_ASP.yaml       \
#      --hyper_conf config/hyper/Spade_VB_edit_real.yaml          \
#      --params exp/${title}/checkpoints/best_valid_score.pt"

title=NaiveFCN_Wav2Vec2_ASP-VB_paste_real
cmd="--title ${title} --model NaiveFCN --device cuda --batch 16 \
     --model_conf config/model/NaiveFCN_Wav2Vec2_ASP.yaml       \
     --hyper_conf config/hyper/Spade_VB_paste_real.yaml         \
     --params exp/${title}/checkpoints/best_valid_score.pt"

# title=NaiveFCN_Wav2Vec2_ASP-VB_edit_resyn
# cmd="--title ${title} --model NaiveFCN --device cuda --batch 16 \
#      --model_conf config/model/NaiveFCN_Wav2Vec2_ASP.yaml       \
#      --hyper_conf config/hyper/Spade_VB_edit_resyn.yaml         \
#      --params exp/${title}/checkpoints/best_valid_score.pt"

# title=NaiveFCN_Wav2Vec2_ASP-VB_edit_resyn_v2
# cmd="--title ${title} --model NaiveFCN --device cuda --batch 16 \
#      --model_conf config/model/NaiveFCN_Wav2Vec2_ASP_v2.yaml    \
#      --hyper_conf config/hyper/Spade_VB_edit_resyn.yaml         \
#      --params exp/${title}/checkpoints/best_valid_score.pt"

# title=NaiveFCN_Wav2Vec2_ASP-VB_edit+paste_real
# cmd="--title ${title} --model NaiveFCN --device cuda --batch 16 \
#      --model_conf config/model/NaiveFCN_Wav2Vec2_ASP.yaml       \
#      --hyper_conf config/hyper/Spade_VB_edit+paste_real.yaml    \
#      --params exp/${title}/checkpoints/best_valid_score.pt"

# title=NaiveFCN_Wav2Vec2_ASP-VB_paste_real_v2
# cmd="--title ${title} --model NaiveFCN --device cuda --batch 16 \
#      --model_conf config/model/NaiveFCN_Wav2Vec2_ASP.yaml       \
#      --hyper_conf config/hyper/Spade_VB_paste_real_v2.yaml      \
#      --params exp/${title}/checkpoints/best_valid_score.pt"


##### Ablation ####################################################################################

# title=NaiveFCN_MSTFT_ASP-VB_edit_resyn
# cmd="--title ${title} --model NaiveFCN --device cuda --batch 512 \
#      --model_conf config/model/NaiveFCN_MSTFT_ASP.yaml           \
#      --hyper_conf config/hyper/Spade_VB_edit_resyn.yaml          \
#      --params exp/${title}/checkpoints/best_valid_score.pt"

# title=NaiveFCN_Wav2Vec2_randinit_ASP-VB_edit_resyn
# cmd="--title ${title} --model NaiveFCN --device cuda --batch 16    \
#      --model_conf config/model/NaiveFCN_Wav2Vec2_randinit_ASP.yaml \
#      --hyper_conf config/hyper/Spade_VB_edit_resyn.yaml            \
#      --params exp/${title}/checkpoints/best_valid_score.pt"

###################################################################################################

# title=NaiveFCN_WavLM_ASP-HAD
# cmd="--title ${title} --model NaiveFCN --device cuda --batch 16 \
#      --model_conf config/model/NaiveFCN_WavLM_ASP.yaml          \
#      --hyper_conf config/hyper/Spade_HAD.yaml                   \
#      --params exp/${title}/checkpoints/best_valid_score.pt"

# title=NaiveFCN_WavLM_ASP-VB_edit_real
# cmd="--title ${title} --model NaiveFCN --device cuda --batch 16 \
#      --model_conf config/model/NaiveFCN_WavLM_ASP.yaml          \
#      --hyper_conf config/hyper/Spade_VB_edit_real.yaml          \
#      --params exp/${title}/checkpoints/best_valid_score.pt"

# title=NaiveFCN_WavLM_ASP-VB_paste_real
# cmd="--title ${title} --model NaiveFCN --device cuda --batch 16 \
#      --model_conf config/model/NaiveFCN_WavLM_ASP.yaml          \
#      --hyper_conf config/hyper/Spade_VB_paste_real.yaml         \
#      --params exp/${title}/checkpoints/best_valid_score.pt"

# title=NaiveFCN_WavLM_ASP-VB_edit_resyn
# cmd="--title ${title} --model NaiveFCN --device cuda --batch 16 \
#      --model_conf config/model/NaiveFCN_WavLM_ASP.yaml          \
#      --hyper_conf config/hyper/Spade_VB_edit_resyn.yaml         \
#      --params exp/${title}/checkpoints/best_valid_score.pt"

###################################################################################################

# title=SEResNet34_base_MSTFT_ASP-HAD
# cmd="--title ${title} --model Spade --device cuda --batch 48  \
#      --model_conf config/model/SEResNet34_base_MSTFT_ASP.yaml \
#      --hyper_conf config/hyper/Spade_HAD.yaml                 \
#      --params exp/${title}/checkpoints/best_valid_score.pt"

# title=SEResNet34_base_MSTFT_ASP-VB_edit_real
# cmd="--title ${title} --model Spade --device cuda --batch 48  \
#      --model_conf config/model/SEResNet34_base_MSTFT_ASP.yaml \
#      --hyper_conf config/hyper/Spade_VB_edit_real.yaml        \
#      --params exp/${title}/checkpoints/best_valid_score.pt"

# title=SEResNet34_base_MSTFT_ASP-VB_paste_real
# cmd="--title ${title} --model Spade --device cuda --batch 48  \
#      --model_conf config/model/SEResNet34_base_MSTFT_ASP.yaml \
#      --hyper_conf config/hyper/Spade_VB_paste_real.yaml       \
#      --params exp/${title}/checkpoints/best_valid_score.pt"

# title=SEResNet34_base_MSTFT_ASP-VB_edit_resyn
# cmd="--title ${title} --model Spade --device cuda --batch 48  \
#      --model_conf config/model/SEResNet34_base_MSTFT_ASP.yaml \
#      --hyper_conf config/hyper/Spade_VB_edit_resyn.yaml       \
#      --params exp/${title}/checkpoints/best_valid_score.pt"

# title=SEResNet34_base_MSTFT_ASP-VB_edit+paste_real
# cmd="--title ${title} --model Spade --device cuda --batch 48  \
#      --model_conf config/model/SEResNet34_base_MSTFT_ASP.yaml \
#      --hyper_conf config/hyper/Spade_VB_edit+paste_real.yaml  \
#      --params exp/${title}/checkpoints/best_valid_score.pt"

# title=SEResNet34_base_MSTFT_ASP-VB_paste_real_v2
# cmd="--title ${title} --model Spade --device cuda --batch 48  \
#      --model_conf config/model/SEResNet34_base_MSTFT_ASP.yaml \
#      --hyper_conf config/hyper/Spade_VB_paste_real_v2.yaml    \
#      --params exp/${title}/checkpoints/best_valid_score.pt"

# title=SEResNet34_base_drop_MSTFT_ASP-VB_paste_real
# cmd="--title ${title} --model Spade --device cuda --batch 48       \
#      --model_conf config/model/SEResNet34_base_drop_MSTFT_ASP.yaml \
#      --hyper_conf config/hyper/Spade_VB_paste_real.yaml            \
#      --params exp/${title}/checkpoints/best_valid_score.pt"

# title=SEResNet34_base_MSTFT_ASP-VB_paste_real_v3
# cmd="--title ${title} --model Spade --device cuda --batch 48  \
#      --model_conf config/model/SEResNet34_base_MSTFT_ASP.yaml \
#      --hyper_conf config/hyper/Spade_VB_paste_real_v3.yaml    \
#      --params exp/${title}/checkpoints/best_valid_score.pt"

# mkdir -p exp/${title}
# logfile=exp/${title}/history.log

# if [ -f ${logfile} ]
# then
#      echo "Experiment ${title} exists."
# else
#      python main_spade.py --train ${cmd} > ${logfile}
# fi
python main_spade.py --test  ${cmd}
# python main_spade.py --plot  ${cmd}

###################################################################################################

# title=Transsion_MSTFT-HAD
# cmd="--title ${title} --model Transsion --device cuda --batch 512 \
#      --model_conf config/model/Transsion_MSTFT.yaml               \
#      --hyper_conf config/hyper/Transsion_HAD.yaml                 \
#      --params exp/${title}/checkpoints/best_valid_score.pt"

# title=Transsion_MSTFT-VB_edit_real
# cmd="--title ${title} --model Transsion --device cuda --batch 512 \
#      --model_conf config/model/Transsion_MSTFT.yaml               \
#      --hyper_conf config/hyper/Transsion_VB_edit_real.yaml        \
#      --params exp/${title}/checkpoints/best_valid_score.pt"

# title=Transsion_MSTFT-VB_paste_real
# cmd="--title ${title} --model Transsion --device cuda --batch 512 \
#      --model_conf config/model/Transsion_MSTFT.yaml               \
#      --hyper_conf config/hyper/Transsion_VB_paste_real.yaml       \
#      --params exp/${title}/checkpoints/best_valid_score.pt"

# title=Transsion_MSTFT-VB_edit_resyn
# cmd="--title ${title} --model Transsion --device cuda --batch 512 \
#      --model_conf config/model/Transsion_MSTFT.yaml               \
#      --hyper_conf config/hyper/Transsion_VB_edit_resyn.yaml       \
#      --params exp/${title}/checkpoints/best_valid_score.pt"

# title=Transsion_MSTFT-VB_edit+paste_real
# cmd="--title ${title} --model Transsion --device cuda --batch 512 \
#      --model_conf config/model/Transsion_MSTFT.yaml               \
#      --hyper_conf config/hyper/Transsion_VB_edit+paste_real.yaml  \
#      --params exp/${title}/checkpoints/best_valid_score.pt"

# title=Transsion_MSTFT-VB_paste_real_v2
# cmd="--title ${title} --model Transsion --device cuda --batch 512 \
#      --model_conf config/model/Transsion_MSTFT.yaml               \
#      --hyper_conf config/hyper/Transsion_VB_paste_real_v2.yaml    \
#      --params exp/${title}/checkpoints/best_valid_score.pt"

# title=Transsion_drop_MSTFT-VB_paste_real
# cmd="--title ${title} --model Transsion --device cuda --batch 512 \
#      --model_conf config/model/Transsion_drop_MSTFT.yaml          \
#      --hyper_conf config/hyper/Transsion_VB_paste_real.yaml       \
#      --params exp/${title}/checkpoints/best_valid_score.pt"

# title=Transsion_MSTFT-VB_paste_real_v3
# cmd="--title ${title} --model Transsion --device cuda --batch 512 \
#      --model_conf config/model/Transsion_MSTFT.yaml               \
#      --hyper_conf config/hyper/Transsion_VB_paste_real_v3.yaml    \
#      --params exp/${title}/checkpoints/best_valid_score.pt"

# mkdir -p exp/${title}
# logfile=exp/${title}/history.log

# if [ -f ${logfile} ]
# then
#      echo "Experiment ${title} exists."
# else
#      python main_trans.py --train ${cmd} > ${logfile}
# fi
# python main_trans.py --test ${cmd}

###################################################################################################

# title=ConvGRU_MSTFT-HAD
# cmd="--title ${title} --model ConvGRU --device cuda --batch 128 \
#      --model_conf config/model/ConvGRU_MSTFT.yaml               \
#      --hyper_conf config/hyper/ConvGRU_HAD.yaml                 \
#      --params exp/${title}/checkpoints/best_valid_score.pt"

# title=ConvGRU_MSTFT-VB_edit_real
# cmd="--title ${title} --model ConvGRU --device cuda --batch 128 \
#      --model_conf config/model/ConvGRU_MSTFT.yaml               \
#      --hyper_conf config/hyper/ConvGRU_VB_edit_real.yaml        \
#      --params exp/${title}/checkpoints/best_valid_score.pt"

# title=ConvGRU_MSTFT-VB_paste_real
# cmd="--title ${title} --model ConvGRU --device cuda --batch 128 \
#      --model_conf config/model/ConvGRU_MSTFT.yaml               \
#      --hyper_conf config/hyper/ConvGRU_VB_paste_real.yaml       \
#      --params exp/${title}/checkpoints/best_valid_score.pt"

# title=ConvGRU_MSTFT-VB_edit_resyn
# cmd="--title ${title} --model ConvGRU --device cuda --batch 128 \
#      --model_conf config/model/ConvGRU_MSTFT.yaml               \
#      --hyper_conf config/hyper/ConvGRU_VB_edit_resyn.yaml       \
#      --params exp/${title}/checkpoints/best_valid_score.pt"

# title=ConvGRU_MSTFT-VB_edit+paste_real
# cmd="--title ${title} --model ConvGRU --device cuda --batch 128 \
#      --model_conf config/model/ConvGRU_MSTFT.yaml               \
#      --hyper_conf config/hyper/ConvGRU_VB_edit+paste_real.yaml  \
#      --params exp/${title}/checkpoints/best_valid_score.pt"

# title=ConvGRU_MSTFT-VB_paste_real_v2
# cmd="--title ${title} --model ConvGRU --device cuda --batch 128 \
#      --model_conf config/model/ConvGRU_MSTFT.yaml               \
#      --hyper_conf config/hyper/ConvGRU_VB_paste_real_v2.yaml    \
#      --params exp/${title}/checkpoints/best_valid_score.pt"

# title=ConvGRU_drop_MSTFT-VB_paste_real
# cmd="--title ${title} --model ConvGRU --device cuda --batch 128 \
#      --model_conf config/model/ConvGRU_drop_MSTFT.yaml          \
#      --hyper_conf config/hyper/ConvGRU_VB_paste_real.yaml       \
#      --params exp/${title}/checkpoints/best_valid_score.pt"

# title=ConvGRU_MSTFT-VB_paste_real_v3
# cmd="--title ${title} --model ConvGRU --device cuda --batch 128 \
#      --model_conf config/model/ConvGRU_MSTFT.yaml               \
#      --hyper_conf config/hyper/ConvGRU_VB_paste_real_v3.yaml    \
#      --params exp/${title}/checkpoints/best_valid_score.pt"

# mkdir -p exp/${title}
# logfile=exp/${title}/history.log

# if [ -f ${logfile} ]
# then
#      echo "Experiment ${title} exists."
# else
#      python main_crnn.py --train ${cmd} > ${logfile}
# fi
# python main_crnn.py --test ${cmd}