# title=SEResNet34_base_MSTFT_ASP-HAD
# cmd="--title ${title} --model Spade --device cuda --batch 48  \
#      --model_conf config/model/SEResNet34_base_MSTFT_ASP.yaml \
#      --hyper_conf config/hyper/Spade_HAD.yaml                 \
#      --params exp/${title}/checkpoints/best_valid_loss.pt"

# title=SEResNet34_base_MSTFT_ASP-Voicebox
# cmd="--title ${title} --model Spade --device cuda --batch 4   \
#      --model_conf config/model/SEResNet34_base_MSTFT_ASP.yaml \
#      --hyper_conf config/hyper/Spade_VB_edit.yaml             \
#      --params exp/${title}/checkpoints/best_valid_loss.pt"

# title=SEResNet34_base_MSTFT_token-Voicebox
# cmd="--title ${title} --model Spade_v2 --device cuda --batch 4  \
#      --model_conf config/model/SEResNet34_base_MSTFT_token.yaml \
#      --hyper_conf config/hyper/Spade_VB_edit.yaml               \
#      --params exp/${title}/checkpoints/best_valid_loss.pt"

# title=NaiveFCN_WavLM_ASP-HAD
# cmd="--title ${title} --model NaiveFCN --device cuda --batch 4 \
#      --model_conf config/model/NaiveFCN_WavLM_ASP.yaml         \
#      --hyper_conf config/hyper/Spade_HAD.yaml                  \
#      --params exp/${title}/checkpoints/best_valid_loss.pt"


# mkdir -p exp/${title}
# logfile=exp/${title}/history.log

# if [ -f ${logfile} ]
# then
#      echo "Experiment ${title} exists."
# else
#      python main_spade.py --train ${cmd} > ${logfile}
# fi

# python main_spade.py --test  ${cmd}





# title=Transsion_MSTFT-HAD
# cmd="--title ${title} --model Transsion --device cuda --batch 48 \
#      --model_conf config/model/Transsion_MSTFT.yaml              \
#      --hyper_conf config/hyper/Transsion_HAD.yaml                \
#      --params exp/${title}/checkpoints/best_valid_loss.pt"

# mkdir -p exp/${title}
# logfile=exp/${title}/history.log

# if [ -f ${logfile} ]
# then
#      echo "Experiment ${title} exists."
# else
#      python main_trans.py --train ${cmd} > ${logfile}
# fi

# python main_trans.py --test ${cmd}





title=ConvGRU_MSTFT-HAD
cmd="--title ${title} --model ConvGRU --device cuda --batch 48 \
     --model_conf config/model/ConvGRU_MSTFT.yaml              \
     --hyper_conf config/hyper/ConvGRU_HAD.yaml                \
     --params exp/${title}/checkpoints/best_valid_loss.pt"

mkdir -p exp/${title}
logfile=exp/${title}/history.log

if [ -f ${logfile} ]
then
     echo "Experiment ${title} exists."
else
     python main_crnn.py --train ${cmd} > ${logfile}
fi

# python main_crnn.py --test ${cmd}