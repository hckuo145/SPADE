title=SEResNet34_base_MSTFT_ASP-HAD
# title=SEResNet34_base_MSTFT_ASP-HAD-utter_only
cmd="--title ${title} --model Spade --device cuda --batch 16  \
     --model_conf config/model/SEResNet34_base_MSTFT_ASP.yaml \
     --hyper_conf config/hyper/HAD.yaml                       \
     --params exp/${title}/checkpoints/best_valid_loss.pt"


# title=SEResNet34_base_MSTFT_ASP-Voicebox
# cmd="--title ${title} --model Spade --device cuda --batch 4   \
#      --model_conf config/model/SEResNet34_base_MSTFT_ASP.yaml \
#      --hyper_conf config/hyper/Voicebox.yaml                  \
#      --params exp/${title}/checkpoints/best_valid_loss.pt"


title=SEResNet34_base_MSTFT_ASP-Voicebox_v2
# title=SEResNet34_base_MSTFT_ASP-Voicebox_v2-utter_only
cmd="--title ${title} --model Spade --device cuda --batch 4   \
     --model_conf config/model/SEResNet34_base_MSTFT_ASP.yaml \
     --hyper_conf config/hyper/Voicebox_v2.yaml               \
     --params exp/${title}/checkpoints/best_valid_loss.pt"


mkdir -p exp/${title}
logfile=exp/${title}/history.log

if [ -f ${logfile} ]
then
     echo "Experiment ${title} exists."
else
     python main.py --train ${cmd} > ${logfile}
fi

# python main.py --test  ${cmd}