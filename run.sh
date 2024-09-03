title=NaiveFCN_Wav2Vec2-ft_ASP__SINE_edit_resyn
cmd="--title ${title} --model NaiveFCN --device cuda --batch 16 \
     --model_conf config/model/NaiveFCN_Wav2Vec2-ft_ASP.yaml    \
     --hyper_conf config/hyper/Spade_SINE_edit_resyn.yaml       \
     --params exp/${title}/checkpoints/best_valid_score.pt"


# mkdir -p exp/${title}
# logfile=exp/${title}/history.log

# if [ -f ${logfile} ]
# then
#      echo "Experiment ${title} exists."
# else
#      python main_spade.py --train ${cmd} > ${logfile}
# fi
python main_spade.py --test  ${cmd}