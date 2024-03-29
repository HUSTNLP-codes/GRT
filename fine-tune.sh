dt=`date '+%Y%m%d_%H%M%S'`

dataset="csqa"
model='roberta-large'
#shift
#shift
args=$@


 echo "***** hyperparameters *****"
 echo "dataset: $dataset"
 echo "enc_name: $model"
 echo "batch_size: $bs"
 echo "learning_rate: elr $elr dlr $dlr"
 echo "edge_encoder_dim $enc_dim gsc_layer $k"
 echo "******************************"

save_dir_pref='experiments'
logs_dir_pref='logs/csqa/'
mkdir -p $save_dir_pref
mkdir -p $logs_dir_pref

#n_epochs=80

n_epochs=30
bs=128
mbs=8
ebs=4
enc_dim=32
ckpt_dir = "./saved_models/pretrain/model.pt.X"


#elr="1e-5"
#dlr="1e-2"

elr="6e-5"
dlr="1e-4"
weight_decay="1e-2"



dropout="1e-1"
dropoutf="1e-1"
drop_ratio="0.05"
k=2

tr_dim=1024
ffn_dim=2048
num_heads=16
lambda="10"

ent_emb=tzw
random_ent_emb=false

###### Training ######
for seed in 1; do
  python3 -u pre-train --dataset $dataset \
      --encoder $model -k $k \
      -elr $elr -dlr $dlr -bs $bs -mbs ${mbs} -ebs ${ebs} --weight_decay ${weight_decay} --seed 0 --checkpoint_dir $ckpt_dir\
      --n_epochs $n_epochs --max_epochs_before_stop 20  \
      --train_adj data/${dataset}/graph/train.graph.adj.ori2.metapath.2.q2a.seq.pk \
      --dev_adj data/${dataset}/graph/dev.graph.adj.ori2.metapath.2.q2a.seq.pk \
      --test_adj  data/${dataset}/graph/test.graph.adj.ori2.metapath.2.q2a.seq.pk\
      --train_statements  data/${dataset}/statement/train_with_triple.statement2.jsonl \
      --dev_statements  data/${dataset}/statement/dev_with_triple.statement.jsonl \
      --test_statements  data/${dataset}/statement/test_with_triple.statement.jsonl \
      --max_seq_len 200     \
      --num_relation 38    \
      --unfreeze_epoch 5 \
      --log_interval 10 \
      --transformer_dim ${tr_dim} \
      --transformer_ffn_dim ${ffn_dim} \
      --num_heads ${num_heads} \
      --dropouttr ${dropout} \
      --dropoutf ${dropoutf} \
      --lr_schedule "warmup_linear" \
      --save_model \
      --max_node_num 44 \
      --inverse_relation \
      --drop_ratio ${drop_ratio} \
      --lambda_rpe ${lambda} \
      --without_amp \
      --ent_emb ${ent_emb//,/ } \
  | tee -a $logs_dir_pref/newFT_path.${dataset}_${elr}.${dlr}.${weight_decay}.${dropout}_${tr_dim}.${ffn_dim}.${num_heads}__seed${seed}_${dt}.log.txt
done
