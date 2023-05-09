python main.py \
--hoi \
--dataset_file hico \
--hoi_path work/hico_20160224_det \
--num_obj_classes 80 \
--num_verb_classes 117 \
--backbone resnet50 \
--set_cost_bbox 2.5 \
--set_cost_giou 1 \
--bbox_loss_coef 2.5 \
--giou_loss_coef 1 \
--output_dir outputs/hico/ts_model/ \
--model_name hoi_ts \
--epochs 80 \
--lr_drop 60 \
--num_workers 1

# --pretrained params/detr-r50-pre.pth \


# python -m torch.distributed.launch \
# --nproc_per_node=1 \
# --use_env \
# main.py \
# --pretrained params/detr-r50-pre.pth \
# --hoi \
# --dataset_file hico \
# --hoi_path data/hico_20160224_det \
# --num_obj_classes 80 \
# --num_verb_classes 117 \
# --backbone resnet50 \
# --set_cost_bbox 2.5 \
# --set_cost_giou 1 \
# --bbox_loss_coef 2.5 \
# --giou_loss_coef 1 \
# --output_dir outputs/hico/ts_model/ \
# --model_name hoi_ts \
# --epochs 80 \
# --lr_drop 60



# conda install cudatoolkit=10.2 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/linux-64/

