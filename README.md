# Progressive Multimodal Pivot Learning: Towards Semantic Discordance Understanding as Humans

## Training

python mis_prompt.py --seed #seed --beta=#beta --lr=#lr --prompt_length=#prompt_length --n_fusion_layers=#fusion_layer --batch_size=64 --class_num=#class_of_dataset --config=./configs --dataset=#dataset --dev_dataset=dev.txt --device=cuda:1 --file_path=#output_file --test_dataset=test.txt --train_dataset=train.txt --type=train