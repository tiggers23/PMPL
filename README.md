# Progressive Multimodal Pivot Learning: Towards Semantic Discordance Understanding as Humans
This repo contains the code for the PMPL research project, which focuses on improving the senmantic discordance understanding of existing multimodal recognition models. To perform semantic discordance understanding as humans, we propose a novel multimodal learning paradigm, namely multimodal pivot learning paradigm. This code is based on the [PMF](https://github.com/yaoweilee/PMF) implementation.

## Dataset
### Download
In this paper, we apply three dataset: Twitter-15/17, CrisisMMD, and MM-IMDB. Among this datasets, Twitter-15/17 is a re-annotated version with the unimodal labels by Chen et al., which can be found in [HFIR](https://github.com/code-chendl/HFIR), while you can access the data of CrsisMMD in [crisismmd](https://crisisnlp.qcri.org/crisismmd) (to be noticed, we use the version is crisismmdv1.0). Then, MM-IMDB can be downloaded in [mmimdb](http://lisi1.unal.edu.co/mmimdb/mmimdb.tar.gz).
### pre-process
To quickly start our project, we recommond you pre-process the text file of download data into one '.json' file for each dataset as the below fromation:
''' 
{'id': image_file_path, 'text': text_content, 'text_label': text_label, 'image_label': image_label, 'label': overall_label}
'''



## Training

python PMPL_main.py --seed #seed --beta=#beta --lr=#lr --prompt_length=#prompt_length --n_fusion_layers=#fusion_layer --batch_size=64 --class_num=#class_of_dataset --config=./configs --dataset=#dataset --dev_dataset=dev.txt --device=cuda:1 --file_path=#output_file --test_dataset=test.txt --train_dataset=train.txt --type=train
