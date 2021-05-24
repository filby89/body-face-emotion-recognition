# Fusing Body Posture With Facial Expressions for Joint Recognition of Affect in Child–Robot Interaction

PyTorch code for the paper [Fusing Body Posture With Facial Expressions for Joint Recognition of Affect in Child–Robot Interaction](https://ieeexplore.ieee.org/abstract/document/8769871).

You can find the preprint at [arXiv](https://arxiv.org/abs/1901.01805).

### Preparation
* Download the [BRED dataset](https://zenodo.org/record/3233060) and extract it inside the project.
* Create a directory "saved_scores" to save the outputs in numpy format.
* Note that there is an error in the annotations.csv file. Replace "spontaneous" in the path with "game" and "acted" with "pre-game" to get the correct paths.

### Training


Train a model using only the skeleton (SEP):

>  python main.py --db babyrobot --epochs 200 --step_size 150 --add_body_dnn --num_classes 7 --num_total_iterations=1 --exp_name "BODY-ONLY" --optimizer sgd --weight_decay 1e-3 --lr 1e-1 --batch_size 12 --use_labels body



Train a model using only the CNN features (SEP):

> python main.py --db babyrobot --epochs 200 --step_size 150 --use_cnn_features --num_classes 7 --num_total_iterations=1 --exp_name "FACE_ONLY" --optimizer sgd --weight_decay 1e-3 --lr 1e-1 --batch_size 12 --use_labels face



Combine skeleton and CNN features (feature fusion - Joint-1L)

> python main.py --db babyrobot --epochs 200 --step_size 150 --add_body_dnn --use_cnn_features --num_classes 7 --num_total_iterations=1 --exp_name "JOINT" --optimizer sgd --weight_decay 1e-3 --lr 1e-1 --batch_size 12


Combine skeleton and CNN features (Hierarchical training - HMT-3A)

>  python main.py --db babyrobot --epochs 200 --step_size 150 --add_body_dnn --use_cnn_features --num_classes 7 --num_total_iterations=1 --optimizer sgd --weight_decay 1e-3 --lr 1e-1 --batch_size 12 --split_branches --do_fusion --exp_name "HMT-3a"


Combine skeleton and CNN features (Hierarchical training - HMT-3B)

>  python main.py --db babyrobot --epochs 200 --step_size 150 --add_body_dnn --use_cnn_features --num_classes 7 --num_total_iterations=1 --optimizer sgd --weight_decay 1e-3 --lr 1e-1 --batch_size 12 --split_branches --add_whole_body_branch --exp_name "HMT-3b"


Combine skeleton and CNN features (Hierarchical training - HMT-4) adding final fusion branch (HMT-4)

> python main.py --db babyrobot --epochs 200 --step_size 150 --add_body_dnn --use_cnn_features --num_classes 7 --num_total_iterations=1 --optimizer sgd --weight_decay 1e-3 --lr 1e-1 --batch_size 12 --split_branches --do_fusion --add_whole_body_branch --exp_name "ΗΜΤ-4"




## Citation
If you use this code for your research, consider citing our paper.
```
@ARTICLE{8769871,  
	author={P. P. {Filntisis} and N. {Efthymiou} and P. {Koutras} and G. {Potamianos} and P. {Maragos}},  
	journal={IEEE Robotics and Automation Letters},   
	title={Fusing Body Posture With Facial Expressions for Joint Recognition of Affect in Child–Robot Interaction},   year={2019},  
	volume={4},  
	number={4},  
	pages={4011-4018},  
	doi={10.1109/LRA.2019.2930434}}
```


### Contact 
For questions feel free to open an issue.
