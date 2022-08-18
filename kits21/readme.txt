##Course code and title: COMP5200M | MSc project

##Environment
-Python version: Python 3.9

-Cuda version: Cuda 11.3

-Pytorch version: 1.8.0

-Numpy version: 1.16.0


##Type of project: Empirical Investigation (EI) projects   


##Run Configuration: GPU/CPU


##Python packages:
Packages are shown in the front of each python files.


##Parameters of the Network
-Type of Network: 2D U-Net
-n_channels=1, n_classes=4, bilinear=args.bilinear
-             epochs: int = 20,
              batch_size: int = 1,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False




##The structure of the system:

1. train.py: Train the data
2. readnpy.py: transfer npy to png
3.process_data.py: processing the data
4.predict.py : Predicting
5.precision.py : Accuracy Dice scores
6.generate_mhd.py : Generate mhd images




##Checkpoints are the models that trained by train.py [Checkpoint 1/2/3/4]



##The output of the precision
                     The Dice Score Calculated by HECs

Models	  Kidney and Masses          Kidney Mass          Tumour
       (Kidney + Tumour + Cyst)	   (Tumour + Cyst)       (Tumour)

Model 1	     0.531797	               0.365912	            0.702381
Model 2	     0.588634	               0.447516	            0.759195
Model 3	     0.598449	               0.460400	            0.703089
Model 4	     0.596160	               0.455582	            0.698275




##The program didn't submit to the KiTS  Challenge website.


#train.py reference: https://blog.csdn.net/CQUSongYuxin/article/details/108342591
