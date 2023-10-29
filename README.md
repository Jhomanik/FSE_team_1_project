# FSE_team_1_project


Our project  is called ImageCaption. In it we will provide a repository for demonstrating the work of a trained neural network, which, based on pictures, generates their text description. The repository will contain a neural network model, unit tests, main code for training and code for work demonstration. We use Docker for easy launching and distribution and unit tests for testing.

Our network will be based on article https://arxiv.org/abs/1502.03044

Team members:

Alexandr Voronin 

Ksenia Lapshova 

Nikita Kornilov 

Pavel Bartenev 

Yulia Sergeeva


In order to run and develop our network:
1) Create docker from DockerFile with command
   
`docker build -t FSE_group_1 . -f Dockerfile`

2) Run docker image and share directory images

   docker run --name pytorch-container  -it  -v $(pwd):/app/FSE_team_1_project FSE_group_1
   
3) For demo on jpg  image in folder Image run python command in docker
   
   python inference.py Images/image.jpg Data/captions_tokenized.json Data/CaptionNetBest2.pth

4) To test solution run python command in docker
   
   python tests.py  

5) You can also install all required packages using prereqs.sh script, compile files using build.sh script and run tests with build.sh 
   
