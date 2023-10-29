# FSE_team_1_project


Our project  is called ImageCaption. In it provide a repository for demonstrating the work of a trained neural network, which, based on pictures, generates their text description. The repository contains a neural network model, unit tests, main code for training and code for work demonstration. We use Docker for easy launching and distribution and unit tests for testing.

Our network is based on article https://arxiv.org/abs/1502.03044

Team members:

Alexandr Voronin 

Ksenia Lapshova 

Nikita Kornilov 

Pavel Bartenev 

Yulia Sergeeva


In order to run and develop our network:

1) Clone represitory and go to its directory
   
2) Create docker from DockerFile with command
   
   `docker build -t text_network_image . -f Dockerfile.txt`

3) Run docker image and share directory with images

   `docker run --name network_container  -it  -v $(pwd)/Images:/app/FSE_team_1_project/Images text_network_image`
   
4) For demo on jpg image from your system, put it in folder `Images`  and then run python command in docker(instead `Images/man.jpg` one can put `Images/your_image_name.jpg`)
   
   `python inference.py Images/man.jpg Data/captions_tokenized.json Data/CaptionNetBest2.pth`

5) To test solution run python command in docker
   
   `python tests.py`  

6) You can also install all required packages using prereqs.sh script, compile files using build.sh script and run tests with build.sh 
   
