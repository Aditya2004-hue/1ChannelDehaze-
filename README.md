# 1ChannelDehaze-
Gray Scale Dehazing 

 Step 1 - Create & activate virtual environment
- `python -m venv venv`
- `.\venv\Scripts\activate`

 Step 2 - Install dependencies
- `pip install -r requirements.txt`

 Step 3 - Convert GT and hazy images to .npy
- `python dataset.py`

 Step 4 - Train the RGB model
- `python train.py`

 Step 5 - Evaluate the model and save results
- `python evaluate.py`
